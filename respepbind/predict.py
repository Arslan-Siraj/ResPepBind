#!/usr/bin/env python3
"""ResPepBind 10-fold CPU/GPU predictions for peptide-protein pairs, pair FASTA or CSV.

Embeddings are resolved by sequence hash (safe across datasets), and missing
embeddings are generated with the ProtT5 encoder used by the supplied training
pipeline. No training occurs. Check README for provenance/CPU performance notes.
"""
from __future__ import annotations

import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Architecture bundled with the installed Python package.
from .model import EmbeddingStore, build_model, collate_batch, compute_metrics, _extract_label
from .weights import DEFAULT_MODEL_DIR, ensure_installed
DEFAULT_MODEL = 'Rostlab/prot_t5_xl_uniref50'
AA = set('ACDEFGHIKLMNPQRSTVWYX')


def normalized_sequence(value: str, field: str) -> str:
    sequence = re.sub(r'\s+', '', str(value)).upper()
    sequence = re.sub('[UZOB]', 'X', sequence)
    if not sequence or set(sequence) - AA:
        chars = ''.join(sorted(set(sequence) - AA))
        raise ValueError(f'{field} must be a nonempty amino-acid sequence; unsupported characters: {chars!r}')
    return sequence


def maybe_label(value, context):
    if value is None or str(value).strip() == '':
        return None
    text = str(value).strip()
    if text in ('0', '1', '0.0', '1.0'):
        return int(float(text))
    raise ValueError(f'{context}: label must be 0, 1, or empty; received {value!r}')


@dataclass
class Pair:
    record_id: str
    peptide: str
    protein: str
    label: int | None


def load_fasta(path: Path) -> list[Pair]:
    """FASTA with one >header and Peptide:/Protein: fields per pair, wraps accepted."""
    pairs = []
    header = None
    pep_parts, prot_parts = [], []
    active = None

    def flush():
        if header is None:
            return
        if not pep_parts or not prot_parts:
            raise ValueError(f'FASTA record {header!r} must have Peptide: and Protein: lines')
        label = _extract_label(header)
        pairs.append(Pair(header, normalized_sequence(''.join(pep_parts), 'Peptide'),
                          normalized_sequence(''.join(prot_parts), 'Protein'), label))

    with path.open(encoding='utf-8-sig') as fh:
        for line_number, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                flush()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f'Empty FASTA header at line {line_number}')
                pep_parts, prot_parts, active = [], [], None
                continue
            if header is None:
                raise ValueError(f'FASTA sequence encountered before header (line {line_number})')
            matched = re.match(r'(?i)^(peptide|pep|protein|prot)\s*:\s*(.*)$', line)
            if matched:
                active = 'pep' if matched.group(1).lower() in ('peptide', 'pep') else 'prot'
                line = matched.group(2).strip()
            if active == 'pep':
                pep_parts.append(line)
            elif active == 'prot':
                prot_parts.append(line)
            else:
                raise ValueError(f'Expected Peptide: or Protein: at line {line_number}')
    flush()
    if not pairs:
        raise ValueError('FASTA file contains no peptide/protein records')
    return pairs


def load_csv(path: Path) -> list[Pair]:
    pairs = []
    with path.open(newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError('CSV has no header')
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        def col(*options):
            return next((columns[k] for k in options if k in columns), None)
        peptide_col = col('peptide', 'peptide_sequence', 'pep', 'seq1')
        protein_col = col('protein', 'protein_sequence', 'prot', 'seq2')
        id_col = col('id', 'record_id', 'pair_id')
        label_col = col('true_label', 'label', 'y', 'target')
        if not peptide_col or not protein_col:
            raise ValueError('CSV must contain peptide/protein or peptide_sequence/protein_sequence columns')
        for n, row in enumerate(reader, 1):
            record_id = (row.get(id_col) if id_col else None) or f'pair_{n}'
            record_id = str(record_id).strip()
            label = maybe_label(row.get(label_col) if label_col else None, f'CSV row {n}')
            pairs.append(Pair(record_id,
                              normalized_sequence(row.get(peptide_col, ''), f'CSV row {n} peptide'),
                              normalized_sequence(row.get(protein_col, ''), f'CSV row {n} protein'), label))
    if not pairs:
        raise ValueError('CSV file contains no pairs')
    return pairs


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('ascii')).hexdigest()


def torch_load_trusted(path):
    # Prefer PyTorch's restricted deserializer for released model weights and caches.
    # Legacy PyTorch versions may not support weights_only; use trusted files only.
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')
    except Exception as exc:
        raise RuntimeError(
            f'Unable to safely load {path}. Use an official, inference-only ResPepBind '
            'checkpoint or a trusted tensor cache; do not load untrusted pickle files.'
        ) from exc


def find_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Missing 10-fold model directory: {run_dir}')
    paths = []
    for n in range(1, 11):
        folder = run_dir / f'fold{n}'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing fold{n} in {run_dir}')
        found = list(folder.glob('*.pt'))
        if len(found) != 1:
            raise ValueError(f'Expected exactly one .pt checkpoint in {folder}, found {len(found)}')
        paths.append(found[0])
    return paths


def read_threshold(run_dir: Path, override: float | None) -> float:
    if override is not None:
        threshold = override
    else:
        report = run_dir / 'ensemble_independent_test_metrics.csv'
        if not report.is_file():
            raise FileNotFoundError(f'Missing training-derived threshold report: {report}. Pass --threshold explicitly.')
        with report.open(newline='') as fh:
            rows = list(csv.DictReader(fh))
        vals = {float(r['threshold']) for r in rows if r.get('ensemble_threshold_strategy', '').strip() == 'oof_youden'}
        if len(vals) != 1:
            raise ValueError(f'Expected one oof_youden threshold in {report}, got {sorted(vals)}. Pass --threshold.')
        threshold = vals.pop()
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f'Threshold must be between 0 and 1: {threshold}')
    return float(threshold)


class ProtT5Encoder:
    """Match residue extraction in original pepri_hgt_training.py on CPU/GPU."""
    def __init__(self, model_name: str, chunk_size: int, overlap: int, device: torch.device):
        if not 0 <= overlap < chunk_size:
            raise ValueError('Require 0 <= chunk_overlap < chunk_size')
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = device
        print(f'[ProtT5] Loading {model_name} on {device} (first uncached sequence)...', flush=True)
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise RuntimeError('Missing ProtT5 dependencies. Install transformers and sentencepiece; see README.') from exc
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name).eval().to(device)
        # Original ProtT5 extraction used half precision on CUDA, float32 on CPU.
        if device.type == 'cuda':
            self.model.half()
        else:
            self.model.float()
        print(f'[ProtT5] Model ready on {device}', flush=True)

    @torch.inference_mode()
    def _one_chunk(self, seq: str) -> torch.Tensor:
        encoded = self.tokenizer(' '.join(seq), add_special_tokens=True,
                                 padding=False, return_tensors='pt')
        out = self.model(input_ids=encoded['input_ids'].to(self.device),
                         attention_mask=encoded['attention_mask'].to(self.device))
        emb = out.last_hidden_state[0, :len(seq)].float().cpu().contiguous()
        if emb.shape[0] != len(seq) or emb.shape[-1] != 1024:
            raise ValueError(f'Unexpected ProtT5 embedding shape: {list(emb.shape)} for {len(seq)} residues')
        return emb

    def embed(self, seq: str) -> torch.Tensor:
        n = len(seq)
        if n <= self.chunk_size:
            return self._one_chunk(seq)
        total = torch.zeros(n, 1024, dtype=torch.float32)
        counts = torch.zeros(n, 1, dtype=torch.float32)
        step = self.chunk_size - self.overlap
        for start in range(0, n, step):
            end = min(start + self.chunk_size, n)
            total[start:end] += self._one_chunk(seq[start:end])
            counts[start:end] += 1
            if end == n:
                break
        return (total / counts.clamp_min(1)).contiguous()


class EmbeddingResolver:
    def __init__(self, model: str, cache: Path, existing: Path | None,
                 trust_id_cache: bool, generate_missing: bool, chunk_size: int,
                 overlap: int, device: torch.device):
        slug = re.sub('[^a-zA-Z0-9_-]', '_', model)[:60]
        precision = 'fp16' if device.type == 'cuda' else 'fp32'
        model_key = sha1(f'{model}|chunk={chunk_size}|overlap={overlap}|{precision}')[:12]
        self.cache = cache / f'{slug}_{model_key}'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.existing = EmbeddingStore(existing) if existing is not None else None
        self.trust_id_cache = trust_id_cache
        self.generate_missing = generate_missing
        self.model_name = model
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.plm = None
        self.memo = {}
        self.sources = {}
        if trust_id_cache:
            print('[WARNING] --trust-id-cache allows ID-only lookup. Reused IDs across datasets can return incorrect embeddings.', file=sys.stderr)

    def _accept(self, tensor, seq, input_dim, source):
        if tensor.ndim != 2 or tensor.shape[1] != input_dim or not 1 <= tensor.shape[0] <= len(seq):
            raise ValueError(f'Embedding from {source}: got shape {tuple(tensor.shape)}, expected [L,{input_dim}], 1<=L<={len(seq)}')
        return tensor.detach().float().cpu().contiguous()

    def get(self, seq: str, pair_id: str, molecule: str, input_dim: int):
        h = sha1(seq)
        if h in self.memo:
            return self.memo[h], self.sources[h]
        cached = self.cache / f'{h}.pt'
        if cached.is_file():
            tensor = self._accept(torch_load_trusted(cached), seq, input_dim, str(cached))
            source = 'generated_cache'
        else:
            tensor = None
            source = ''
            if self.existing is not None:
                # Default: only sequence-derived keys, never untrusted record names.
                # The original EmbeddingStore understands sha1, md5 and seq_<hash>.
                try:
                    tensor = self._accept(self.existing.get(h, seq), seq, input_dim, f'existing sequence hash {h}')
                    source = 'existing_sequence_hash'
                except FileNotFoundError:
                    pass
                if tensor is None and self.trust_id_cache:
                    identifier = f'{pair_id}_{"pep" if molecule == "peptide" else "prot"}'
                    try:
                        tensor = self._accept(self.existing.get(identifier, seq), seq, input_dim, f'existing ID {identifier}')
                        source = 'existing_unverified_pair_id'
                    except FileNotFoundError:
                        pass
            if tensor is None:
                if not self.generate_missing:
                    raise FileNotFoundError(f'No embedding for {molecule} sequence SHA1={h}; use --auto-embed (default), or provide compatible embedding cache')
                if self.plm is None:
                    self.plm = ProtT5Encoder(self.model_name, self.chunk_size, self.overlap, self.device)
                print(f'[ProtT5] Generating {molecule} embedding: residues={len(seq)} sha1={h[:12]}', flush=True)
                tensor = self._accept(self.plm.embed(seq), seq, input_dim, 'new ProtT5 embedding')
                # Save to dedicated model-specific local cache, not the training embedding directory.
                temp = cached.with_suffix('.partial')
                torch.save(tensor, temp)
                temp.replace(cached)
                source = f'new_prott5_{self.device.type}'
        self.memo[h] = tensor
        self.sources[h] = source
        return tensor, source


class ResolvedPairs(Dataset):
    def __init__(self, pairs: list[Pair], resolver: EmbeddingResolver, pep_len: int, prot_len: int, input_dim: int):
        self.pairs = pairs
        self.embeddings = []
        for i, pair in enumerate(pairs, 1):
            pep, p_source = resolver.get(pair.peptide, pair.record_id, 'peptide', input_dim)
            prot, q_source = resolver.get(pair.protein, pair.record_id, 'protein', input_dim)
            self.embeddings.append((pep[:pep_len], prot[:prot_len], p_source, q_source))
            if i == 1 or i % 25 == 0 or i == len(pairs):
                print(f'[Embeddings] Resolved {i}/{len(pairs)} pairs', flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        pep, prot, _, _ = self.embeddings[index]
        return dict(pep=pep, prot=prot,
                    label=torch.tensor(float(pair.label or 0), dtype=torch.float32),
                    peptide_id=pair.record_id+'_pep', protein_id=pair.record_id+'_prot',
                    raw_header=pair.record_id)


def predict(args):
    torch.set_num_threads(args.threads)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu or check NVIDIA/PyTorch installation')
        device = torch.device(args.device)
        # Fail early for an invalid CUDA index.
        torch.cuda.get_device_properties(device)
    else:
        device = torch.device('cpu')
    print(f'[DEVICE] Classifier and uncached ProtT5 embeddings: {device}', flush=True)
    print(f'[MODEL] Model implementation: {sys.modules["respepbind.model"].__file__}', flush=True)
    # Limits also set in run_cpu.sh; no CUDA used by design.
    if args.run_dir is None:
        args.run_dir = ensure_installed(DEFAULT_MODEL_DIR, allow_download=not args.no_download_weights)
    else:
        if not args.run_dir.exists():
            raise FileNotFoundError(f'Custom --run-dir does not exist: {args.run_dir}. Download checkpoints or omit --run-dir for automatic installation.')
    paths = find_checkpoints(args.run_dir)
    checkpoints = [torch_load_trusted(p) for p in paths]
    configs = [SimpleNamespace(**ckpt['args']) for ckpt in checkpoints]
    fields = ('variant', 'interaction_features', 'hidden_dim', 'mlp_blocks', 'max_peptide_len',
              'max_protein_len', 'dropout', 'input_dropout', 'block_dropouts',
              'head_dropout', 'mlp_expand_ratio', 'head_ratio', 'activation', 'norm_type', 'residual_scale')
    for n, (ckpt, cfg) in enumerate(zip(checkpoints[1:], configs[1:]), 2):
        for k in fields:
            if getattr(cfg, k, None) != getattr(configs[0], k, None):
                raise ValueError(f'fold{n} mismatched configuration: {k}')
        if int(ckpt['input_dim']) != int(checkpoints[0]['input_dim']):
            raise ValueError(f'fold{n} embedding dimension mismatch')
    input_dim = int(checkpoints[0]['input_dim'])
    if input_dim != 1024:
        raise ValueError(f'Automatic ProtT5 encoder produces 1024 dimensions, checkpoint expects {input_dim}')
    pep_len, prot_len = int(configs[0].max_peptide_len), int(configs[0].max_protein_len)
    threshold = read_threshold(args.run_dir, args.threshold)

    if args.fasta:
        pairs = load_fasta(args.fasta)
    elif args.csv:
        pairs = load_csv(args.csv)
    else:
        pairs = [Pair(args.id, normalized_sequence(args.peptide, 'Peptide'),
                      normalized_sequence(args.protein, 'Protein'), None)]
    print(f'[INFERENCE] Pairs={len(pairs)}; folds=10; input_dim={input_dim}; max lengths={pep_len}/{prot_len}; threshold={threshold:.6f}', flush=True)
    print('[INFERENCE] ProtT5 is loaded only if an embedding is missing.', flush=True)

    resolver = EmbeddingResolver(args.encoder_model, args.cache_dir, args.embedding_dir,
                                 args.trust_id_cache, not args.existing_only,
                                 args.chunk_size, args.chunk_overlap, device)
    dataset = ResolvedPairs(pairs, resolver, pep_len, prot_len, input_dim)
    # Free the large encoder before running the 10 small classifier checkpoints.
    resolver.plm = None
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_batch)
    predictions = []
    for i, (path, ckpt, cfg) in enumerate(zip(paths, checkpoints, configs), 1):
        model = build_model(cfg, input_dim).to(device).eval()
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        arr = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch['pep'].to(device), batch['prot'].to(device),
                               batch['pep_mask'].to(device), batch['prot_mask'].to(device))
                arr.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist())
        if len(arr) != len(pairs):
            raise RuntimeError('Internal mismatch in prediction count')
        predictions.append(np.asarray(arr, dtype=np.float64))
        print(f'[INFERENCE] fold{i}/10 predicted {len(arr)} pairs', flush=True)
        del model
    mean = np.mean(np.stack(predictions), axis=0)
    output = args.output_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['id', 'true_label', 'peptide', 'protein', 'peptide_length', 'protein_length',
                  'interaction_probability', 'predicted_label', 'interaction_result', 'threshold',
                  'peptide_embedding_source', 'protein_embedding_source'] + [f'probability_fold{i}' for i in range(1, 11)]
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k, (pair, score) in enumerate(zip(pairs, mean)):
            label = int(score >= threshold)
            _, _, psource, qsource = dataset.embeddings[k]
            record = dict(id=pair.record_id, true_label='' if pair.label is None else pair.label,
                          peptide=pair.peptide, protein=pair.protein,
                          peptide_length=len(pair.peptide), protein_length=len(pair.protein),
                          interaction_probability=float(score), predicted_label=label,
                          interaction_result='Interaction' if label else 'No Interaction',
                          threshold=threshold, peptide_embedding_source=psource,
                          protein_embedding_source=qsource)
            record.update({f'probability_fold{i+1}': float(predictions[i][k]) for i in range(10)})
            writer.writerow(record)
    print(f'[DONE] Saved {len(pairs)} predictions to {output}', flush=True)

    labels = [pair.label for pair in pairs]
    if all(y is not None for y in labels) and len(set(labels)) == 2:
        metrics = compute_metrics(labels, mean, threshold)
        metrics['n_samples'] = len(pairs)
        metrics_path = output.with_suffix('.metrics.json')
        metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
        print(f'[METRICS] {json.dumps(metrics, indent=2)}', flush=True)
        print(f'[DONE] Saved {metrics_path}', flush=True)
    else:
        print('[NOTE] Metrics skipped: labels are missing or there is only one class.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--peptide', help='One peptide sequence (requires --protein)')
    source.add_argument('--fasta', type=Path, help='Pair FASTA with Peptide: and Protein: fields')
    source.add_argument('--csv', type=Path, help='CSV with peptide/protein columns')
    parser.add_argument('--protein', help='One protein sequence (requires --peptide)')
    parser.add_argument('--id', default='pair_1', help='Identifier for a direct sequence pair')
    parser.add_argument('--run-dir', type=Path, default=None, help='Optional local 10-fold model directory; omit to auto-download/cached official seed-2026 weights')
    parser.add_argument('--no-download-weights', action='store_true', help='Disable first-run weights download (use preinstalled cache or --run-dir)')
    parser.add_argument('--output-csv', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--embedding-dir', type=Path, default=None, help='Optional existing precomputed embedding directory')
    parser.add_argument('--cache-dir', type=Path, default=Path.home() / '.cache' / 'respepbind' / 'embeddings', help='Reusable generated ProtT5 embeddings')
    parser.add_argument('--encoder-model', default=DEFAULT_MODEL, help='Exact ProtT5 encoder model or local path')
    parser.add_argument('--existing-only', action='store_true', help='Never load ProtT5; fail if sequence embeddings missing')
    parser.add_argument('--trust-id-cache', action='store_true', help='Allow matching pair IDs in existing embeddings (risky for reused IDs)')
    parser.add_argument('--threshold', type=float, help='Override training-derived OOF Youden threshold')
    parser.add_argument('--chunk-size', type=int, default=1022, help='ProtT5 encoder residues per chunk')
    parser.add_argument('--chunk-overlap', type=int, default=128, help='Overlapping residues between chunks')
    parser.add_argument('--batch-size', type=int, default=32, help='Classifier inference batch size')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, cuda:0, cuda:1 or auto')
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU compute threads')
    args = parser.parse_args()
    if bool(args.peptide) != bool(args.protein):
        parser.error('--peptide and --protein must be passed together')
    if args.device != 'auto' and args.device != 'cpu' and not re.fullmatch(r'cuda(?::[0-9]+)?',args.device):
        parser.error('--device must be cpu, auto, cuda, or cuda:N')
    if args.threads < 1 or args.batch_size < 1 or args.chunk_size < 1:
        parser.error('threads, batch-size and chunk-size must be positive')
    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error('chunk-overlap must be >=0 and smaller than chunk-size')
    if args.embedding_dir is not None and not args.embedding_dir.exists():
        parser.error(f'Embedding directory does not exist: {args.embedding_dir}')
    predict(args)


if __name__ == '__main__':
    main()
