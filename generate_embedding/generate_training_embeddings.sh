#!/usr/bin/env bash
set -euo pipefail

# ResPepBind: generate ProtT5 residue embeddings for the TRAINING dataset.
# Place this file in ResPepBind/generate_embedding/.
# Run: ./generate_training_embeddings.sh
# GPU selector example: CUDA_VISIBLE_DEVICES=0 ./generate_training_embeddings.sh
# Uses the same extraction settings as generate_all_independent_embeddings.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXTRACTOR="$SCRIPT_DIR/pepri_hgt_training.py"
INPUT="$PROJECT_DIR/data/training/Training.fasta"
EMBEDDING_DIR="$SCRIPT_DIR/embeddings"
SPLIT="Training"
OUTPUT_DIR="$EMBEDDING_DIR/$SPLIT"
LOG_DIR="$SCRIPT_DIR/embedding_logs"
LOG="$LOG_DIR/Training_embedding.log"
MANIFEST="$LOG_DIR/Training_embedding_manifest.json"

[[ -f "$EXTRACTOR" ]] || { echo "ERROR: Extractor not found: $EXTRACTOR" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "ERROR: Training FASTA not found: $INPUT" >&2; exit 1; }
mkdir -p "$EMBEDDING_DIR" "$LOG_DIR"

# Match the independent extraction script: use the GPU + default FP16.
python - <<'PY'
import torch, transformers
from transformers import T5EncoderModel, T5Tokenizer
print(f'[ENV] Python/PyTorch={torch.__version__}; transformers={transformers.__version__}')
print(f'[ENV] CUDA available={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise SystemExit('ERROR: No CUDA GPU available. Refusing CPU extraction so precision matches the independent runs.')
print(f'[ENV] GPU={torch.cuda.get_device_name(0)}')
PY

printf '\n=========================================================\n'
printf 'ResPepBind Training — ProtT5 embedding extraction\n'
printf 'FASTA:      %s\n' "$INPUT"
printf 'Extractor:  %s\n' "$EXTRACTOR"
printf 'Output:     %s\n' "$OUTPUT_DIR"
printf 'Log:        %s\n' "$LOG"
printf 'Encoder:    Rostlab/prot_t5_xl_uniref50\n'
printf 'Settings:   protein=5000 peptide=50 chunk=1022 overlap=128 GPU=FP16\n'
printf '=========================================================\n\n'

# The extractor skips a pair only when BOTH output files already exist.
# If extraction was interrupted, simply rerun this script to resume.
python "$EXTRACTOR" \
    --extract_embeddings \
    --extract_fasta "$INPUT" \
    --extract_split_name "$SPLIT" \
    --embedding_dir "$EMBEDDING_DIR" \
    --max_protein_len 5000 \
    --max_peptide_len 50 \
    --prott5_chunk_size 1022 \
    --prott5_chunk_overlap 128 \
    2>&1 | tee "$LOG"

# Verify that EVERY FASTA record has nonempty peptide and protein embeddings.
# Save a small machine-readable manifest for future reproducibility checks.
python - "$INPUT" "$OUTPUT_DIR" "$EXTRACTOR" "$MANIFEST" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sys
import torch, transformers

fasta, output, extractor, manifest = map(Path, sys.argv[1:])
identifiers = [line[1:].strip() for line in fasta.open(encoding='utf-8') if line.startswith('>')]
if not identifiers:
    raise SystemExit('ERROR: FASTA contains no records')
if len(set(identifiers)) != len(identifiers):
    raise SystemExit('ERROR: Duplicate pair IDs within Training FASTA')

missing = []
for pair_id in identifiers:
    for molecule in ('peptide', 'protein'):
        file = output / f'{pair_id}_{molecule}.pt'
        if not file.is_file() or file.stat().st_size == 0:
            missing.append(str(file))

print(f'[VERIFY] Training: {len(identifiers):,} pairs; {2*len(identifiers):,} expected embeddings; missing/empty={len(missing)}')
if missing:
    print('\n'.join(missing[:20]))
    raise SystemExit('ERROR: Training embedding extraction incomplete')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

metadata = {
    'dataset': 'Training',
    'generated_or_verified_utc': datetime.now(timezone.utc).isoformat(),
    'fasta': str(fasta.resolve()),
    'fasta_sha256': sha256(fasta),
    'extraction_script': str(extractor.resolve()),
    'extraction_script_sha256': sha256(extractor),
    'embedding_dir': str(output.resolve()),
    'n_pairs': len(identifiers),
    'n_embedding_files_expected': 2 * len(identifiers),
    'encoder_model': 'Rostlab/prot_t5_xl_uniref50',
    'max_protein_len': 5000,
    'max_peptide_len': 50,
    'chunk_size': 1022,
    'chunk_overlap': 128,
    'embedding_dimension': 1024,
    'device': 'cuda',
    'half_precision_on_gpu': True,
    'pytorch': torch.__version__,
    'transformers': transformers.__version__,
}
manifest.write_text(json.dumps(metadata, indent=2) + '\n')
print(f'[DONE] All Training embeddings verified: {output}')
print(f'[DONE] Manifest saved: {manifest}')
PY

printf '\nTraining embeddings are complete.\n'
printf 'IMPORTANT: The training loader must use only %s, not its parent.\n' "$OUTPUT_DIR"


# env: pepphys_gnn