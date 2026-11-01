#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SEED="${SEED:-2026}"
FOLDS="${FOLDS:-10}"
GPU_IDS="${GPU_IDS:-0,1}"
TRAINER="${TRAINER:-$PARENT_DIR/pep_resmlp_train.py}"
EMBEDDING_DIR="${EMBEDDING_DIR:-../generate/embeddings}" #  pass your embedding directory here
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/coldstart_outputs_seed${SEED}_${FOLDS}fold}"
MANIFEST="$OUTPUT_ROOT/splits/split_manifest.csv"
CONFIG="$OUTPUT_ROOT/splits/split_config.json"
for file in "$TRAINER" "$MANIFEST" "$CONFIG"; do
  [[ -f "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done
[[ -d "$EMBEDDING_DIR" ]] || { echo "ERROR: embedding directory missing: $EMBEDDING_DIR" >&2; exit 1; }
# Do not rebuild the splits and do not use this concurrently with the old sequential runner.
GPU_IDS="$GPU_IDS" python - "$MANIFEST" "$CONFIG" "$TRAINER" "$EMBEDDING_DIR" "$OUTPUT_ROOT" "$SEED" "$FOLDS" <<'END_PYTHON'
import csv, json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

manifest, config, trainer, embeddings, output, seed, folds = sys.argv[1:]
gpus = [p.strip() for p in os.environ['GPU_IDS'].split(',')]
if len(gpus) != 2 or len(set(gpus)) != 2 or not all(x.isdigit() for x in gpus):
    raise SystemExit('GPU_IDS must have two different device IDs, e.g. GPU_IDS=0,1')
cfg = json.loads(Path(config).read_text())
if int(cfg['seed']) != int(seed) or int(cfg['folds']) != int(folds):
    raise SystemExit('Existing split configuration does not match SEED/FOLDS')
rows = list(csv.DictReader(open(manifest, newline='')))
if len(rows) != 3 * len(cfg['thresholds']) * int(folds):
    raise SystemExit(f'Unexpected split count: {len(rows)}')
for gpu in gpus:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    result = subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'], env=env, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f'GPU {gpu} not usable: {result.stderr or result.stdout}')
    print(f'Physical GPU {gpu}: {result.stdout.strip()}', flush=True)

lock = threading.Lock()
def notice(message):
    with lock:
        print(message, flush=True)

def worker(gpu, assigned):
    finished = skipped = 0
    failures = []
    for row in assigned:
        tag = f"{row['scenario']}_id{row['identity']}_fold{row['fold']}"
        if row['valid'] != '1':
            skipped += 1
            notice(f'[GPU {gpu}] invalid partition: {tag}')
            continue
        target = Path(output) / 'trained' / tag
        metrics = target / 'independent_test_metrics.csv'
        if metrics.is_file() and metrics.stat().st_size > 0:
            skipped += 1
            notice(f'[GPU {gpu}] existing result skipped: {tag}')
            continue
        target.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, trainer,
            '--train_fasta', row['train_fasta'],
            '--embedding_dir', embeddings,
            '--protocol', 'single_split',
            '--internal_val_fraction', '0.1',
            '--test_fastas', row['test_fasta'],
            '--test_split_names', 'coldstart_test',
            '--variant', 'residual_interaction_mlp',
            '--hidden_dim', '128', '--mlp_blocks', '3',
            '--epochs', '50', '--batch_size', '32',
            '--lr', '1e-3', '--weight_decay', '5e-4',
            '--dropout', '0.25', '--input_dropout', '0.20',
            '--block_dropouts', '0.25,0.25,0.25',
            '--head_dropout', '0.15',
            '--mlp_expand_ratio', '2.0', '--head_ratio', '0.5',
            '--activation', 'gelu', '--norm_type', 'layernorm',
            '--residual_scale', '1.0',
            '--interaction_features', 'p,q,absdiff',
            '--monitor', 'f1', '--patience', '5',
            '--max_generalization_gap', '0.20',
            '--gap_patience', '5',
            '--min_epochs_before_gap_stop', '15',
            '--gap_val_loss_tolerance', '0.03',
            '--max_peptide_len', '50', '--max_protein_len', '1024',
            '--grad_clip', '1.0', '--seed', seed,
            '--output_dir', str(target), '--report_format', 'csv', '--device', 'cuda',
        ]
        log_file = target / 'training_gpu.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='7')
        notice(f'[GPU {gpu}] START {tag} (train={row["train_n"]}, test={row["test_n"]}); log={log_file}')
        with log_file.open('w') as out:
            result = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode or not metrics.is_file() or metrics.stat().st_size == 0:
            failures.append(tag)
            notice(f'[GPU {gpu}] FAILED {tag}; inspect {log_file}')
            break
        finished += 1
        notice(f'[GPU {gpu}] DONE {tag}')
    return finished, skipped, failures

notice(f'Seed {seed}; {folds} folds; {len(rows)} partitions; devices {gpus}')
# Deterministic disjoint assignment ensures that no two processes write to the same run directory.
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(worker, gpu, rows[i::2]) for i, gpu in enumerate(gpus)]
    results = [future.result() for future in futures]
completed = sum(x[0] for x in results)
skipped = sum(x[1] for x in results)
failed = [tag for result in results for tag in result[2]]
notice(f'Finished: {completed} new evaluations; {skipped} skipped; {len(failed)} failures')
if failed:
    raise SystemExit('Failed: ' + ', '.join(failed))
END_PYTHON


## GPU_IDS=0,1 SEED=2026 FOLDS=10 ./run_coldstart.sh 