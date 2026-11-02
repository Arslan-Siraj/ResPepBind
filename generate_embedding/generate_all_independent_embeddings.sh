#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. This file belongs in ResPepBind/generate_embedding/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXTRACTOR="$SCRIPT_DIR/pepri_hgt_training.py"
FASTA_DIR="$PROJECT_DIR/data/independent"
EMBEDDING_DIR="$SCRIPT_DIR/embeddings"
LOG_DIR="$SCRIPT_DIR/embedding_logs"

# All independent sets by default, or choose particular sets as arguments.
if (( $# > 0 )); then
    DATASETS=("$@")
else
    DATASETS=(Test251 Test1440 Test167 Leadspep Test3664)
fi

[[ -f "$EXTRACTOR" ]] || { echo "ERROR: Missing extractor: $EXTRACTOR" >&2; exit 1; }
mkdir -p "$EMBEDDING_DIR" "$LOG_DIR"

# Pre-check every input before running the expensive encoder.
for NAME in "${DATASETS[@]}"; do
    case "$NAME" in
        Test251|Test1440|Test167|Leadspep|Test3664) ;;
        *) echo "ERROR: Unknown dataset '$NAME'" >&2; exit 1 ;;
    esac
    [[ -f "$FASTA_DIR/${NAME}.fasta" ]] || {
        echo "ERROR: Missing FASTA: $FASTA_DIR/${NAME}.fasta" >&2
        exit 1
    }
done

python - <<'PY'
import torch, transformers
from transformers import T5EncoderModel, T5Tokenizer
print(f'[ENV] PyTorch={torch.__version__}; transformers={transformers.__version__}')
print(f'[ENV] CUDA available={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA not available. Refusing CPU extraction to prevent unexpected slow processing.')
print(f'[ENV] GPU={torch.cuda.get_device_name(0)}')
PY

printf '\nOutput root: %s\n' "$EMBEDDING_DIR"
for NAME in "${DATASETS[@]}"; do
    INPUT="$FASTA_DIR/${NAME}.fasta"
    LOG="$LOG_DIR/${NAME}_embedding.log"

    printf '\n======================================================\n'
    printf 'Generating embeddings for %s\n' "$NAME"
    printf 'FASTA: %s\nLog: %s\n' "$INPUT" "$LOG"
    printf '======================================================\n'

    # The extractor skips each pair if both .pt files already exist,
    # so an interrupted dataset can be resumed by rerunning this script.
    python "$EXTRACTOR" \
        --extract_embeddings \
        --extract_fasta "$INPUT" \
        --extract_split_name "$NAME" \
        --embedding_dir "$EMBEDDING_DIR" \
        --max_protein_len 5000 \
        --max_peptide_len 50 \
        --prott5_chunk_size 1022 \
        --prott5_chunk_overlap 128 \
        2>&1 | tee "$LOG"

    # Check pair-wise output completeness, not only the total number of files.
    python - "$INPUT" "$EMBEDDING_DIR/$NAME" <<'PY'
from pathlib import Path
import sys
fasta, folder = map(Path, sys.argv[1:])
ids = [line[1:].strip() for line in fasta.open(encoding='utf-8') if line.startswith('>')]
missing = []
for pair_id in ids:
    for molecule in ('peptide', 'protein'):
        path = folder / f'{pair_id}_{molecule}.pt'
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
print(f'[VERIFY] {folder.name}: {len(ids)} pairs; {2 * len(ids)} expected embeddings; missing/empty={len(missing)}')
if missing:
    print('\n'.join(missing[:20]))
    raise SystemExit('ERROR: Incomplete embedding extraction')
PY
    printf '[DONE] %s\n' "$NAME"
done

printf '\nAll selected independent embeddings verified.\n'
printf 'Saved under: %s/{Test251,Test1440,Test167,Leadspep,Test3664}/\n' "$EMBEDDING_DIR"
printf 'IMPORTANT: In training, load only the matching dataset folder, not the common parent.\n'

# env: pepphys_gnn