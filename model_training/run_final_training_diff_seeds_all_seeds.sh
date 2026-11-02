#!/usr/bin/env bash
set -Eeuo pipefail

# ResPepBind: 4 independent seeds x 10-fold CV, 5 independent datasets.
# Uses pep_resmlp_train_split_aware.py and the pepphys_gnn Conda environment.
# Separate output root prevents overwriting existing final_pepresmlp_runs.
# Default physical GPU is 1; override using CUDA_VISIBLE_DEVICES=0 if desired.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/pep_resmlp_train.py"
REPORT_SCRIPT="$SCRIPT_DIR/pep_resmlp_report.py"
TRAIN_FASTA="$PROJECT_DIR/data/training/Training.fasta"
EMBEDDING_DIR="$PROJECT_DIR/generate_embedding/embeddings"
INDEPENDENT_DIR="$PROJECT_DIR/data/independent"
BASE_OUTPUT_DIR="$SCRIPT_DIR/final_pepresmlp_runs_like_before"

SEEDS=(42 123 2025 2026)
TEST_NAMES=(Test251 Test1440 Test167 Leadspep Test3664)
TEST_FASTAS=()

for file in "$TRAIN_SCRIPT" "$REPORT_SCRIPT" "$TRAIN_FASTA"; do
  [[ -f "$file" ]] || { echo "ERROR: Missing file: $file" >&2; exit 1; }
done
for split in Training "${TEST_NAMES[@]}"; do
  [[ -d "$EMBEDDING_DIR/$split" ]] || {
    echo "ERROR: Missing embedding directory: $EMBEDDING_DIR/$split" >&2
    exit 1
  }
done
for split in "${TEST_NAMES[@]}"; do
  fasta="$INDEPENDENT_DIR/$split.fasta"
  [[ -f "$fasta" ]] || { echo "ERROR: Missing FASTA: $fasta" >&2; exit 1; }
  TEST_FASTAS+=("$fasta")
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || { echo "ERROR: Conda initialization missing: $CONDA_SH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate pepphys_gnn

python -m py_compile "$TRAIN_SCRIPT" "$REPORT_SCRIPT"
python - <<'PY'
import os
import torch
print('PyTorch:', torch.__version__, flush=True)
print('CUDA available:', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is unavailable in pepphys_gnn.')
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
print('Visible device (cuda:0):', torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p "$BASE_OUTPUT_DIR"
cp -- "$0" "$BASE_OUTPUT_DIR/$(basename "$0")"
echo "Output root: $BASE_OUTPUT_DIR"
echo "Seeds: ${SEEDS[*]}"
echo "Physical GPU selection: $CUDA_VISIBLE_DEVICES"

for SEED in "${SEEDS[@]}"; do
  OUT="$BASE_OUTPUT_DIR/pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed${SEED}"
  if [[ -f "$OUT/RUN_COMPLETED" ]]; then
    echo "Skipping completed seed $SEED: $OUT"
    continue
  fi

  mkdir -p "$OUT"
  cp -- "$TRAIN_SCRIPT" "$OUT/pep_resmlp_train_split_aware.py"
  cp -- "$REPORT_SCRIPT" "$OUT/pep_resmlp_report.py"
  cp -- "$0" "$OUT/$(basename "$0")"
  printf 'seed=%s\nCUDA_VISIBLE_DEVICES=%s\nconda_environment=%s\nembedding_root=%s\n' \
    "$SEED" "$CUDA_VISIBLE_DEVICES" "pepphys_gnn" "$EMBEDDING_DIR" > "$OUT/run_config.txt"

  echo "=== Training seed $SEED: 10 folds; GPU $CUDA_VISIBLE_DEVICES; output: $OUT ==="
  python "$TRAIN_SCRIPT" \
    --train_fasta "$TRAIN_FASTA" \
    --embedding_dir "$EMBEDDING_DIR" \
    --train_split_name Training \
    --protocol kfold --num_folds 10 \
    --variant residual_interaction_mlp \
    --interaction_features p,q,absdiff \
    --hidden_dim 128 --mlp_blocks 3 \
    --epochs 50 --batch_size 32 \
    --lr 1e-3 --weight_decay 5e-4 \
    --dropout 0.25 --input_dropout 0.20 \
    --block_dropouts 0.25,0.25,0.25 --head_dropout 0.15 \
    --mlp_expand_ratio 2.0 --head_ratio 0.5 \
    --activation gelu --norm_type layernorm --residual_scale 1.0 \
    --monitor f1 --patience 5 --grad_clip 1.0 \
    --max_generalization_gap 0.20 --gap_patience 5 \
    --min_epochs_before_gap_stop 15 --gap_val_loss_tolerance 0.03 \
    --max_peptide_len 50 --max_protein_len 1024 \
    --checkpoint best_pepresmlp.pt \
    --output_dir "$OUT" \
    --test_fastas "${TEST_FASTAS[@]}" \
    --test_split_names "${TEST_NAMES[@]}" \
    --device cuda --seed "$SEED" --num_workers 8 \
    --ensemble_kfold --ensemble_threshold_strategies oof_youden \
    --save_oof_predictions --save_ensemble_predictions \
    --report_format both 2>&1 | tee "$OUT/training.log"

  python "$REPORT_SCRIPT" --run_dir "$OUT" --out_dir "$OUT/publication_summary" \
    2>&1 | tee "$OUT/report.log"

  touch "$OUT/RUN_COMPLETED"
  echo "=== Completed seed $SEED ==="
done

# Safe here because this report reads only the dedicated 'like_before' output root.
python "$REPORT_SCRIPT" --parent_dir "$BASE_OUTPUT_DIR" \
  --out_dir "$BASE_OUTPUT_DIR/combined_publication_summary" \
  2>&1 | tee "$BASE_OUTPUT_DIR/combined_report.log"

echo "Finished: $BASE_OUTPUT_DIR"
