#!/usr/bin/env bash
set -Eeuo pipefail

# ResPepBind: four seeds x 10 folds, OOF threshold, independent evaluation.
# IMPORTANT: pep_resmlp_train.py MUST have a split-aware embedding loader.
# The original recursive EmbeddingStore can mix same-named .pt files across splits.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# This script is intended to reside in ResPepBind/scripts_finalised_model/.
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/pep_resmlp_train_split_aware.py"
REPORT_SCRIPT="$SCRIPT_DIR/pep_resmlp_report.py"
TRAIN_FASTA="$PROJECT_DIR/data/training/Training.fasta"
EMBEDDING_DIR="$PROJECT_DIR/generate_embedding/embeddings"
INDEPENDENT_DIR="$PROJECT_DIR/data/independent"
BASE_OUTPUT_DIR="$SCRIPT_DIR/final_pepresmlp_runs"

SEEDS=(42 123)
TEST_NAMES=(Test251 Test1440 Test167 Leadspep Test3664)
TEST_FASTAS=()

# Avoid silently reproducing the known original filename-collision bug.
if [[ "${SPLIT_AWARE_LOADER_CONFIRMED:-0}" != "1" ]]; then
  cat >&2 <<'MSG'
ERROR: Split-aware embedding loader has not been confirmed.
The original pep_resmlp_train.py recursively indexes embeddings by basename,
which can mix Training and independent .pt files with identical names.
Patch and verify the loader first. Then run with:
  SPLIT_AWARE_LOADER_CONFIRMED=1 ./run_final_training.sh
This flag is an acknowledgement, NOT an automatic verification of the Python fix.
MSG
  exit 2
fi

for f in "$TRAIN_SCRIPT" "$REPORT_SCRIPT" "$TRAIN_FASTA"; do
  [[ -f "$f" ]] || { echo "ERROR: Missing file: $f" >&2; exit 1; }
done
for split in Training "${TEST_NAMES[@]}"; do
  [[ -d "$EMBEDDING_DIR/$split" ]] || { echo "ERROR: Missing embedding split: $EMBEDDING_DIR/$split" >&2; exit 1; }
done
for split in "${TEST_NAMES[@]}"; do
  f="$INDEPENDENT_DIR/$split.fasta"
  [[ -f "$f" ]] || { echo "ERROR: Missing independent FASTA: $f" >&2; exit 1; }
  TEST_FASTAS+=("$f")
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || { echo "ERROR: Missing Conda initialization: $CONDA_SH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate pepphys_gnn

python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is unavailable in ResPepBind_training.')
print('Visible GPU:', torch.cuda.get_device_name(0))
PY

mkdir -p "$BASE_OUTPUT_DIR"
cp "$0" "$BASE_OUTPUT_DIR/run_final_training.sh"

for SEED in "${SEEDS[@]}"; do
  OUT="$BASE_OUTPUT_DIR/pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed${SEED}"
  if [[ -f "$OUT/RUN_COMPLETED" ]]; then
    echo "Skipping completed seed $SEED: $OUT"
    continue
  fi
  mkdir -p "$OUT"
  cp "$TRAIN_SCRIPT" "$OUT/pep_resmlp_train.py"
  cp "$REPORT_SCRIPT" "$OUT/pep_resmlp_report.py"
  cp "$0" "$OUT/run_final_training.sh"
  printf 'seed=%s\nCUDA_VISIBLE_DEVICES=%s\nembedding_root=%s\n' "$SEED" "$CUDA_VISIBLE_DEVICES" "$EMBEDDING_DIR" > "$OUT/run_config.txt"

  echo "=== Training seed $SEED / 10 folds; output: $OUT ==="
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
done

python "$REPORT_SCRIPT" --parent_dir "$BASE_OUTPUT_DIR" \
  --out_dir "$BASE_OUTPUT_DIR/combined_publication_summary" \
  2>&1 | tee "$BASE_OUTPUT_DIR/combined_report.log"
echo "Finished: $BASE_OUTPUT_DIR"


# env: pepphys_gnn