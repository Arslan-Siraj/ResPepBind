
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PepResMLP FINAL: fixed p,q,absdiff / 3 residual blocks
# 10-fold CV + out-of-fold threshold + independent evaluation
# Files may reside in scripts_finalised_model (no cd to Training).
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES=1

# Activate the environment containing torch / scikit-learn.
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: Conda initialization not found: $CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate pepphys_gnn

TRAIN_SCRIPT="$SCRIPT_DIR/pep_resmlp_train.py"
REPORT_SCRIPT="$SCRIPT_DIR/pep_resmlp_report.py"

# Existing published benchmark paths; edit these if data were moved.
TRAIN_FASTA="$SCRIPT_DIR/../data/training/training.fasta"
EMBEDDING_DIR="$SCRIPT_DIR/../generate_embedding/embeddings"
TEST251="$SCRIPT_DIR/../data/independent/Test251.fasta"
TEST1440="$SCRIPT_DIR/../data/independent/Test1440.fasta"
TEST167="$SCRIPT_DIR/../data/independent/Test167.fasta"
LEADSPEP="$SCRIPT_DIR/../data/independent/Leadspep.fasta"

SEEDS=(42 123 2026 1997)
BASE_OUTPUT_DIR="$SCRIPT_DIR/final_pepresmlp_runs"

for path in "$TRAIN_SCRIPT" "$REPORT_SCRIPT" "$TRAIN_FASTA" "$TEST251" "$TEST1440" "$TEST167" "$LEADSPEP"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Required file not found: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$EMBEDDING_DIR" && ! -f "$EMBEDDING_DIR" ]]; then
  echo "ERROR: Embedding path not found: $EMBEDDING_DIR" >&2
  exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"
for SEED in "${SEEDS[@]}"; do
  OUT="$BASE_OUTPUT_DIR/pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed${SEED}"
  if [[ -s "$OUT/ensemble_independent_test_metrics.csv" ]]; then
    echo "Skipping completed run: $OUT"
    continue
  fi
  mkdir -p "$OUT"
  cp "$TRAIN_SCRIPT" "$OUT/pep_resmlp_train.py"
  cp "$REPORT_SCRIPT" "$OUT/pep_resmlp_report.py"
  cp "$SCRIPT_DIR/run_final_training.sh" "$OUT/run_final_training.sh"

  echo "============================================"
  echo "Training PepResMLP seed=$SEED  GPU=$CUDA_VISIBLE_DEVICES"
  echo "Output: $OUT"
  echo "============================================"

  python "$TRAIN_SCRIPT" \
    --train_fasta "$TRAIN_FASTA" \
    --embedding_dir "$EMBEDDING_DIR" \
    --train_split_name train \
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
    --test_fastas "$TEST251" "$TEST1440" "$TEST167" "$LEADSPEP" \
    --test_split_names Test251 Test1440 Test167 Leadspep \
    --device cuda --seed "$SEED" --num_workers 8 \
    --ensemble_kfold --ensemble_threshold_strategies oof_youden \
    --save_oof_predictions --save_ensemble_predictions \
    --report_format both

  python "$REPORT_SCRIPT" --run_dir "$OUT" --out_dir "$OUT/publication_summary"
done

python "$REPORT_SCRIPT" --parent_dir "$BASE_OUTPUT_DIR" \
  --out_dir "$BASE_OUTPUT_DIR/combined_publication_summary"
echo "Finished. Reports: $BASE_OUTPUT_DIR"

