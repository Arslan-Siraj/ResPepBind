#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# ResPepBind: Independent prediction using saved 10-fold models
#
# Model: Original like_before, seed 42
# Device: CPU
# Embeddings: Dataset-specific directories
# Training: NOT performed
# ============================================================

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SCRIPT="$SCRIPT_DIR/pep_resmlp_predict.py"

RUN_DIR="$SCRIPT_DIR/final_pepresmlp_runs_like_before/pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed42"

DATA_DIR="$PROJECT_DIR/data/independent"
EMBEDDING_ROOT="$PROJECT_DIR/generate_embedding/embeddings"

OUT_DIR="$RUN_DIR/final_independent_predictions_corrected"

# ------------------------------------------------------------
# Conda environment
# ------------------------------------------------------------

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate pepphys_gnn

# ------------------------------------------------------------
# Verify files
# ------------------------------------------------------------

[[ -f "$SCRIPT" ]] || {
    echo "ERROR: Prediction script not found: $SCRIPT"
    exit 1
}

[[ -d "$RUN_DIR" ]] || {
    echo "ERROR: Run directory not found: $RUN_DIR"
    exit 1
}

# ------------------------------------------------------------
# Verify all 10 fold checkpoints
# ------------------------------------------------------------

echo "============================================"
echo "Checking saved model checkpoints"
echo "============================================"

for FOLD in $(seq 1 10); do

    CHECKPOINT="$RUN_DIR/fold${FOLD}/best_pepresmlp.pt"

    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "ERROR: Missing checkpoint: $CHECKPOINT"
        exit 1
    fi

    echo "Fold ${FOLD}: OK"

done

echo "All 10 fold checkpoints exist."

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

mkdir -p "$OUT_DIR"

# ------------------------------------------------------------
# Independent test datasets
# ------------------------------------------------------------

DATASETS=(
    Test251
    Test1440
    Test167
    Test3664
    Leadspep
)

# ------------------------------------------------------------
# Run independent predictions
# ------------------------------------------------------------

for NAME in "${DATASETS[@]}"; do

    FASTA="$DATA_DIR/${NAME}.fasta"
    EMBEDDINGS="$EMBEDDING_ROOT/$NAME"
    OUTPUT="$OUT_DIR/${NAME}_predictions.csv"

    [[ -f "$FASTA" ]] || {
        echo "ERROR: Missing FASTA: $FASTA"
        exit 1
    }

    [[ -d "$EMBEDDINGS" ]] || {
        echo "ERROR: Missing embeddings: $EMBEDDINGS"
        exit 1
    }

    echo
    echo "============================================"
    echo "Predicting: $NAME"
    echo "Device: CPU"
    echo "Model: like_before seed42"
    echo "FASTA: $FASTA"
    echo "Embeddings: $EMBEDDINGS"
    echo "Output: $OUTPUT"
    echo "============================================"

    python "$SCRIPT" \
        --run_dir "$RUN_DIR" \
        --fasta "$FASTA" \
        --embedding_dir "$EMBEDDINGS" \
        --output_csv "$OUTPUT" \
        --device cpu \
        --batch_size 64

    echo "Completed: $NAME"

done

echo
echo "============================================"
echo "ALL INDEPENDENT PREDICTIONS COMPLETED"
echo "============================================"
echo "Results saved to:"
echo "$OUT_DIR"