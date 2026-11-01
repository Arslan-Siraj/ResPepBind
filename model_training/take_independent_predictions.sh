#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8


DATA="../data"

SCRIPT="pep_resmlp_predict.py"
RUN_DIR="Final_model"
EMBEDDINGS="../generate_embedding/embeddings"

OUT="$BASE/final_independent_predictions"
mkdir -p "$OUT"

for NAME in Test251 Test1440 Test167 Test3664 Leadspep; do
    echo "Predicting $NAME using CPU"

    python "$SCRIPT" \
        --run_dir "$RUN_DIR" \
        --fasta "$DATA/independent/${NAME}.fasta" \
        --embedding_dir "$EMBEDDINGS" \
        --output_csv "$OUT/${NAME}_predictions.csv" \
        --device cpu \
        --batch_size 64
done