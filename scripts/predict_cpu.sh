#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
exec respepbind --device cpu --threads "${OMP_NUM_THREADS}" "$@"
