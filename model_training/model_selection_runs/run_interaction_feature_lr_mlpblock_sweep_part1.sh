#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Interaction-feature + learning-rate + MLP-depth sweep
# PART 1/2
#
# Run the two scripts in separate screen sessions and set different GPUs:
#   part1: CUDA_VISIBLE_DEVICES=0
#   part2: CUDA_VISIBLE_DEVICES=1
# ============================================================

export CUDA_VISIBLE_DEVICES=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ResPepBind_training

SCRIPT="residual_interaction_mlp_training_interaction_features.py"

TRAIN_FASTA="$SCRIPT_DIR/../../data/training/training.fasta"
EMBEDDING_DIR="$SCRIPT_DIR/../generate_embedding/embeddings"
TEST251="$SCRIPT_DIR/../../data/independent/Test251.fasta"
TEST1440="$SCRIPT_DIR/../../data/independent/Test1440.fasta"
TEST167="$SCRIPT_DIR/../../data/independent/Test167.fasta"
LEADSPEP="$SCRIPT_DIR/../../data/independent/Leadspep.fasta"
NEW_IND="$SCRIPT_DIR/../../data/independent/Test3664.fasta"

MONITOR="f1"
NUM_FOLDS=10
HIDDEN_DIM=128

DROPOUT="0.25"
INPUT_DROPOUT="0.20"
HEAD_DROPOUT="0.15"

MLP_EXPAND_RATIO="2.0"
HEAD_RATIO="0.5"
ACTIVATION="gelu"
NORM_TYPE="layernorm"
RESIDUAL_SCALE="1.0"

WEIGHT_DECAY="5e-4"
BATCH_SIZE=32
EPOCHS=50

PATIENCE=5
MAX_GENERALIZATION_GAP="0.20"
GAP_PATIENCE=5
MIN_EPOCHS_BEFORE_GAP_STOP=15
GAP_VAL_LOSS_TOLERANCE="0.03"
MAX_PEPTIDE_LEN=50
GRAD_CLIP="1.0"

ENSEMBLE_THRESHOLD_STRATEGIES="oof_youden"

FIG_FORMAT="pdf"
SAVE_ENSEMBLE_PREDICTIONS=1
MERGE_ALL_FOLD_RESULTS_IND=0

SEEDS=(42)

BASE_OUTPUT_DIR="run_training_feature"
BASE_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

mkdir -p "${BASE_OUTPUT_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Training script not found: ${SCRIPT}"
  echo "Edit SCRIPT=... at the top of this file."
  exit 1
fi

# Each line: interaction_features|learning_rate|mlp_blocks
RUN_GRID=(
  "p,q,absdiff,product|1e-3|2"
  "p,q,absdiff,product|1e-3|3"
  "p,q,absdiff,product|1e-4|2"
  "p,q,absdiff,product|1e-4|3"
  "p,q,absdiff,product|1e-5|2"
  "p,q,absdiff,product|1e-5|3"
  "p,q,absdiff|1e-3|2"
  "p,q,absdiff|1e-3|3"
  "p,q,absdiff|1e-4|2"
  "p,q,absdiff|1e-4|3"
  "p,q,absdiff|1e-5|2"
  "p,q,absdiff|1e-5|3"
  "p,q,product|1e-3|2"
  "p,q,product|1e-3|3"
  "p,q,product|1e-4|2"
)

for ITEM in "${RUN_GRID[@]}"; do
  IFS="|" read -r INTERACTION_FEATURES LR MLP_BLOCKS <<< "${ITEM}"

  for SEED in "${SEEDS[@]}"; do

    if [[ "${MLP_BLOCKS}" -eq 2 ]]; then
      BLOCK_DROPOUTS="0.25,0.25"
    elif [[ "${MLP_BLOCKS}" -eq 3 ]]; then
      BLOCK_DROPOUTS="0.25,0.25,0.25"
    else
      echo "Unsupported MLP_BLOCKS=${MLP_BLOCKS}"
      exit 1
    fi

    LR_TAG="${LR//./p}"
    LR_TAG="${LR_TAG//-/m}"

    DROPOUT_TAG="${DROPOUT//./p}"
    INPUT_DROPOUT_TAG="${INPUT_DROPOUT//./p}"
    HEAD_DROPOUT_TAG="${HEAD_DROPOUT//./p}"
    EXPAND_TAG="${MLP_EXPAND_RATIO//./p}"
    HEAD_RATIO_TAG="${HEAD_RATIO//./p}"
    RES_SCALE_TAG="${RESIDUAL_SCALE//./p}"
    FEATURE_TAG="${INTERACTION_FEATURES//,/}"
    FEATURE_TAG="${FEATURE_TAG//[^A-Za-z0-9]/_}"

    RUN_NAME="resmlp_sweep_${MONITOR}_${NUM_FOLDS}fold_B${BATCH_SIZE}_HD${HIDDEN_DIM}_MLP${MLP_BLOCKS}_feat${FEATURE_TAG}_inDO${INPUT_DROPOUT_TAG}_blockDO${DROPOUT_TAG}_headDO${HEAD_DROPOUT_TAG}_exp${EXPAND_TAG}_head${HEAD_RATIO_TAG}_${ACTIVATION}_${NORM_TYPE}_res${RES_SCALE_TAG}_wd${WEIGHT_DECAY}_lr${LR_TAG}_seed${SEED}_${BASE_TIMESTAMP}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

    if [[ -d "${OUTPUT_DIR}" ]]; then
      echo "Skipping existing output directory: ${OUTPUT_DIR}"
      continue
    fi

    mkdir -p "${OUTPUT_DIR}"

    cp "$(realpath "$0")" "${OUTPUT_DIR}/run_script.sh"
    cp "${SCRIPT}" "${OUTPUT_DIR}/${SCRIPT}"

    echo "============================================================"
    echo "Starting sweep run PART 1/2 on GPU ${CUDA_VISIBLE_DEVICES}"
    echo "seed=${SEED}"
    echo "interaction_features=${INTERACTION_FEATURES}"
    echo "lr=${LR}"
    echo "mlp_blocks=${MLP_BLOCKS}"
    echo "output_dir=${OUTPUT_DIR}"
    echo "============================================================"

    CMD=(
      python "${SCRIPT}"
      --train_fasta "${TRAIN_FASTA}"
      --embedding_dir "${EMBEDDING_DIR}"
      --train_split_name train
      --protocol kfold
      --num_folds "${NUM_FOLDS}"
      --variant residual_interaction_mlp
      --hidden_dim "${HIDDEN_DIM}"
      --mlp_blocks "${MLP_BLOCKS}"
      --epochs "${EPOCHS}"
      --batch_size "${BATCH_SIZE}"
      --lr "${LR}"
      --weight_decay "${WEIGHT_DECAY}"
      --dropout "${DROPOUT}"
      --input_dropout "${INPUT_DROPOUT}"
      --block_dropouts "${BLOCK_DROPOUTS}"
      --head_dropout "${HEAD_DROPOUT}"
      --mlp_expand_ratio "${MLP_EXPAND_RATIO}"
      --head_ratio "${HEAD_RATIO}"
      --activation "${ACTIVATION}"
      --norm_type "${NORM_TYPE}"
      --residual_scale "${RESIDUAL_SCALE}"
      --interaction_features "${INTERACTION_FEATURES}"
      --monitor "${MONITOR}"
      --patience "${PATIENCE}"
      --checkpoint "best_residual_interaction_mlp_${MONITOR}_${NUM_FOLDS}fold_HD${HIDDEN_DIM}_MLP${MLP_BLOCKS}_feat${FEATURE_TAG}_inDO${INPUT_DROPOUT_TAG}_blockDO${DROPOUT_TAG}_headDO${HEAD_DROPOUT_TAG}_lr${LR_TAG}_seed${SEED}.pt"
      --output_dir "${OUTPUT_DIR}"
      --test_fastas "${TEST251}" "${TEST1440}" "${TEST167}" "${LEADSPEP}"
      --test_split_names Test251 Test1440 Test167 Leadspep
      --device cuda
      --seed "${SEED}"
      --num_workers 8
      --max_generalization_gap "${MAX_GENERALIZATION_GAP}"
      --gap_patience "${GAP_PATIENCE}"
      --min_epochs_before_gap_stop "${MIN_EPOCHS_BEFORE_GAP_STOP}"
      --gap_val_loss_tolerance "${GAP_VAL_LOSS_TOLERANCE}"
      --max_peptide_len "${MAX_PEPTIDE_LEN}"
      --ensemble_kfold
      --ensemble_threshold_strategies "${ENSEMBLE_THRESHOLD_STRATEGIES}"
      --report_format csv
      --grad_clip "${GRAD_CLIP}"
    )

    if [[ "${SAVE_ENSEMBLE_PREDICTIONS}" -eq 1 ]]; then
      CMD+=(--save_ensemble_predictions)
    fi

    if [[ "${MERGE_ALL_FOLD_RESULTS_IND}" -eq 1 ]]; then
      CMD+=(--merge_all_fold_results_ind)
    fi

    "${CMD[@]}"

    echo "Finished run:"
    echo "${OUTPUT_DIR}"
    echo

  done
done

echo "All PART 1/2 sweep runs completed."
echo "Base output directory: ${BASE_OUTPUT_DIR}"
