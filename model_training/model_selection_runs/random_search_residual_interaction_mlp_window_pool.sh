#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ResPepBind_training

SCRIPT="residual_interaction_mlp_training_publication_clean_final_window_pool.py"

TRAIN_FASTA="$SCRIPT_DIR/../../data/training/training.fasta"
EMBEDDING_DIR="$SCRIPT_DIR/../generate_embedding/embeddings"
TEST251="$SCRIPT_DIR/../../data/independent/Test251.fasta"
TEST1440="$SCRIPT_DIR/../../data/independent/Test1440.fasta"
TEST167="$SCRIPT_DIR/../../data/independent/Test167.fasta"
LEADSPEP="$SCRIPT_DIR/../../data/independent/Leadspep.fasta"
NEW_IND="$SCRIPT_DIR/../../data/independent/Test3664.fasta"
# -----------------------------
# Fixed model hyperparameters
# -----------------------------

MONITOR="f1"
NUM_FOLDS=10
HIDDEN_DIM=128
MLP_BLOCKS=3

DROPOUT="0.25"
INPUT_DROPOUT="0.20"
BLOCK_DROPOUTS="0.25,0.25,0.25"
HEAD_DROPOUT="0.15"

INTERACTION_FEATURES="p,q,absdiff,product"

MLP_EXPAND_RATIO="2.0"
HEAD_RATIO="0.5"
ACTIVATION="gelu"
NORM_TYPE="layernorm"
RESIDUAL_SCALE="1.0"

EPOCHS=60
PATIENCE=5
GAP_PATIENCE=5
MIN_EPOCHS_BEFORE_GAP_STOP=15
GAP_VAL_LOSS_TOLERANCE="0.03"
MAX_PEPTIDE_LEN=50
MAX_PROTEIN_LEN=1024
PROTEIN_POOLING_MODE="window_mean"
GRAD_CLIP="1.0"

ENSEMBLE_THRESHOLD_STRATEGIES="oof_youden"

# Publication-output options
FIG_FORMAT="pdf"  # png or pdf
MERGE_ALL_INDEPENDENT_PREDICTIONS=1

# Reproducibility seeds
SEEDS=(42)

# -----------------------------
# Random-search space
# -----------------------------

N_RANDOM_TRIALS=20

BATCH_SIZE_CHOICES=(16 32 64)
LR_CHOICES=("1e-4" "1e-3")
WEIGHT_DECAY_CHOICES=("5e-4")
MAX_GENERALIZATION_GAP_CHOICES=("0.15" "0.20" "0.25")

# Optional: make bash-level random sampling reproducible.
# Change this value if you want a different random-search sequence.
RANDOM=42

BASE_OUTPUT_DIR="run_training_window_pool"
BASE_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

mkdir -p "${BASE_OUTPUT_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Training script not found: ${SCRIPT}"
  echo "Please copy residual_interaction_mlp_training_publication_clean_final.py into:"
  echo "  $(pwd)"
  exit 1
fi

for SEED in "${SEEDS[@]}"; do
  for TRIAL in $(seq 1 "${N_RANDOM_TRIALS}"); do

    # -----------------------------
    # Sample one random configuration
    # -----------------------------

    BATCH_SIZE="${BATCH_SIZE_CHOICES[$RANDOM % ${#BATCH_SIZE_CHOICES[@]}]}"
    LR="${LR_CHOICES[$RANDOM % ${#LR_CHOICES[@]}]}"
    WEIGHT_DECAY="${WEIGHT_DECAY_CHOICES[$RANDOM % ${#WEIGHT_DECAY_CHOICES[@]}]}"
    MAX_GENERALIZATION_GAP="${MAX_GENERALIZATION_GAP_CHOICES[$RANDOM % ${#MAX_GENERALIZATION_GAP_CHOICES[@]}]}"

    echo "============================================================"
    echo "Starting random-search publication run"
    echo "trial=${TRIAL}/${N_RANDOM_TRIALS}"
    echo "seed=${SEED}"
    echo "hidden_dim=${HIDDEN_DIM}, mlp_blocks=${MLP_BLOCKS}"
    echo "interaction_features=${INTERACTION_FEATURES}"
    echo "dropout=${DROPOUT}, input_dropout=${INPUT_DROPOUT}, block_dropouts=${BLOCK_DROPOUTS}, head_dropout=${HEAD_DROPOUT}"
    echo "mlp_expand_ratio=${MLP_EXPAND_RATIO}, head_ratio=${HEAD_RATIO}, activation=${ACTIVATION}, norm_type=${NORM_TYPE}, residual_scale=${RESIDUAL_SCALE}"
    echo "lr=${LR}, weight_decay=${WEIGHT_DECAY}, batch_size=${BATCH_SIZE}, max_generalization_gap=${MAX_GENERALIZATION_GAP}"
    echo "max_protein_len=${MAX_PROTEIN_LEN}, protein_pooling_mode=${PROTEIN_POOLING_MODE}"
    echo "fig_format=${FIG_FORMAT}"
    echo "============================================================"

    LR_TAG="${LR//./p}"
    LR_TAG="${LR_TAG//-/m}"

    WD_TAG="${WEIGHT_DECAY//./p}"
    WD_TAG="${WD_TAG//-/m}"

    GAP_TAG="${MAX_GENERALIZATION_GAP//./p}"

    DROPOUT_TAG="${DROPOUT//./p}"
    INPUT_DROPOUT_TAG="${INPUT_DROPOUT//./p}"
    HEAD_DROPOUT_TAG="${HEAD_DROPOUT//./p}"
    EXPAND_TAG="${MLP_EXPAND_RATIO//./p}"
    HEAD_RATIO_TAG="${HEAD_RATIO//./p}"
    RES_SCALE_TAG="${RESIDUAL_SCALE//./p}"
    FEATURE_TAG="${INTERACTION_FEATURES//,/}"
    FEATURE_TAG="${FEATURE_TAG//[^A-Za-z0-9]/_}"

    RUN_NAME="random_resmlp_trial${TRIAL}_${MONITOR}_${NUM_FOLDS}fold_B${BATCH_SIZE}_HD${HIDDEN_DIM}_MLP${MLP_BLOCKS}_feat${FEATURE_TAG}_inDO${INPUT_DROPOUT_TAG}_blockDO${DROPOUT_TAG}_headDO${HEAD_DROPOUT_TAG}_exp${EXPAND_TAG}_head${HEAD_RATIO_TAG}_${ACTIVATION}_${NORM_TYPE}_res${RES_SCALE_TAG}_gap${GAP_TAG}_wd${WD_TAG}_lr${LR_TAG}_seed${SEED}_${BASE_TIMESTAMP}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

    mkdir -p "${OUTPUT_DIR}"

    cp "$(realpath "$0")" "${OUTPUT_DIR}/run_script.sh"
    cp "${SCRIPT}" "${OUTPUT_DIR}/${SCRIPT}"

    # Save sampled hyperparameters for later comparison.
    cat > "${OUTPUT_DIR}/random_search_config.tsv" <<EOF
trial	${TRIAL}
seed	${SEED}
batch_size	${BATCH_SIZE}
lr	${LR}
weight_decay	${WEIGHT_DECAY}
max_generalization_gap	${MAX_GENERALIZATION_GAP}
max_protein_len	${MAX_PROTEIN_LEN}
protein_pooling_mode	${PROTEIN_POOLING_MODE}
hidden_dim	${HIDDEN_DIM}
mlp_blocks	${MLP_BLOCKS}
interaction_features	${INTERACTION_FEATURES}
input_dropout	${INPUT_DROPOUT}
block_dropouts	${BLOCK_DROPOUTS}
head_dropout	${HEAD_DROPOUT}
mlp_expand_ratio	${MLP_EXPAND_RATIO}
head_ratio	${HEAD_RATIO}
activation	${ACTIVATION}
norm_type	${NORM_TYPE}
residual_scale	${RESIDUAL_SCALE}
monitor	${MONITOR}
num_folds	${NUM_FOLDS}
EOF

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
      --checkpoint "best_residual_interaction_mlp_${MONITOR}_${NUM_FOLDS}fold_HD${HIDDEN_DIM}_MLP${MLP_BLOCKS}_feat${FEATURE_TAG}_inDO${INPUT_DROPOUT_TAG}_blockDO${DROPOUT_TAG}_headDO${HEAD_DROPOUT_TAG}_gap${GAP_TAG}_wd${WD_TAG}_lr${LR_TAG}_seed${SEED}_trial${TRIAL}.pt"
      --output_dir "${OUTPUT_DIR}"
      --test_fastas "${TEST251}" "${TEST1440}" "${TEST167}" "${LEADSPEP}" "${NEW_IND}"
      --test_split_names Test251 Test1440 Test167 Leadspep NewInd
      --device cuda
      --seed "${SEED}"
      --num_workers 8
      --max_generalization_gap "${MAX_GENERALIZATION_GAP}"
      --gap_patience "${GAP_PATIENCE}"
      --min_epochs_before_gap_stop "${MIN_EPOCHS_BEFORE_GAP_STOP}"
      --gap_val_loss_tolerance "${GAP_VAL_LOSS_TOLERANCE}"
      --max_peptide_len "${MAX_PEPTIDE_LEN}"
      --max_protein_len "${MAX_PROTEIN_LEN}"
      --protein_pooling_mode "${PROTEIN_POOLING_MODE}"
      --ensemble_kfold
      --ensemble_threshold_strategies "${ENSEMBLE_THRESHOLD_STRATEGIES}"
      --save_oof_predictions
      --save_ensemble_predictions
      --report_format csv
      --fig_format "${FIG_FORMAT}"
      --grad_clip "${GRAD_CLIP}"
    )

    if [[ "${MERGE_ALL_INDEPENDENT_PREDICTIONS}" -eq 1 ]]; then
      CMD+=(--merge_all_independent_predictions)
    fi

    "${CMD[@]}"

    echo "Finished trial=${TRIAL}, seed=${SEED}"
    echo "Outputs saved to ${OUTPUT_DIR}"
    echo

  done
done

echo "All random-search publication runs completed."
echo "Base output directory: ${BASE_OUTPUT_DIR}"
