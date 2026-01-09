#!/bin/bash
set -e

########################################
# User configuration
########################################

# Execution environment
RUN_ENV="server"        # local | server

# Experiment mode
MODE="explain"          # train | explain
MODEL="dgcnn"

# Paths
PROJECT_ROOT="/home_data/home/hugf2022/code/gcn-emotion"
CONDA_ENV="emotion"

DATA_ROOT="/public/home/hugf2022/emotion/seediv/eeg_feature_bands/b"

EXP_NAME="explain_dtabg"
OUTPUT_ROOT="results"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

########################################
# Training arguments
########################################

CHECKPOINTS_FOLDER="checkpoints/${EXP_NAME}"
BATCH_SIZE=64
LR=1e-3
NUM_EPOCHS=100
TRAIN_RATIO=0.7
VAL_RATIO=0.15
EXP_TIMES=5

########################################
# Explain arguments
########################################

FEATURE_KEY="de_LDS"
NUM_ELECTRODES=62
IN_CHANNELS=5
NUM_CLASSES=4
TEST_SESSIONS="1 2 3"
CHECKPOINT_PATH="checkpoints/train_dtabg/best_model_exp1.pth"

########################################
# Prepare output directory
########################################

mkdir -p "${OUTPUT_DIR}"

########################################
# Build command
########################################

CMD="python main.py \
  --mode ${MODE} \
  --model ${MODEL} \
  --data_root ${DATA_ROOT} \
  --output_dir ${OUTPUT_DIR}"

if [ "${MODE}" = "train" ]; then
  CMD="${CMD} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num_epochs ${NUM_EPOCHS} \
    --train_ratio ${TRAIN_RATIO} \
    --val_ratio ${VAL_RATIO} \
    --exp_times ${EXP_TIMES} \
    --num_electrodes ${NUM_ELECTRODES} \
    --in_channels ${IN_CHANNELS} \
    --num_classes ${NUM_CLASSES} \
    --checkpoints_folder ${CHECKPOINTS_FOLDER}"
fi

if [ "${MODE}" = "explain" ]; then
  CMD="${CMD} \
    --feature_key ${FEATURE_KEY} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --num_electrodes ${NUM_ELECTRODES} \
    --in_channels ${IN_CHANNELS} \
    --num_classes ${NUM_CLASSES} \
    --test_sessions ${TEST_SESSIONS}"
fi

########################################
# Run
########################################

if [ "${RUN_ENV}" = "local" ]; then
  echo "[INFO] Running locally"
  cd "${PROJECT_ROOT}"
  source ~/anaconda3/bin/activate "${CONDA_ENV}"
  echo "[CMD] ${CMD}"
  eval "${CMD}"

elif [ "${RUN_ENV}" = "server" ]; then
  echo "[INFO] Submitting job to Slurm"
  echo "[CMD] ${CMD}"
  sbatch scripts/run.slurm "${CMD}"

else
  echo "[ERROR] Unknown RUN_ENV=${RUN_ENV}"
  exit 1
fi
