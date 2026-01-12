#!/bin/bash
set -e

########################################
# User configuration
########################################

# Execution environment
RUN_ENV="server"        # local | server

# Paths
# PROJECT_ROOT="/public/home/yangzhy22022/study/gcn-emotion"
# CONDA_ENV="gcn"
# DATA_ROOT="/public/home/yangzhy22022/storage/datasets/seediv/eeg_feature_bands/dtabg"
PROJECT_ROOT="/home_data/home/hugf2022/code/gcn-emotion"
CONDA_ENV="emotion"
DATA_ROOT="/public/home/hugf2022/emotion/seediv/eeg_feature_smooth"

# Output naming
EXP_NAME="dgcnn_111"  # 修改这里区分不同实验
OUTPUT_ROOT="results"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
CHECKPOINTS_FOLDER="checkpoints/${EXP_NAME}"

########################################
# Training Hyperparameters
########################################

MODEL="dgcnn"
BATCH_SIZE=64
LR=1e-1
NUM_EPOCHS=50
TRAIN_RATIO=0.7
VAL_RATIO=0.2
EXP_TIMES=15
NUM_ELECTRODES=62
IN_CHANNELS=5
NUM_CLASSES=4
SPLIT="within_subject"

########################################
# Prepare output directory
########################################

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINTS_FOLDER}"

########################################
# Build command
########################################

CMD="python main.py \
  --mode train \
  --model ${MODEL} \
  --data_root ${DATA_ROOT} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${NUM_EPOCHS} \
  --train_ratio ${TRAIN_RATIO} \
  --val_ratio ${VAL_RATIO} \
  --exp_times ${EXP_TIMES} \
  --num_electrodes ${NUM_ELECTRODES} \
  --in_channels ${IN_CHANNELS} \
  --num_classes ${NUM_CLASSES} \
  --checkpoints_folder ${CHECKPOINTS_FOLDER} \
  --split ${SPLIT}"

########################################
# Run
########################################

if [ "${RUN_ENV}" = "local" ]; then
  echo "[INFO] Running Training locally"
  cd "${PROJECT_ROOT}"
  source ~/anaconda3/bin/activate "${CONDA_ENV}"
  echo "[CMD] ${CMD}"
  eval "${CMD}"

elif [ "${RUN_ENV}" = "server" ]; then
  echo "[INFO] Submitting Training job to Slurm"
  echo "[CMD] ${CMD}"
  # 提交给专门的 train.slurm
  sbatch scripts/train.slurm "${CMD}"

else
  echo "[ERROR] Unknown RUN_ENV=${RUN_ENV}"
  exit 1
fi