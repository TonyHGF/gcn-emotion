#!/bin/bash
set -e

########################################
# User configuration
########################################

# Execution environment
RUN_ENV="server"        # local | server

# Paths
PROJECT_ROOT="/home_data/home/hugf2022/code/gcn-emotion"
CONDA_ENV="emotion"
DATA_ROOT="/public/home/hugf2022/emotion/seediv/eeg_feature_bands/bg"

# Output naming
EXP_NAME="explain_bg"
OUTPUT_ROOT="results"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

########################################
# Explain Hyperparameters
########################################

MODEL="dgcnn"
FEATURE_KEY="de_LDS"
NUM_ELECTRODES=62
IN_CHANNELS=2
NUM_CLASSES=4
TEST_SESSIONS="1 2 3"

# !!! 重要：这里指定你要解释的模型路径 !!!
CHECKPOINT_PATH="checkpoints/train_bg100/best_model_exp5.pth"

########################################
# Prepare output directory
########################################

mkdir -p "${OUTPUT_DIR}"

########################################
# Build command
########################################

CMD="python main.py \
  --mode explain \
  --model ${MODEL} \
  --data_root ${DATA_ROOT} \
  --output_dir ${OUTPUT_DIR} \
  --feature_key ${FEATURE_KEY} \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --num_electrodes ${NUM_ELECTRODES} \
  --in_channels ${IN_CHANNELS} \
  --num_classes ${NUM_CLASSES} \
  --test_sessions ${TEST_SESSIONS}"

########################################
# Run
########################################

if [ "${RUN_ENV}" = "local" ]; then
  echo "[INFO] Running Explanation locally"
  cd "${PROJECT_ROOT}"
  source ~/anaconda3/bin/activate "${CONDA_ENV}"
  echo "[CMD] ${CMD}"
  eval "${CMD}"

elif [ "${RUN_ENV}" = "server" ]; then
  echo "[INFO] Submitting Explanation job to Slurm"
  echo "[CMD] ${CMD}"
  # 提交给专门的 explain.slurm
  sbatch scripts/explain.slurm "${CMD}"

else
  echo "[ERROR] Unknown RUN_ENV=${RUN_ENV}"
  exit 1
fi