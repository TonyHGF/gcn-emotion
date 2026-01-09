#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00        # 训练通常需要更长时间，这里设为4小时
#SBATCH -J train_dgcnn         # Job Name
#SBATCH -o logs/train_%x_%j.out # Log区分 train
#SBATCH -e logs/train_%x_%j.err

set -e

PROJECT_ROOT="/home_data/home/hugf2022/code/gcn-emotion"
CONDA_ENV="emotion"

cd ${PROJECT_ROOT}
source ~/anaconda3/bin/activate ${CONDA_ENV}

echo "========================================"
echo "[INFO] Job Type: TRAINING"
echo "[INFO] Start Time: $(date)"
echo "[INFO] Command:"
echo "$1"
echo "========================================"

eval "$1"

echo "========================================"
echo "[INFO] Finished at: $(date)"
echo "========================================"