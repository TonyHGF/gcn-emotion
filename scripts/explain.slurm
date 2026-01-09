#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00        # 解释任务通常较快，1小时足矣
#SBATCH -J explain_dgcnn       # Job Name
#SBATCH -o logs/exp_%x_%j.out  # Log区分 explain
#SBATCH -e logs/exp_%x_%j.err

set -e

PROJECT_ROOT="/home_data/home/hugf2022/code/gcn-emotion"
CONDA_ENV="emotion"

cd ${PROJECT_ROOT}
source ~/anaconda3/bin/activate ${CONDA_ENV}

echo "========================================"
echo "[INFO] Job Type: EXPLANATION"
echo "[INFO] Start Time: $(date)"
echo "[INFO] Command:"
echo "$1"
echo "========================================"

eval "$1"

echo "========================================"
echo "[INFO] Finished at: $(date)"
echo "========================================"