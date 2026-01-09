#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J dgcnn
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e

PROJECT_ROOT="/home_data/home/hugf2022/code/gcn-emotion"
CONDA_ENV="emotion"

cd ${PROJECT_ROOT}
source ~/anaconda3/bin/activate ${CONDA_ENV}

echo "[INFO] Running command:"
echo "$1"

eval "$1"
