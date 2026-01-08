#!/bin/bash
#SBATCH -p bme_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G              # 稍微多给点内存，防止被 Kill
#SBATCH -J data_check
#SBATCH -o logs/check_result.out
#SBATCH -e logs/check_result.err

source ~/anaconda3/bin/activate emotion
python preprocessing/check_data.py