#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J main
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


cd /home_data/home/hugf2022/code/gcn-emotion
source ~/anaconda3/bin/activate emotion

python main.py
