#!/bin/bash
#SBATCH -p bme_cpu                # 1. 修改为 CPU 分区（请根据你学校/单位的实际分区名修改，如 'cpu' 或 'compute'）
#SBATCH --nodes=1                 # 2. 预处理通常在单节点上运行效率最高，避免跨节点通信延迟
#SBATCH --ntasks=1                # 运行一个任务实例
#SBATCH --cpus-per-task=8        # 3. 增加 CPU 核心数（根据你的数据量，可以设为 8, 16, 32 等）
#SBATCH --mem=16G                 # 4. 预处理通常很吃内存，建议加大内存申请
#SBATCH --time=01:00:00           # 5. 根据预处理时长调整时间限制
#SBATCH -J preprocess             # 任务名称
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# 切换到工作目录
cd /home_data/home/hugf2022/code/gcn-emotion/preprocessing

# 激活环境
source ~/anaconda3/bin/activate emotion


python extract_bands.py