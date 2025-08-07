#!/bin/bash
#SBATCH --job-name=se3     # Job name
#SBATCH -p gpu-super.q                    
#SBATCH --gres=gpu:6               # Request 1 GPU
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --ntasks=1                   # One task
#SBATCH --nodelist=nova[015]
#SBATCH -o logs/%j.q
#SBATCH -e logs/%j.e
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Avoid port conflict
export MASTER_PORT=$((12000 + RANDOM % 1000))
export MASTER_ADDR=127.0.0.1

python -m scripts.train.train --config configs/common.yaml --version v1.0 --model_note se3_opt
