#!/bin/bash
#SBATCH --job-name=opt     # Job name
#SBATCH -p gpu-super.q                    
#SBATCH --gres=gpu:2               # Request 1 GPU
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --ntasks=1                   # One task
#SBATCH --nodelist=nova[010]
#SBATCH -o logs/a%j.q
#SBATCH -e logs/a%j.e
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Avoid port conflict
export MASTER_PORT=$((12000 + RANDOM % 1000))
export MASTER_ADDR=127.0.0.1

python -m scripts.train.train_opt --config configs/ablation.yaml --version ablation --model_note abrun --debug 
