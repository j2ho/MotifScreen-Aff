#!/bin/bash
#SBATCH --job-name=benchmark_array
#SBATCH -p gpu-micro.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -o test_benchmark_results/logs/array_%A_%a.out
#SBATCH -e test_benchmark_results/logs/array_%A_%a.err
#SBATCH --array=1-<NUM_TASKS>%200
#SBATCH --nice=10000

# Arguments from benchmark_pipeline.py

TARGETS_CSV="{args.targets_csv}"
OUTPUT_DIR="{args.output_dir}"

cd /data/galaxy4/user/j2ho/projects/MotifScreen-Aff
export MASTER_PORT=$((12000 + RANDOM % 1000))
export MASTER_ADDR=127.0.0.1

# Get the line for this task
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TARGETS_CSV")
TARGET=$(echo $LINE | cut -d, -f1)
PDB=$(echo $LINE | cut -d, -f2)
LIGAND_IDX=$(echo $LINE | cut -d, -f3)

CONFIG="${OUTPUT_DIR}/configs/inference_config_${TARGET}_${PDB}_${LIGAND_IDX}.yaml"

echo "Running inference for $TARGET / $PDB / $LIGAND_IDX"
python run_motifscreen_unified.py --mode inference --inference_config "$CONFIG"