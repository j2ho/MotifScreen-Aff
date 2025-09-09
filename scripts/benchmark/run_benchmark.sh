#!/bin/bash
# Convenience script for running the benchmark pipeline

# Default parameters
TARGETS_CSV="/home/j2ho/projects/vs_benchmarks/chembl_pdbnr_beststr_TEST.csv"
MODEL_PATH="models/full_msk_ablation/best.pkl"
MODEL_CONFIG="configs/ablation.yaml"
MODEL_NAME="ablation_full"
OUTPUT_DIR="benchmark_results_ablation"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --targets_csv)
      TARGETS_CSV="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model_config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max_targets)
      MAX_TARGETS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "===== MotifScreen-Aff Benchmark Pipeline ====="
echo "Targets CSV: $TARGETS_CSV"
echo "Model Path: $MODEL_PATH"
echo "Model Config: $MODEL_CONFIG"  
echo "Model Name: $MODEL_NAME"
echo "Output Dir: $OUTPUT_DIR"
if [ ! -z "$MAX_TARGETS" ]; then
    echo "Max Targets: $MAX_TARGETS"
fi
echo "=============================================="

# Build the command
CMD="python scripts/benchmark/benchmark_pipeline.py"
CMD="$CMD --targets_csv $TARGETS_CSV"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --model_config $MODEL_CONFIG"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --output_dir $OUTPUT_DIR"

if [ ! -z "$MAX_TARGETS" ]; then
    CMD="$CMD --max_targets $MAX_TARGETS"
fi

# Step 1: Prepare configs and job scripts
echo ""
echo "Step 1: Preparing configs and job scripts..."
$CMD --mode prepare

if [ $? -eq 0 ]; then
    echo ""
    echo "Step 1 completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review the generated configs in: $OUTPUT_DIR/configs/"
    echo "2. Review the generated job scripts in: $OUTPUT_DIR/scripts/"
    echo "3. Submit jobs with: bash $OUTPUT_DIR/scripts/submit_all.sh"
    echo "4. Monitor jobs with: squeue -u \$USER"
    echo "5. After jobs complete, collect results with:"
    echo "   $CMD --mode collect"
else
    echo "Step 1 failed!"
    exit 1
fi