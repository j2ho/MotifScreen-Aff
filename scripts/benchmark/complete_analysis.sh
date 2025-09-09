#!/bin/bash
# Complete benchmark analysis pipeline for MotifScreen-Aff

# Default parameters
TARGETS_CSV="/home/j2ho/projects/vs_benchmarks/chembl_pdbnr_beststr_TEST.csv"
MODEL_PATH="models/full_msk_ablation/best.pkl"
MODEL_CONFIG="configs/ablation.yaml"
MODEL_NAME="ablation_full"
OUTPUT_DIR="benchmark_results_ablation"
DATA_DIR="/home/j2ho/projects/vs_benchmarks/data"

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
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --max_targets)
      MAX_TARGETS="$2"
      shift 2
      ;;
    --collect_only)
      COLLECT_ONLY=true
      shift
      ;;
    --analyze_only)
      ANALYZE_ONLY=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "===== MotifScreen-Aff Complete Benchmark Analysis ====="
echo "Targets CSV: $TARGETS_CSV"
echo "Model Path: $MODEL_PATH"
echo "Model Config: $MODEL_CONFIG"  
echo "Model Name: $MODEL_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "Data Dir: $DATA_DIR"
if [ ! -z "$MAX_TARGETS" ]; then
    echo "Max Targets: $MAX_TARGETS"
fi
echo "======================================================="

# Build the benchmark command
BENCHMARK_CMD="python scripts/benchmark/benchmark_pipeline.py"
BENCHMARK_CMD="$BENCHMARK_CMD --targets_csv $TARGETS_CSV"
BENCHMARK_CMD="$BENCHMARK_CMD --model_path $MODEL_PATH"
BENCHMARK_CMD="$BENCHMARK_CMD --model_config $MODEL_CONFIG"
BENCHMARK_CMD="$BENCHMARK_CMD --model_name $MODEL_NAME"
BENCHMARK_CMD="$BENCHMARK_CMD --output_dir $OUTPUT_DIR"

if [ ! -z "$MAX_TARGETS" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --max_targets $MAX_TARGETS"
fi

# Collect results if not analyze_only
if [ "$ANALYZE_ONLY" != true ]; then
    echo ""
    echo "Step 1: Collecting benchmark results..."
    echo "======================================="
    $BENCHMARK_CMD --mode collect
    
    if [ $? -ne 0 ]; then
        echo "Collection failed!"
        exit 1
    fi
fi

# Skip analysis if collect_only
if [ "$COLLECT_ONLY" == true ]; then
    echo "Collection complete. Skipping analysis (--collect_only specified)."
    exit 0
fi

# Analyze results
PER_TARGET_DIR="$OUTPUT_DIR/${MODEL_NAME}_per_target_results"

echo ""
echo "Step 2: Analyzing results with AUROC calculation..."
echo "=================================================="

python scripts/benchmark/analyze_results.py \
    --results_dir "$PER_TARGET_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "$DATA_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "===== COMPLETE ANALYSIS FINISHED ====="
    echo "Results available in:"
    echo "- Per-target results: $PER_TARGET_DIR/"
    echo "- Analysis results: ${OUTPUT_DIR}/${MODEL_NAME}_analysis/"
    echo "- Combined results: ${OUTPUT_DIR}/${MODEL_NAME}_benchmark_results.csv"
    echo "- Metrics summary: ${OUTPUT_DIR}/${MODEL_NAME}_analysis/${MODEL_NAME}_all_metrics.csv"
    echo "======================================="
else
    echo "Analysis failed!"
    exit 1
fi