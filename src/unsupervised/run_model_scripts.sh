#!/bin/bash

# Exit immediately if any command fails
set -e

# 1. Default paths
DEFAULT_EMB="../../data/datasets/embedding_data/"
DEFAULT_CSV="../../data/datasets/train_swap_FakeBrisbane.csv"
OUTPUT_ARG="../../data/datasets/unsupervised_learning_results"


# 2. Use command-line arguments if provided; otherwise use defaults
EMB_ARG="${1:-$DEFAULT_EMB}"
CSV_ARG="${2:-$DEFAULT_CSV}"
OUTPUT_ARG="${3:-$OUTPUT_ARG}"

# 3. Define the specific target script
TARGET_SCRIPT="./model_run_scripts/run_unsupervised_learning_pipeline.py"

# Check if the target script file actually exists
if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Error: Script '$TARGET_SCRIPT' not found."
    exit 1
fi

echo "=================================================="
echo "Starting unsupervised learning pipeline..."
echo " Embedding Dir: $EMB_ARG"
echo " CSV Path:      $CSV_ARG"
echo " Output Dir:    $OUTPUT_ARG"
echo "=================================================="

echo ""
echo "--> Running: $(basename "$TARGET_SCRIPT")"
echo "--------------------------------------------------"

# 4. Run the single target script and pass the 3 arguments
python3 "$TARGET_SCRIPT" "$EMB_ARG" "$CSV_ARG" "$OUTPUT_ARG"

echo ""
echo "=================================================="
echo "Pipeline script processed successfully!"
echo "=================================================="