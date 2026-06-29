#!/bin/bash

# Exit immediately if any command fails
set -e

# Default paths
DEFAULT_CSV="../../data/datasets/train_swap_FakeBrisbane.csv"
DEFAULT_QASM="../mqtloader/qasm"
DEFAULT_OUTPUT="../../data/datasets/embedding_data/"

# Use command-line arguments if provided; otherwise use defaults
CSV_ARG="${1:-$DEFAULT_CSV}"
QASM_ARG="${2:-$DEFAULT_QASM}"
OUTPUT_ARG="${3:-$DEFAULT_OUTPUT}"

# 3. Define the target directory containing your scripts
SCRIPTS_DIR="./embedding_generation_scripts"

# Check if the scripts directory actually exists
if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "Error: Directory '$SCRIPTS_DIR' not found."
    exit 1
fi

echo "=================================================="
echo "Starting embedding generation pipeline..."
echo " CSV Path:   $CSV_ARG"
echo " QASM Dir:   $QASM_ARG"
echo " Output Dir: $OUTPUT_ARG"
echo "=================================================="

# 4. Loop through and execute every .py file in the folder
for script in "$SCRIPTS_DIR"/*.py; do
    # Check if any .py files actually exist to avoid running the literal string "*.py"
    [ -e "$script" ] || continue
    
    echo ""
    echo "--> Running: $(basename "$script")"
    echo "--------------------------------------------------"
    
    # Run the script and pass the 3 arguments
    python3 "$script" "$CSV_ARG" "$QASM_ARG" "$OUTPUT_ARG"
done

echo ""
echo "=================================================="
echo "All embedding scripts processed successfully!"
echo "=================================================="