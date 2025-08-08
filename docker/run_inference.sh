#!/bin/bash
set -e  # Exit on error
set -o pipefail

# Arguments from the challenge runner
TEST_DATA_DIR="$1"
PRED_DIR="$2"

echo "=== Starting run_inference.sh ==="
echo "Test data directory: $TEST_DATA_DIR"
echo "Prediction output directory: $PRED_DIR"

# Create output dirs
mkdir -p "$PRED_DIR"

# Set working directories inside the writable prediction volume
PREPROCESSED_DIR="$PRED_DIR/preprocessed_data"
mkdir -p "$PREPROCESSED_DIR"

export TMPDIR="$PRED_DIR/tmp"
mkdir -p "$TMPDIR"

# Path to checkpoint (ensure it's inside your Docker image)
CHECKPOINT_PATH="/workspace/checkpoint"  # adjust if needed

# Full path to MNI atlas inside image
ATLAS_FILENAME="/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

# Sanity check
if [ ! -f "$ATLAS_FILENAME" ]; then
  echo "ERROR: Atlas not found at $ATLAS_FILENAME"
  exit 1
fi

echo "=== Step 1: Preprocessing ==="
python3 /workspace/preprocessing.py \
    --data "$TEST_DATA_DIR" \
    --saveto "$PREPROCESSED_DIR" \
    --atlas "$ATLAS_FILENAME"

echo "=== Step 2: Inference ==="
python3 /workspace/inference.py \
    --chkp_path "$CHECKPOINT_PATH" \
    --data_path "$PREPROCESSED_DIR" \
    --output_path "$PRED_DIR"

echo "=== Finished ==="
