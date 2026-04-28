#!/bin/bash

# Usage: ./run_templ_one.sh <gpu_id> <dataset> [gpu_count]
# Example: ./run_templ_one.sh 0 wiki-Vote
# Example: ./run_templ_one.sh 0,1,2 wiki-Vote 3

GPU_ID=$1
DATASET=$2
GPU_COUNT=${3:-1}  # Default to 1 if not specified

# Validate arguments
if [ -z "$GPU_ID" ] || [ -z "$DATASET" ]; then
    echo "Error: Missing arguments"
    echo "Usage: $0 <gpu_id> <dataset> [gpu_count]"
    echo "  gpu_id: GPU device ID(s), e.g., '0' or '0,1,2'"
    echo "  dataset: Dataset name"
    echo "  gpu_count: Number of GPUs to use (default: 1)"
    exit 1
fi

cd build

echo ">>> Start $DATASET on GPU $GPU_ID (GPU_count=$GPU_COUNT)"

# Base arguments shared across all runs
BASE_ARGS="--config ../config/graphs/${DATASET}.config --all --printworkload --GPU_count $GPU_COUNT"

# List of walker types to run
WALKERS=(
    "Node2vec"
    "Node2vec_weighted"
    "Metapath"
    "Metapath_weighted"
    "PPR_second"
)

# Run each walker type
for walker in "${WALKERS[@]}"; do
    echo ">>> Running $walker"
    CUDA_VISIBLE_DEVICES=$GPU_ID ./bin/flowwalker $BASE_ARGS --$walker
    echo ">>>"
done

echo ">>> End $DATASET"

