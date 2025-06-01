#!/bin/bash

# Define the models array (matching the Python script)
models=(
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)

# Loop over difficulties and models
for difficulty in sample easy medium hard; do
    for i in "${!models[@]}"; do
        model_name="${models[$i]}"
        echo "========================================"
        echo "Running evaluation for:"
        echo "Difficulty: $difficulty"
        echo "Model: $model_name (index: $i)"
        echo "========================================"
        
        python3 benchmarking.py --difficulty $difficulty --model_idx $i
        
        # Check if the previous command failed
        if [ $? -ne 0 ]; then
            echo "Error running model $model_name on $difficulty dataset"
            # Continue with next model instead of exiting
            continue
        fi
        
        echo "Completed: $model_name on $difficulty dataset"
        echo ""
    done
done

echo "All evaluations completed!"