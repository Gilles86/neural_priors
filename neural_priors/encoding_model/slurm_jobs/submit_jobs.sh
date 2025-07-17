#!/bin/bash

# Ensure a model label is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 <model_label>"
    exit 1
fi

# Get the model label from the first argument
model_label=$1

# Define the script names
scripts=("fit_encoding_model.sh" "fit_encoding_model_cv.sh")

# Define all flag combinations, including an empty one
flags=(
    "--smoothed --log_space"
    "--smoothed"
    "--log_space"
    ""
)

# Submit all combinations as array jobs
for script in "${scripts[@]}"; do
    for flag in "${flags[@]}"; do
        if [[ -z "$flag" ]]; then
            echo "Submitting: sbatch --array=1-41 $script $model_label"
            sbatch --array=1-41 "$script" "$model_label"
        else
            echo "Submitting: sbatch --array=1-41 $script $model_label $flag"
            sbatch --array=1-41 "$script" "$model_label" $flag
        fi
    done
done
