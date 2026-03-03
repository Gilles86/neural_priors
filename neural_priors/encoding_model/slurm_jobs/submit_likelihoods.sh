#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Null model + models 0-35 + models 115-145
MODELS=(-1 $(seq 0 35) $(seq 115 145))

for MODEL in "${MODELS[@]}"; do
    sbatch "${SCRIPT_DIR}/calc_likelihood.sh" ${MODEL} --smoothed
    sbatch "${SCRIPT_DIR}/calc_likelihood.sh" ${MODEL} --smoothed --fit_responses
done
