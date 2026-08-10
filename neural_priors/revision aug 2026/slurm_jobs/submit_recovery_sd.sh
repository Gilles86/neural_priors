#!/bin/bash
# Width-scaling (sd_wide_scale) parameter recovery simulations.
#
# 8 cells (pooled/subjectwise x sd_scale 1.0/1.287794 x design full/censored),
# 100 iterations each, split into chunks of 10 iterations per array task:
#
#   sbatch --array=0-79 submit_recovery_sd.sh
#
# task -> cell = task / 10, start_iteration = (task % 10) * 10
#
#SBATCH --job-name=recovery_sd
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/gdehol/logs/recovery_sd_%A_%a.txt
set -eo pipefail
export PYTHONUNBUFFERED=1
export KERAS_BACKEND=tensorflow

TASK=${SLURM_ARRAY_TASK_ID}
CELL=$((TASK / 10))
START=$(( (TASK % 10) * 10 ))

SAMPLINGS=(pooled pooled pooled pooled subject subject subject subject)
SD_SCALES=(1.0 1.0 1.287794 1.287794 1.0 1.0 1.287794 1.287794)
DESIGNS=(full censored full censored full censored full censored)

SAMPLING=${SAMPLINGS[$CELL]}
SD_SCALE=${SD_SCALES[$CELL]}
DESIGN=${DESIGNS[$CELL]}

EXTRA_ARGS=()
if [ "$SAMPLING" = "subject" ]; then
    EXTRA_ARGS+=(--sample_subject)
fi

cd "$HOME/git/neural_priors/neural_priors/revision aug 2026"

exec $HOME/data/conda/envs/neural_priors_gp/bin/python -u simulate_data_sd.py \
    --design "$DESIGN" \
    --sd_wide_scale "$SD_SCALE" \
    --start_iteration "$START" \
    --n_iterations 10 \
    --noise 0.5 \
    --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    "${EXTRA_ARGS[@]}"
