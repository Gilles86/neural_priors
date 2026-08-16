#!/bin/bash
# Width-scaling (sd_wide_scale / r_sigma) recovery, dense sweep of generative
# values (not just null=1.0 / empirical=1.29), quick diagnostic (10 iterations
# per value, not the full 100 used for production Fig. S9). One array task per
# generative value, each running all 10 iterations for that value:
#
#   sbatch --array=0-6 submit_recovery_sd_sweep.sh
#
#SBATCH --job-name=recovery_sd_sweep
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/gdehol/logs/recovery_sd_sweep_%A_%a.txt
set -eo pipefail
export PYTHONUNBUFFERED=1
export KERAS_BACKEND=tensorflow

SD_SCALES=(0.6 0.8 1.0 1.2 1.4 1.6 1.8)
SD_SCALE=${SD_SCALES[$SLURM_ARRAY_TASK_ID]}

cd "$HOME/git/neural_priors/neural_priors/revision aug 2026"

exec $HOME/data/conda/envs/neural_priors_gp/bin/python -u simulate_data_sd_sweep.py \
    --sweep "$SD_SCALE" \
    --design full \
    --noise 0.8 \
    --n_iterations 10 \
    --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors
