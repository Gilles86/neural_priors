#!/bin/bash
#SBATCH --job-name=nprf_gp
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=240:00
#SBATCH --output=/dev/null

# Hierarchical-Bayesian (GP-prior) vs classical LogGaussianPRF fit on NPC.
# Submit as: sbatch --array=4,8,10,26 fit_gp_prior.sh NPCr
# Optional second arg: --range narrow | --range wide | --range both

PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
ROI=${1:-NPCr}
RANGE_FLAG="${2:-}"

LOGFILE="$HOME/logs/nprf_gp_sub-${PARTICIPANT_LABEL}_roi-${ROI}.txt"
mkdir -p "$(dirname "$LOGFILE")"
scontrol update JobId=$SLURM_JOB_ID JobName="nprf_gp_${ROI}_s${PARTICIPANT_LABEL}"
exec > "$LOGFILE" 2>&1

# Direct path to env binary — avoids `conda run` buffering, no module load.
export PYTHONUNBUFFERED=1
PYTHON=$HOME/data/conda/envs/neural_priors3/bin/python

$PYTHON -u $HOME/git/neural_priors/neural_priors/encoding_model/fit_gp_prior.py \
    $PARTICIPANT_LABEL \
    --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    --roi $ROI $RANGE_FLAG
