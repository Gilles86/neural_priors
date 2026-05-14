#!/bin/bash
#SBATCH --job-name=nprf_gp
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=360:00
#SBATCH --output=/dev/null

# Hierarchical-Bayesian (GP-prior) vs classical LogGaussianPRF fit on NPC.
# Submit as: sbatch --array=4,8,10,26 fit_gp_prior.sh NPCr
# Optional second arg: extra flags, e.g. "--smoothed" or "--range wide"

PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
ROI=${1:-NPCr}
EXTRA_FLAGS="${2:-}"

# Tag the log file so smoothed and unsmoothed runs don't overwrite.
SMOOTH_TAG=""
if [[ "$EXTRA_FLAGS" == *"--smoothed"* ]]; then
    SMOOTH_TAG=".smoothed"
fi
LOGFILE="$HOME/logs/nprf_gp_sub-${PARTICIPANT_LABEL}_roi-${ROI}${SMOOTH_TAG}.txt"
mkdir -p "$(dirname "$LOGFILE")"
scontrol update JobId=$SLURM_JOB_ID JobName="nprf_gp_${ROI}${SMOOTH_TAG}_s${PARTICIPANT_LABEL}"
exec > "$LOGFILE" 2>&1

# Direct path to env binary — avoids `conda run` buffering, no module load.
export PYTHONUNBUFFERED=1
# Dedicated env for this project — uses braincoder from ~/git/braincoder_main
# (pinned to main), so it can't be disturbed by branch switches in the
# active ~/git/braincoder checkout used by other projects.
PYTHON=$HOME/data/conda/envs/neural_priors_gp/bin/python

$PYTHON -u $HOME/git/neural_priors/neural_priors/encoding_model/fit_gp_prior.py \
    $PARTICIPANT_LABEL \
    --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    --roi $ROI $EXTRA_FLAGS
