#!/bin/bash
#SBATCH --job-name=np_extract
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/dev/null

# Dump ROI-masked single-trial betas for one subject (SLURM array over subjects).
#   sbatch --array=1-41 extract_roi_trials.sh [--smoothed]

. $HOME/init_conda.sh
source activate neural_priors2

PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
SMOOTHED_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --smoothed) SMOOTHED_FLAG="--smoothed" ;;
    esac
done

SUFFIX=""
[[ -n "$SMOOTHED_FLAG" ]] && SUFFIX="_smoothed"
LOGFILE="/home/gdehol/logs/np_extract_sub-${PARTICIPANT_LABEL}${SUFFIX}.txt"
scontrol update JobId=$SLURM_JOB_ID JobName="np_extract_s${PARTICIPANT_LABEL}${SUFFIX}"

python -m neural_priors.value_comparison.extract_roi_trials \
    $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    --out_dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/value_comparison \
    $SMOOTHED_FLAG > "$LOGFILE" 2>&1
