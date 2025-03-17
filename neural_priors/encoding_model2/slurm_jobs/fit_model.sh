#!/bin/bash
#SBATCH --job-name=fit_nprf2
#SBATCH --ntasks=1
#SBATCH --time=15:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G  # Request more memory
#SBATCH --output=/home/gdehol/logs/fit_nprf2_%A-%a.txt  # Default SLURM log

# Load environment
. $HOME/init_conda.sh
source activate neural_priors2

# Get participant label and model number
PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
MODEL=${1:?Error: No model number provided}
SMOOTHED_FLAG=""
SMOOTHED_SUFFIX="raw"

# Check script arguments
for arg in "$@"; do
    case "$arg" in
        --smoothed)
            SMOOTHED_FLAG="--smoothed"
            SMOOTHED_SUFFIX="smoothed"
            ;;
    esac
done

# Define dynamic log file
LOGFILE="/home/gdehol/logs/nprf2_fit_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_model-${MODEL}_${SMOOTHED_SUFFIX}.txt"

# Run the encoding model fit and redirect output manually
python $HOME/git/neural_priors/neural_priors/encoding_model2/fit_model.py \
    $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    $SMOOTHED_FLAG --model $MODEL > "$LOGFILE" 2>&1