#!/bin/bash
#SBATCH --job-name=nprf_fit_joint_alphagaussian
#SBATCH --ntasks=1
#SBATCH --time=15:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G  # Request more memory
#SBATCH --output=/home/gdehol/logs/nprf_fit_joint_alphagaussian_%A-%a.txt  # Default SLURM log

# Load environment
. $HOME/init_conda.sh
source activate tf2-gpu

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
LOGFILE="/home/gdehol/logs/nprf_fit_joint_alphagaussian_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_model-${MODEL}_${SMOOTHED_SUFFIX}_${LOG_SPACE_SUFFIX}.txt"

# Run the encoding model fit and redirect output manually
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_alpha_model.py \
    $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    $SMOOTHED_FLAG --model $MODEL > "$LOGFILE" 2>&1