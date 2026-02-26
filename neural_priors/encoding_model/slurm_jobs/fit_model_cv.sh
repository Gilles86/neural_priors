#!/bin/bash
#SBATCH --job-name=nprf2_fit_cv
#SBATCH --ntasks=1
#SBATCH --time=60:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G  # Request more memory
#SBATCH --output=/home/gdehol/logs/nprf2_fit_cv_%A-%a.txt  # Default SLURM log

# Load environment
. $HOME/init_conda.sh
source activate neural_priors2

# Get participant label and model number
PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
MODEL=${1:?Error: No model number provided}
SMOOTHED_FLAG=""
FIT_RESPONSES_FLAG=""
SMOOTHED_SUFFIX="raw"
FIT_RESPONSES_SUFFIX="nofit"

# Parse script arguments
for arg in "$@"; do
    case "$arg" in
        --smoothed)
            SMOOTHED_FLAG="--smoothed"
            SMOOTHED_SUFFIX="smoothed"
            ;;
        --fit_responses)
            FIT_RESPONSES_FLAG="--fit_responses"
            FIT_RESPONSES_SUFFIX="fitresp"
            ;;

        --censored)
            CENSORED_FLAG="--censored"
            CENSORED_SUFFIX="censored"
            ;;
        --*)
            echo "Warning: Unknown argument '$arg' will be ignored."
            ;;
    esac
done

# Define dynamic log file
LOGFILE="/home/gdehol/logs/nprf2_fit_cv_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_model-${MODEL}_${SMOOTHED_SUFFIX}_${FIT_RESPONSES_SUFFIX}.txt"

# Run the cross-validated encoding model fit and redirect output manually
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_model_cv.py \
    $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    $SMOOTHED_FLAG $FIT_RESPONSES_FLAG --model $MODEL > "$LOGFILE" 2>&1
