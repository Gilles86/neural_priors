#!/bin/bash
#SBATCH --job-name=decode_neural_priors
#SBATCH --ntasks=1
#SBATCH --time=300:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G  # Request more memory
#SBATCH --output=/home/gdehol/logs/decode_neural_priors_%A-%a.txt  # Default SLURM log

# Load environment
. $HOME/init_conda.sh
source activate neural_priors2

# Get participant label and model number
PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
MODEL=${1:?Error: No model number provided}
N_VOXELS=${2:?Error: No number of voxels provided}
SMOOTHED_FLAG=""
FIT_RESPONSES_FLAG=""
SPHERICAL_FLAG=""
SEPARATE_SIGMAS_FLAG=""
SMOOTHED_SUFFIX="raw"
FIT_RESPONSES_SUFFIX="nofit"
SPHERICAL_SUFFIX="full_covariance"
SEPARATE_SIGMAS_SUFFIX="single_sigma"

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
        --spherical_noise)
            SPHERICAL_FLAG="--spherical_noise"
            SPHERICAL_SUFFIX="spherical"
            ;;
        --separate_sigmas)
            SEPARATE_SIGMAS_FLAG="--separate_sigmas"
            SEPARATE_SIGMAS_SUFFIX="separate_sigmas"
            ;;
    esac
done

# Define dynamic log file
LOGFILE="/home/gdehol/logs/decode_neural_priors_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_model-${MODEL}_${SMOOTHED_SUFFIX}_${FIT_RESPONSES_SUFFIX}_${SPHERICAL_SUFFIX}_${SEPARATE_SIGMAS_SUFFIX}.txt"

# Run the encoding model fit and redirect output manually
python $HOME/git/neural_priors/neural_priors/encoding_model2/decode.py \
    $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    --n_voxels $N_VOXELS \
    $SMOOTHED_FLAG $FIT_RESPONSES_FLAG $SPHERICAL_FLAG $SEPARATE_SIGMAS_FLAG --model_label $MODEL > "$LOGFILE" 2>&1
