#!/bin/bash
#SBATCH --job-name=decode_neural_priors
#SBATCH --ntasks=1
#SBATCH --time=90:00
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
SUBTRACT_WT_BASELINE_FLAG=""
LAMBD_FLAG=""

# Parse script arguments
for arg in "$@"; do
    case "$arg" in
        --smoothed)
            SMOOTHED_FLAG="--smoothed"
            ;;
        --fit_responses)
            FIT_RESPONSES_FLAG="--fit_responses"
            ;;
        --spherical_noise)
            SPHERICAL_FLAG="--spherical_noise"
            ;;
        --separate_sigmas)
            SEPARATE_SIGMAS_FLAG="--separate_sigmas"
            ;;
        --subtract_wt_baseline)
            SUBTRACT_WT_BASELINE_FLAG="--subtract_wt_baseline"
            ;;
        --lambd=*)
            LAMBD_FLAG="--lambd=${arg#--lambd=}"
            ;;
        --*)
            echo "Warning: Unknown argument '$arg' will be ignored."
            ;;
    esac
done

# Build suffix from active flags only
SUFFIX=""
[[ -n "$SMOOTHED_FLAG" ]] && SUFFIX="${SUFFIX}_smoothed"
[[ -n "$FIT_RESPONSES_FLAG" ]] && SUFFIX="${SUFFIX}_fitresp"
[[ -n "$SPHERICAL_FLAG" ]] && SUFFIX="${SUFFIX}_spherical"
[[ -n "$SEPARATE_SIGMAS_FLAG" ]] && SUFFIX="${SUFFIX}_separate_sigmas"
[[ -n "$SUBTRACT_WT_BASELINE_FLAG" ]] && SUFFIX="${SUFFIX}_subtract_wt_baseline"
[[ -n "$LAMBD_FLAG" ]] && SUFFIX="${SUFFIX}_${LAMBD_FLAG#--lambd=}"

# Define dynamic log file
LOGFILE="/home/gdehol/logs/nprf2_decode_sub-${PARTICIPANT_LABEL}_model-${MODEL}${SUFFIX}.txt"

# Run the encoding model fit and redirect output manually
python $HOME/git/neural_priors/neural_priors/encoding_model/decode.py \
    $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    --n_voxels $N_VOXELS \
    $SMOOTHED_FLAG $FIT_RESPONSES_FLAG $SPHERICAL_FLAG $SEPARATE_SIGMAS_FLAG $SUBTRACT_WT_BASELINE_FLAG $LAMBD_FLAG --model_label $MODEL > "$LOGFILE" 2>&1
