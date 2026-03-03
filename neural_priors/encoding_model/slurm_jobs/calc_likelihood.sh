#!/bin/bash
#SBATCH --job-name=nprf2_likelihood
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/gdehol/logs/nprf2_likelihood_%j.txt

# Load environment
. $HOME/init_conda.sh
source activate neural_priors2

MODEL=${1:?Error: No model number provided}
FIT_RESPONSES_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --fit_responses)
            FIT_RESPONSES_FLAG="--fit_responses"
            ;;
        --*)
            echo "Warning: Unknown argument '$arg' will be ignored."
            ;;
    esac
done

# Build suffix from active flags only
SUFFIX="_smoothed"
[[ -n "$FIT_RESPONSES_FLAG" ]] && SUFFIX="${SUFFIX}_fitresp"

LOGFILE="/home/gdehol/logs/nprf2_likelihood_model-${MODEL}${SUFFIX}.txt"

for SUBJECT in $(seq -w 1 41); do
    echo "Running subject ${SUBJECT}, model ${MODEL}, fit_responses=${FIT_RESPONSES_SUFFIX}"
    python $HOME/git/neural_priors/neural_priors/encoding_model/calc_likelihood.py \
        ${SUBJECT} ${MODEL} \
        --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
        --smoothed $FIT_RESPONSES_FLAG
done >> "$LOGFILE" 2>&1
