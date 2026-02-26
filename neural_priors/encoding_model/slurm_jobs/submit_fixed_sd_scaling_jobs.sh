SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ARRAY="1-41"

for MODEL in {115..145}; do
    # ROI fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model.sh"           ${MODEL} --smoothed
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model.sh"           ${MODEL} --smoothed --fit_responses

    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv.sh"           ${MODEL} --smoothed
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv.sh"           ${MODEL} --smoothed --fit_responses

done