#!/bin/bash
# Submit script — 2026-02-26
# Jobs to submit:
#   fit_model / fit_model_whole_brain / fit_model_cv / fit_model_cv_whole_brain
#   Models 34 & 35: smoothed, with and without --fit_responses
#   Censored models 0, 3, 4, 5: smoothed, no fit_responses
# Array: subjects 01-41

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ARRAY="1-41"

# ── Models 34 & 35 ─────────────────────────────────────────────────────────────
for MODEL in 34 35; do
    # ROI fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model.sh"           ${MODEL} --smoothed
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model.sh"           ${MODEL} --smoothed --fit_responses
    # ROI CV fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv.sh"        ${MODEL} --smoothed
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv.sh"        ${MODEL} --smoothed --fit_responses
    # Whole-brain fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_whole_brain.sh"    ${MODEL} --smoothed
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_whole_brain.sh"    ${MODEL} --smoothed --fit_responses
    # Whole-brain CV fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv_whole_brain.sh" ${MODEL} --smoothed
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv_whole_brain.sh" ${MODEL} --smoothed --fit_responses
done

# ── Censored models 0, 3, 4, 5 ─────────────────────────────────────────────────
# (No shift / Free shifts / Efficient shifts / Participant-specific slopes)
# Smoothed, no fit_responses, with --censored flag
for MODEL in 0 3 4 5; do
    # ROI fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model.sh"           ${MODEL} --smoothed --censored
    # ROI CV fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv.sh"        ${MODEL} --smoothed --censored
    # Whole-brain fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_whole_brain.sh"    ${MODEL} --smoothed --censored
    # Whole-brain CV fits
    sbatch --array=${ARRAY} "${SCRIPT_DIR}/fit_model_cv_whole_brain.sh" ${MODEL} --smoothed --censored
done
