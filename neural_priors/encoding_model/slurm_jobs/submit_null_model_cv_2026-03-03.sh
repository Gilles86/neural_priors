#!/bin/bash
# Submit script — 2026-03-03
# Null model (model -1) CV fits: smoothed, with and without --fit_responses
# Model -1 predicts per-voxel training mean — no gradient descent, ~5 min per subject.
# Using --time=5:00 to override the script default so SLURM's backfill scheduler
# can slot these into gaps and run them ahead of longer-running jobs.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ARRAY="1-41"

sbatch --array=${ARRAY} --time=5:00 "${SCRIPT_DIR}/fit_model_cv.sh" -1 --smoothed
sbatch --array=${ARRAY} --time=5:00 "${SCRIPT_DIR}/fit_model_cv.sh" -1 --smoothed --fit_responses
