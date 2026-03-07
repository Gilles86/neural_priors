#!/bin/bash
# Submit decoding jobs — 2026-03-07
# Model 31, smoothed, --fit_responses, all subjects.
# Sweeps:
#   n_voxels: 0, 100, 200, 500
#   subtract_wt_baseline: off / on
#   lambd: 0.0, 0.1, ..., 1.0

SCRIPT="$(dirname "$0")/decode.sh"
ARRAY="1-41"
MODEL=31

for N_VOXELS in 0 100 200 500; do
    for SUBTRACT_FLAG in "" "--subtract_wt_baseline"; do
        for lambd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            echo "Submitting n_voxels=${N_VOXELS} subtract_wt_baseline='${SUBTRACT_FLAG}' lambda=${lambd}"
            sbatch --array=$ARRAY "$SCRIPT" $MODEL $N_VOXELS --smoothed --fit_responses ${SUBTRACT_FLAG} --lambd=${lambd}
        done
    done
done
