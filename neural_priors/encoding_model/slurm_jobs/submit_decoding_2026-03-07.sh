#!/bin/bash
# Submit decoding jobs — 2026-03-07
# Model 31, smoothed, n_voxels=200, all subjects.
# Sweeps:
#   subtract_wt_baseline: off / on
#   lambd: 0.0, 0.1, ..., 1.0

SCRIPT="$(dirname "$0")/decode.sh"
ARRAY="1-41"
MODEL=31
N_VOXELS=200

for SUBTRACT_FLAG in "" "--subtract_wt_baseline"; do
    for lambd in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        echo "Submitting subtract_wt_baseline='${SUBTRACT_FLAG}' lambda=${lambd}"
        sbatch --array=$ARRAY "$SCRIPT" $MODEL $N_VOXELS --smoothed --fit_responses ${SUBTRACT_FLAG} --lambd=${lambd}
    done
done
