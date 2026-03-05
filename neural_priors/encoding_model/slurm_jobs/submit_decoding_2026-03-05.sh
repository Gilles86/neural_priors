#!/bin/bash
# Submit decoding jobs — 2026-03-05
# Model 31, smoothed, all combinations of:
#   fit_responses: ground truth (no flag) / fit_responses (--fit_responses)
#   spherical_noise: yes / no
#   n_voxels: 0 / 100

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ARRAY="1-41"

for N_VOXELS in 0 100; do
    for FIT_RESPONSES_FLAG in "" "--fit_responses"; do
        for SPHERICAL_FLAG in "" "--spherical_noise"; do
            sbatch --array=${ARRAY} "${SCRIPT_DIR}/decode.sh" \
                31 ${N_VOXELS} --smoothed ${FIT_RESPONSES_FLAG} ${SPHERICAL_FLAG}
        done
    done
done
