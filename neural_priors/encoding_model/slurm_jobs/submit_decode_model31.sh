#!/bin/bash
# Submit all decoding variants for model 31 (smoothed, n_voxels=200).
# Covers all combinations of:
#   --fit_responses        (on/off)
#   --spherical_noise      (on/off)
#   --separate_sigmas      (on/off)
#   --subtract_wt_baseline (on/off, skipped when --spherical_noise is set)

SCRIPT="$(dirname "$0")/decode.sh"
ARRAY="1-41"
MODEL=31
N_VOXELS=200

submit() {
    echo "Submitting: $@"
    sbatch --array=$ARRAY "$SCRIPT" $MODEL $N_VOXELS --smoothed "$@"
}

for fit_responses in "" "--fit_responses"; do
    for separate_sigmas in "" "--separate_sigmas"; do

        # Spherical noise (subtract_wt_baseline is irrelevant here)
        submit $fit_responses $separate_sigmas --spherical_noise

        # Full covariance, without baseline subtraction
        submit $fit_responses $separate_sigmas

        # Full covariance, with baseline subtraction
        submit $fit_responses $separate_sigmas --subtract_wt_baseline

    done
done
