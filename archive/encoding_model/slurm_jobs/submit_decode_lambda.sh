#!/bin/bash
# Submit decoding jobs for 10 lambda values (model 31, smoothed, all subjects).

SCRIPT="$(dirname "$0")/decode.sh"
ARRAY="1-41"
MODEL=31
N_VOXELS=200

for lambd in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "Submitting lambda=${lambd}"
    sbatch --array=$ARRAY "$SCRIPT" $MODEL $N_VOXELS --smoothed --fit_responses --lambd=${lambd}
done
