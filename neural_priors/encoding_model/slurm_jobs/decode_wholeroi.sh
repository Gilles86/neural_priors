#!/bin/bash
#SBATCH --job-name=np_decode_wholeroi
#SBATCH --output=/home/gdehol/logs/np_decode_wholeroi_%A-%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=u24-cva0000-305
#
# Opt-in noise-model comparison (decode.py --noise_model compare): spherical
# (production), van Bergen & Jehee (ResidualFitter spherical=False, lambd 0)
# and LowRankNoise (k, T by nested CV, fixed inner parameters) on the same
# folds, encoding parameters and voxel sets: cvR2 > 0, top 500, top 1000 and
# the whole NPCr. Output: derivatives/decoding2/model31.smoothed.spherical_noise
# [.fit_responses].<TAG>/ (never the production tree).
#
#   sbatch --array=1-10,12-22,24-41 --export=ALL,FIT_RESPONSES=1 decode_wholeroi.sh
# Overrides: FIT_RESPONSES=0 (ground truth), TAG=wholeroi, SETS="0 500 1000 all",
#            K_GRID="0 1 2 5 10 20 40 80 120 160", EXTRA="--debug"
set -eo pipefail
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2
export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export CUDA_VISIBLE_DEVICES=""
export BRAINCODER_NOISE=${BRAINCODER_NOISE:-$HOME/git/braincoder-noise}
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

SUB=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
TAG=${TAG:-wholeroi}
SETS=${SETS:-0 500 1000 all}
K_GRID=${K_GRID:-0 1 2 5 10 20 40 80 120 160}
if [ "${FIT_RESPONSES:-1}" = "1" ]; then FR_FLAG="--fit_responses"; else FR_FLAG=""; fi

echo "braincoder-noise at $(git -C $BRAINCODER_NOISE rev-parse --short HEAD)"
echo "neural_priors at $(git -C $HOME/git/neural_priors rev-parse --short HEAD)"

cd $HOME/git/neural_priors/neural_priors/encoding_model
exec python -u decode.py $SUB --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    --model_label 31 --smoothed --spherical_noise $FR_FLAG \
    --noise_model compare --tag $TAG --voxel_sets $SETS --k_grid $K_GRID ${EXTRA}
