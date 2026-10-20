#!/bin/bash
#SBATCH --job-name=np_decode_lrcv
#SBATCH --output=/home/gdehol/logs/np_decode_lrcv_%A-%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=u24-cva0000-305
#
# Opt-in methods study: decoding with braincoder's LowRankNoise (rank k and
# likelihood temperature T by nested leave-one-run-out CV inside the training
# runs) next to the production ResidualFitter noise model fitted on the SAME
# per-fold encoding parameters. Production defaults and derivatives are not
# touched: output goes to derivatives/decoding2/<production key>.<TAG>/.
#
# Runs in the production env (neural_priors2, braincoder ed48b10). LowRankNoise
# is numpy-only and is loaded by file path from the separate noise-models
# checkout ($BRAINCODER_NOISE), so ~/git/braincoder and the env are untouched.
# CPU only (production decoding is CPU too); no GPU preflight needed.
#
#   sbatch --array=1-10,12-22,24-41%10 neural_priors/encoding_model/slurm_jobs/decode_lowrankcv.sh
# Override with --export=ALL,FIT_RESPONSES=0,TAG=lowrankcv,INNER=fixed,LOWRANK_K=cv,EXTRA="--debug"
set -eo pipefail
. $HOME/init_conda.sh
source activate neural_priors2
export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export CUDA_VISIBLE_DEVICES=""
export BRAINCODER_NOISE=${BRAINCODER_NOISE:-$HOME/git/braincoder-noise}

SUB=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
TAG=${TAG:-lowrankcv}
INNER=${INNER:-fixed}
LOWRANK_K=${LOWRANK_K:-cv}
INNER_GROUPS=${INNER_GROUPS:-run}
# FIT_RESPONSES=1 (default) = paper configuration (model31.smoothed.spherical_noise.fit_responses);
# FIT_RESPONSES=0 = ground-truth configuration (model31.smoothed.spherical_noise)
if [ "${FIT_RESPONSES:-1}" = "1" ]; then FR_FLAG="--fit_responses"; else FR_FLAG=""; fi

echo "braincoder-noise at $(git -C $BRAINCODER_NOISE rev-parse --short HEAD) ($(git -C $BRAINCODER_NOISE rev-parse --abbrev-ref HEAD))"
echo "neural_priors at $(git -C $HOME/git/neural_priors rev-parse --short HEAD)"

cd $HOME/git/neural_priors/neural_priors/encoding_model
python decode.py $SUB --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
    --model_label 31 --smoothed --spherical_noise --n_voxels 0 $FR_FLAG \
    --noise_model lowrank_cv --tag $TAG --inner_params $INNER --lowrank_k $LOWRANK_K \
    --inner_groups $INNER_GROUPS ${K_GRID:+--k_grid $K_GRID} ${EXTRA}
