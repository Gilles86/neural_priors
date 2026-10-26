#!/bin/bash
#SBATCH --job-name=np_beh_model
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/np_beh_model_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --exclude=u24-cva0000-305,u24-chaiam0-613
#
# One Bayesian-observer model of the estimation reports per array task
# (neural_priors/cogmodels/behaviour_model_fit.py; RangeEstimationModel on top of
# bauer EstimationTaskModel). JOBS is a ';'-separated list of argument strings for the
# fit script; the array index picks one (1-based).
#
#   sbatch --array=1-3 --export=NONE,CODE_DIR=$HOME/git/np_behaviour_model,BAUER_DIR=$HOME/git/bauer-estimation-np,JOBS="power_truncated --censor --low-rank-mass;log_truncated --censor;linear_truncated --censor" \
#       neural_priors/cogmodels/slurm_jobs/behaviour_model_fit.sh

set -eo pipefail
CODE_DIR="${CODE_DIR:?set CODE_DIR to a clean clone of neural_priors}"
BAUER_DIR="${BAUER_DIR:?set BAUER_DIR to the bauer estimation-model clone}"
export HOME="${HOME:-/home/gdehol}"
IFS=';' read -r -a ALL <<< "$JOBS"
ARGS="${ALL[$((SLURM_ARRAY_TASK_ID - 1))]}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
SCR="/scratch/gdehol/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
export TMPDIR="$SCR"
export PYTENSOR_FLAGS="compiledir=$SCR/pytensor"
export NUMBA_CACHE_DIR="$SCR/numba"
mkdir -p "$TMPDIR"
export PYTHONPATH="$BAUER_DIR:$CODE_DIR"
cd "$CODE_DIR"
DER=/shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/behaviour_model
echo "args: $ARGS  code $(git -C $CODE_DIR rev-parse --short HEAD)  bauer $(git -C $BAUER_DIR rev-parse --short HEAD)"
$HOME/data/conda/envs/bauer/bin/python -u -W ignore -m neural_priors.cogmodels.behaviour_model_fit \
    $ARGS --data ${DATA:-$DER/behaviour_trials.tsv} --out-dir ${OUT_DIR:-$DER/fits}
rm -rf "$SCR"
