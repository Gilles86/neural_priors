#!/bin/bash
#SBATCH --job-name=np_narrowlog
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gdehol/logs/np_narrowlog_%A_%a.txt
#SBATCH --array=1-2

# Does the LINEAR numerosity axis cost anything?
#
# neural_priors' own production model (LinearScalingModel, model 31) is
# LOG-NORMAL over numerosity.  The value-comparison analysis used a linear axis
# to keep the estimator identical to the value pipeline -- a choice about the
# comparison, not a claim about numerosity coding.  This runs the same analysis
# on a log axis to measure what that choice costs.
#
# Restricted to the NARROW range (10-25) because pooling narrow and wide blurs
# the range-shift effect: a voxel's preferred numerosity moves between
# conditions, so a single mu per voxel is the wrong model for pooled data.
# Task 1 = narrow linear, task 2 = narrow log.  The PAIR is the comparison;
# neither is interpretable against the pooled 480-trial run, which differs in
# both range composition and trial count.

set -eo pipefail
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export KERAS_BACKEND=tensorflow
# the extracted betas live on the share, not under /data as they do locally
export NP_VALUE_COMPARISON_DIR=/shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/value_comparison

case $SLURM_ARRAY_TASK_ID in
  1) DATASET=neural_priors_narrow ;;
  2) DATASET=neural_priors_narrow_log ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET}"

export PYTHONPATH="$HOME/git/neural_priors:$HOME/git/value_prf"
cd "$HOME/git/neural_priors"
exec "$HOME/data/conda/envs/abstract_values/bin/python" -u \
    -m neural_priors.value_comparison.run_analysis \
    --dataset "$DATASET" \
    --datadir "$HOME/git/value_prf/notes/data" \
    --n-perm 20
