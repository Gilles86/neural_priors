#!/bin/bash
#SBATCH --job-name=tms_extract
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/dev/null

# Dump session-1 ROI-masked single-trial betas for one tms_risk subject.
#   sbatch --array=1-73 extract_tmsrisk_trials.sh

. $HOME/init_conda.sh
source activate tms_risk_cpu

PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
LOGFILE="/home/gdehol/logs/tms_extract_sub-${PARTICIPANT_LABEL}.txt"
scontrol update JobId=$SLURM_JOB_ID JobName="tms_extract_s${PARTICIPANT_LABEL}"

python -m neural_priors.value_comparison.extract_tmsrisk_trials \
    $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk \
    --out_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/value_comparison \
    > "$LOGFILE" 2>&1
