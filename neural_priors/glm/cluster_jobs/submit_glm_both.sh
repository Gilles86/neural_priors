#!/bin/bash
#SBATCH --job-name=fit_st_denoise_ses-both
#SBATCH --output=/home/gdehol/logs/fit_st_denoise_both_%A-%a.txt
#vSBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=2:30:00

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python $HOME/git/neural_priors/neural_priors/glm/fit_single_trials_denoise.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors