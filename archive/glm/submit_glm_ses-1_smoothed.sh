#!/bin/bash
#SBATCH --job-name=fit_st_denoise_ses-1_smoothed
#SBATCH --output=/home/gdehol/logs/fit_st_denoise_ses-1_smoothed_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=45:00
#SBATCH --mem=32  # Request more memory

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate base

python $HOME/git/neural_priors/neural_priors/glm/fit_single_trials_denoise.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed
# python $HOME/git/tms_risk/tms_risk/glm/fit_single_trials_denoise.py $PARTICIPANT_LABEL 1 --bids_folder /scratch/gdehol/ds-tmsrisk
