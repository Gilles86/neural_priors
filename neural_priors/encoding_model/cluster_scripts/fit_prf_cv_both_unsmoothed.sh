#!/bin/bash
#SBATCH --job-name=nprf_fit_cv_both_unsmoothed
#SBATCH --output=/home/gdehol/logs/fit_nprf_both_unsmoothed_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --time=1:00:00

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors
