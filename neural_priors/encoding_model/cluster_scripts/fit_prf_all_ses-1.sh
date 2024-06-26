#!/bin/bash
#SBATCH --job-name=fit_prf_ses-1_all
#SBATCH --output=/home/gdehol/logs/fit_prf_ses-1_all_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --time=1:00:00

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range narrow
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range wide

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range narrow
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 1 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range wide