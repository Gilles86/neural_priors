#!/bin/bash
#SBATCH --job-name=nprf_fit_joint_gaussian_cv
#SBATCH --output=/home/gdehol/logs/nprf_fit_joint_gaussian_cv_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G  # Request more memory

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
# python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --model 1
# python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --model 1

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --model 2
# python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --model 2

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --model 3
# python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --model 3

# python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --model 4
# python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --model 4

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --model 5
# python $HOME/git/neural_priors/neural_priors/encoding_model/fit_joint_prf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --model 5