#!/bin/bash
#SBATCH --job-name=nprf_fit_gaussian
#SBATCH --output=/home/gdehol/logs/nprf_fit_gaussian_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G  # Request more memory

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range narrow --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range wide --gaussian

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range narrow --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range wide --gaussian

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range wide --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range narrow --gaussian

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range wide --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range narrow --gaussian

python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range wide2 --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range wide2 --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --range wide2 --gaussian
python $HOME/git/neural_priors/neural_priors/encoding_model/fit_prf_cv.py $PARTICIPANT_LABEL 0 --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors --smoothed --range wide2 --gaussian