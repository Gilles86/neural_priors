#!/bin/bash
#SBATCH --job-name=fmriprep_alina
#SBATCH --output=/home/gdehol/logs/res_fmriprep_%A-alina2.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
source /etc/profile.d/lmod.sh
module load singularityce
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
# export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
# export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
singularity run -B /shares/zne.uzh/containers/templateflow:/opt/templateflow -B /shares/zne.uzh/gdehol/ds-neuralpriors:/data -B /scratch/gdehol:/workflow --cleanenv /shares/zne.uzh/containers/fmriprep-23.2.1 /data /data/derivatives/fmriprep participant --participant_label alina2  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative  --dummy-scans 4 --skip_bids_validation -w /workflow --no-submm-recon
