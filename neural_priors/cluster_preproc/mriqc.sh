#!/bin/bash
#SBATCH --job-name=mriqc
#SBATCH --output=/home/gdehol/logs/res_mriqc_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=45:00
source /etc/profile.d/lmod.sh
module load singularityce/3.10.2
export SINGULARITYENV_FS_LICENSE=$FREESURFER_HOME/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
singularity run -B /shares/zne.uzh/containers/templateflow:/opt/templateflow -B /shares/zne.uzh/gdehol/ds-neuralpriors:/data -B /scratch/gdehol:/workflow --cleanenv /shares/zne.uzh/containers/mriqc-24.0.0 /data /data/derivatives/mriqc participant --participant_label $PARTICIPANT_LABEL -w /workflow --no-sub
