#!/bin/bash
#
#SBATCH --job-name=mriqc
#SBATCH --output=/home/cluster/gdehol/logs/res_mriqc_%A-%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
export SINGULARITYENV_FS_LICENSE=$FREESURFER_HOME/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
singularity run --cleanenv /data/gdehol/containers/mriqc-0.16.1.simg /scratch/gdehol/ds-tmsrisk scratch/gdehol/ds-tmsrisk/derivatives/mriqc participant --participant-label $PARTICIPANT_LABEL  --verbose-reports -w /scratch/gdehol/work
