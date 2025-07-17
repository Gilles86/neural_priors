#!/bin/bash
#SBATCH --job-name=create_tf2_gpu
#SBATCH --output=/home/gdehol/logs/create_tf2_gpu_%j.txt
#SBATCH --error=/home/gdehol/logs/create_tf2_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1  # Request a GPU node

echo "🚀 Initializing Conda..."
source $HOME/init_conda.sh

echo "🔄 Removing existing Conda environment: tf2-gpu"
mamba env remove -n tf2-gpu --yes

echo "🚀 Creating new Conda environment: tf2-gpu"
mamba create -n tf2-gpu python=3.10 \
    cuda-version=12.6 cudnn=8.9 \
    tensorflow tensorflow-probability pandas tqdm nilearn \
    -c conda-forge -c nvidia --yes

echo "✅ Conda environment tf2-gpu successfully created!"

echo "🔎 Checking TensorFlow installation..."
source activate tf2-gpu
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

echo "🎉 Done!"
