#!/bin/bash
#SBATCH --job-name=create_tf2_cpu_env
#SBATCH --output=/home/gdehol/logs/create_tf2_cpu_env_%j.txt
#SBATCH --time=03:00:00  # 3 hours should be enough for CPU setup
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G  # Request 32GB RAM

echo "🚀 Initializing Conda..."
source $HOME/init_conda.sh
echo "✅ Done initializing Conda."

# Ensure Mamba is available
echo "🔄 Removing existing Conda environment: tf2-cpu"
mamba env remove -n tf2-cpu -y

echo "🚀 Creating new Conda environment: tf2-cpu (CPU-only TensorFlow)"
mamba create -n tf2-cpu -c conda-forge -y \
    python=3.10 \
    tensorflow \
    tensorflow-probability \
    pandas tqdm nilearn

echo "✅ Conda environment tf2-cpu successfully created!"

# Activate the environment
source activate tf2-cpu

# Check if TensorFlow correctly detects only CPU
echo "🔎 Checking TensorFlow installation..."
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"

echo "✅ Done! CPU-only environment is ready for use."
