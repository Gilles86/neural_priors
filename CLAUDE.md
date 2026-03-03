# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**neural_priors** is a neuroimaging research project studying how the brain encodes numerical information. It combines behavioral data, fMRI imaging, and computational modeling using population receptive field (PRF) models. The BIDS-formatted dataset lives at `/data/ds-neuralpriors` (local) or `/shares/zne.uzh/gdehol/ds-neuralpriors` (cluster).

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
source activate neural_priors2

# Install the package in editable mode
pip install -e .
```

Key dependencies: Python 3.9, TensorFlow 2.14, TensorFlow Probability, braincoder (PRF fitting), nilearn, nipype, GLMsingle.

## Running Scripts

Scripts are run directly as Python modules with subject IDs and BIDS folder paths:

```bash
# Fit encoding model for subject 01 with model 31, smoothed data
python neural_priors/encoding_model/fit_model.py 01 \
    --bids_folder /data/ds-neuralpriors --model 31 --smoothed

# Cross-validation fit
python neural_priors/encoding_model/fit_model_cv.py 01 \
    --bids_folder /data/ds-neuralpriors --model 31 --smoothed

# Decode stimuli from brain activity
python neural_priors/encoding_model/decode.py 01 \
    --bids_folder /data/ds-neuralpriors --model_label 31 --n_voxels 200

# Extract parameters across subjects
python neural_priors/encoding_model/extract_pars.py \
    --bids_folder /data/ds-neuralpriors --model 31
```

## SLURM Cluster Jobs

Jobs are submitted as SLURM arrays (one job per subject). Scripts are in `neural_priors/encoding_model/slurm_jobs/`:

```bash
# Submit fitting job for all subjects (array 1-41), model 31
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model.sh 31 --smoothed

# Submit cross-validation
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model_cv.sh 31 --smoothed

# Submit decoding (requires model label and n_voxels)
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/decode.sh 31 200

# Batch submission of multiple models/configurations
bash neural_priors/encoding_model/slurm_jobs/submit_jobs_2026-02-26.sh
```

Common flags: `--smoothed`, `--fit_responses`, `--censored`, `--spherical_noise`, `--separate_sigmas`.

## Architecture

### Data Access: `Subject` class (`neural_priors/utils/data.py`)

All data access goes through `Subject(subject_id, bids_folder)`. Key methods:
- `get_behavioral_data()` — trial-wise stimulus values and responses
- `get_single_trial_estimates()` — GLMsingle fMRI single-trial amplitudes
- `get_prf_parameters_volume(model_label)` — fitted encoding model parameters per voxel
- `get_brain_mask()` / `get_volume_mask(roi)` — ROI masks (NPCr, NF, NTO regions)
- `get_confounds()` — fMRIPrep motion/acquisition confounds

### Encoding Models (`neural_priors/encoding_model/models.py`)

Two model classes extending `braincoder.models.AlphaGaussianPRF`:

1. **`AlphaDeltaModel`** — Box-Cox power transformation of stimulus space. Parameters: `mu_narrow`, `alpha`, `delta_wide`, `lower_bound_range`, `sd`, `amplitude`, `baseline`. Supports `separate_amplitudes`, `separate_baselines`, `separate_sds`, `rescale_baseline`, `identity_below_range`.

2. **`LinearScalingModel`** (primary production model) — Log-normal tuning curves with linear scaling between narrow/wide conditions. `mu_wide = delta_wide * (mu_narrow - lower_bound_range) + lower_bound_range`. Supports `sd_natural`, `sigma_fwhm`, and the same flags as above.

Both models use **softplus transforms** on positive parameters for numerical stability in TensorFlow gradient descent.

### Model Variants (`neural_priors/encoding_model/fit_model.py`)

35+ numbered model configurations (see comments at top of `fit_model.py`):
- **Models 0–14**: `AlphaDeltaModel` variants testing different parameter constraints
- **Models 15–35**: `LinearScalingModel` variants
- **Models 100+** (e.g., 115–145): `LinearScalingModel` with fixed `sd_ratio` slope values between 1.15–1.45

Use `get_model(model_label)` to instantiate the correct model class/configuration.

### Two-Stage Fitting Pipeline (`neural_priors/encoding_model/fit_model.py`)

1. **Grid search** (`ParameterFitter.fit_grid()`): correlation-based cost, searches parameter space for good initialization
2. **Gradient descent** (`ParameterFitter.fit()`): up to 5000 iterations; parameters classified as fixed (constant), shared (one value across voxels), or per-voxel

### Decoding (`neural_priors/encoding_model/decode.py`)

Bayesian decoding: fits a residual noise model (Student's t), then computes the posterior PDF over stimulus values for held-out trials. Supports separate sigmas per condition and spherical vs. full covariance noise.

### Cross-Validation (`neural_priors/encoding_model/fit_model_cv.py`)

Splits data by `(session, run2)` pairs into train/test folds. Returns cross-validated R² per voxel, used for model comparison and voxel selection in decoding.

## Output Structure

Fitted parameters and results are written back into the BIDS derivatives tree under `derivatives/encoding_model/` as NIfTI volumes and TSV/CSV files, organized by subject, session, and model label.
