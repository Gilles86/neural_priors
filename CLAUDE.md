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

Key dependencies: Python 3.9, TensorFlow 2.15, TensorFlow Probability, braincoder (PRF fitting), nilearn, nipype, GLMsingle. Pinned specs: `environment.yml` (cluster) / `environment_silicon.yml` (local Mac); exact snapshots in `environments/*.lock.yml`.

## Running Scripts

Scripts are run directly as Python modules with subject IDs and BIDS folder paths:

```bash
# Fit encoding model for subject 01 with model 31, smoothed data
python neural_priors/encoding_model/fit_model.py 01 \
    --bids_folder /data/ds-neuralpriors --model_label 31 --smoothed

# Cross-validation fit
python neural_priors/encoding_model/fit_model_cv.py 01 \
    --bids_folder /data/ds-neuralpriors --model_label 31 --smoothed

# Decode stimuli from brain activity (production configuration)
python neural_priors/encoding_model/decode.py 01 \
    --bids_folder /data/ds-neuralpriors --model_label 31 --smoothed \
    --fit_responses --spherical_noise --n_voxels 0

# Extract parameters across subjects (writes to derivatives/extracted_pars/; model label is positional)
python neural_priors/encoding_model/extract_pars.py 31 \
    --bids_folder /data/ds-neuralpriors --smoothed

# Write all parameters for main models into a single long-format summary TSV
# (writes to derivatives/summary_tsvs/main_models_roi-NPCr_desc-groundtruth_parameters.tsv.gz)
python neural_priors/encoding_model/write_parameters_summary.py --smoothed

# Run for a subset of models only
python neural_priors/encoding_model/write_parameters_summary.py --smoothed --models 3,5
```

## SLURM Cluster Jobs

Jobs are submitted as SLURM arrays (one job per subject). Scripts are in `neural_priors/encoding_model/slurm_jobs/`:

```bash
# Submit fitting job for all subjects (array 1-41), model 31
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model.sh 31 --smoothed

# Submit cross-validation
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model_cv.sh 31 --smoothed

# Submit decoding (requires model label and n_voxels; production config uses n_voxels=0)
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/decode.sh 31 0 --smoothed --fit_responses --spherical_noise

# Batch submission of the width-scaling sweep (models 115-135)
bash neural_priors/encoding_model/slurm_jobs/submit_fixed_sd_scaling_jobs.sh
```

Common flags: `--smoothed`, `--fit_responses`, `--censored`, `--spherical_noise`, `--separate_sigmas`.

## Architecture

### Data Access: `Subject` class (`neural_priors/utils/data.py`)

All data access goes through `Subject(subject_id, bids_folder)`. Key methods:
- `get_behavioral_data()` — trial-wise stimulus values and responses
- `get_single_trial_estimates()` — GLMsingle fMRI single-trial amplitudes
- `get_prf_parameters_volume(model_label, use_nifti=False)` — fitted encoding model parameters per voxel. By default reads from pre-extracted TSVs in `derivatives/extracted_pars/` (fast); pass `use_nifti=True` to load directly from NIfTI files. Run `extract_pars.py` first to populate the TSV cache.
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

Output PDF files have MultiIndex columns `(n, range)` where `n` is numerosity and `range` is 0.0 (narrow) or 1.0 (wide). Read with `pd.read_csv(..., header=[0,1], index_col=[0,1])`.

### Cross-Validation (`neural_priors/encoding_model/fit_model_cv.py`)

Splits data by `(session, run2)` pairs into train/test folds. Returns cross-validated R² per voxel, used for model comparison and voxel selection in decoding.

## Main Model Labels

The canonical set of models used in analyses, with descriptive labels:

| Label | Description |
|-------|-------------|
| 0  | No shift (μ_wide = μ_narrow) |
| 1  | Fixed shift (δ=2, no range constraint) |
| 2  | Fitted shift ratio, shared across voxels |
| 3  | Fitted shift ratio, free per voxel |
| 4  | Efficient coding: fixed shift (δ=2) |
| 5  | Efficient coding: shared shift ratio |
| 14 | Free width ratio, per voxel |
| 15 | Fitted width scaling, shared across voxels |
| 31 | Fixed width scaling (δ_σ=1.29) — primary production model |
| 32 | Fixed width scaling + free amplitude ratio |
| 33 | Fixed width scaling + shared amplitude ratio |
| 34 | Fixed tuning, free amplitude per voxel |
| 35 | Fixed tuning, shared amplitude ratio |

"Shared" = one value per participant across all voxels. "Free" = per voxel. "Fixed" = constant.

## Output Structure

Fitted parameters and results are written into the BIDS derivatives tree:
- `derivatives/encoding_models/model{N}.smoothed/sub-{id}/func/` — per-subject NIfTI parameter maps
- `derivatives/extracted_pars/group_roi-{roi}_model-{N}_desc-{desc}_parameters.tsv` — pre-extracted per-model group TSVs (created by `extract_pars.py`)
- `derivatives/summary_tsvs/main_models_roi-{roi}_desc-{desc}_parameters.tsv.gz` — long-format table of all parameters for all main models (created by `write_parameters_summary.py`; columns: `subject`, `model_label`, `model`, `response_fit`, `voxel`, plus all parameter columns)
- `derivatives/decoding2/model{N}.smoothed/sub-{id}/func/` — decoding PDF files

## Rebuttal / response-letter drafts (`notes/`)

Draft texts for reviewer responses and Supplementary Information live in `notes/*.md`. `notes/` is **gitignored** (reviewer correspondence and paper PDFs must not reach the public repo). Formatting convention: **no hard line wrapping — write each paragraph as a single line**, so paragraphs can be copy-pasted directly into the rebuttal letter / manuscript. Headings, table rows, and list items keep their own lines as usual.
