# Distributed range adaptation in human parietal encoding of numbers

**Arthur Prat-Carrabin\*, Gilles de Hollander\*, Saurabh Bedi, Samuel J. Gershman, Christian C. Ruff**
(\*equal contribution)

bioRxiv preprint: https://doi.org/10.1101/2025.09.25.675916

---

## Overview

This repository contains all analysis code for the paper. We study how the brain's number-encoding populations in human parietal cortex adapt their tuning when the range of possible numbers changes — a mechanism we call **distributed range adaptation**.

In brief: 39 participants estimated the number of dots in visual displays while undergoing 3T fMRI, across two contexts that differed in the *range* of possible numerosities (Narrow: 10–25; Wide: 10–40). Using numerical population receptive field (nPRF) models fitted to single-trial BOLD responses, we show that:

1. **Preferred numerosities shift** between conditions following a quantitative efficient-coding prediction: μ_wide = 10 + 2(μ_narrow − 10) for μ_narrow ≥ 10 (slope = ratio of prior widths = 2).
2. **Receptive fields widen** in the Wide condition, with a uniform scaling factor of ~1.3 across participants.
3. **Neural encoding precision decreases** under the wider prior, consistent with dynamic efficient coding.
4. **Individual differences** in neural adaptation correlate with individual differences in behavioral variability.

## Repository structure

```
neural_priors/
├── neural_priors/
│   ├── encoding_model/         # nPRF model fitting and analysis
│   │   ├── models.py               # AlphaDeltaModel and LinearScalingModel
│   │   ├── fit_model.py            # Model definitions, grid search, gradient descent fitting
│   │   ├── fit_model_cv.py         # 8-fold cross-validation fitting
│   │   ├── extract_pars.py         # Extract parameters → derivatives/extracted_pars/
│   │   ├── write_parameters_summary.py  # All-model summary TSV
│   │   ├── decode.py               # Bayesian decoding (Fisher info + posterior PDFs)
│   │   ├── calc_likelihood.py      # Log-likelihood computation
│   │   └── slurm_jobs/             # SLURM submission scripts for cluster
│   ├── glm/
│   │   └── fit_single_trials_denoise.py  # GLMsingle single-trial BOLD estimation
│   ├── utils/
│   │   └── data.py                 # Subject class, data loading utilities
│   ├── surface/                    # Surface projection utilities
│   └── revision feb 2026/          # Analysis notebooks for manuscript revision
│       ├── cvr2.ipynb              # Model comparison via cross-validated R²
│       └── censored_fits.ipynb     # Analyses with stimulus range censoring
├── experiment/                 # Psychophysics task code
├── CLAUDE.md                   # Developer guide for AI-assisted coding
└── environment.yml             # Conda environment specification
```

## Data

The BIDS-formatted dataset is available on OpenNeuro (link forthcoming; for access in the meantime, contact the authors). Locally it lives at:
- `/data/ds-neuralpriors` (local workstation)
- `/shares/zne.uzh/gdehol/ds-neuralpriors` (UZH cluster)

### Dataset summary
- 39 healthy participants (13 female, ages 18–34)
- 2 sessions per participant, ~480 trials total (240 per condition)
- 3T Philips Achieva, 32-channel head coil, 2.5 mm isotropic EPI
- Preprocessed with fMRIPrep 23.2.1; single-trial amplitudes estimated with GLMsingle

## Setup

```bash
conda env create -f environment.yml
conda activate neural_priors2
pip install -e .
```

Key dependencies: Python 3.9, TensorFlow 2.14, TensorFlow Probability, [braincoder](https://github.com/Gilles86/braincoder) (nPRF fitting), nilearn, nipype, GLMsingle.

## Analysis pipeline

The full analysis proceeds in four stages:

### 1. Single-trial GLM (GLMsingle)

Single-trial BOLD amplitudes are estimated using GLMsingle with cross-validated HRF selection, L2 regularization, and GLMDenoise noise regressors. Outputs live in `derivatives/glmsingle/`.

### 2. Fit nPRF encoding models

Models are fit per subject, per ROI (default: right NPC), using a two-stage procedure: correlation-based grid search followed by ADAM gradient descent (up to 5000 iterations).

```bash
# Fit best-fitting model (model 31: efficient shift + fixed width scaling) for subject 01
python neural_priors/encoding_model/fit_model.py 01 \
    --bids_folder /data/ds-neuralpriors --model_label 31 --smoothed

# Cross-validated fit for model comparison
python neural_priors/encoding_model/fit_model_cv.py 01 \
    --bids_folder /data/ds-neuralpriors --model_label 31 --smoothed
```

On the cluster, SLURM array jobs submit all 39 subjects at once:
```bash
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model.sh 31 --smoothed
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model_cv.sh 31 --smoothed
```

### 3. Extract parameters

Pre-extract parameters for all subjects into a single group TSV (required for fast loading):
```bash
python neural_priors/encoding_model/extract_pars.py \
    --model_label 31 --smoothed --bids_folder /data/ds-neuralpriors

# Or write all main models together with all parameters
python neural_priors/encoding_model/write_parameters_summary.py --smoothed
```

Outputs:
- `derivatives/extracted_pars/group_roi-NPCr_model-31_desc-groundtruth_parameters.tsv` — per-model group TSVs used by `Subject.get_prf_parameters_volume()`
- `derivatives/summary_tsvs/main_models_roi-NPCr_desc-groundtruth_parameters.tsv` — long-format table across all main models (columns: `subject`, `model_label`, `model`, `response_fit`, `voxel`, plus all parameter columns)

### 4. Bayesian decoding

Fits a residual Student's *t* noise model and computes posterior PDFs over numerosity for held-out trials. Used to derive Fisher information and simulate ideal decoder variability.

```bash
python neural_priors/encoding_model/decode.py 01 \
    --model_label 31 --smoothed --spherical_noise --n_voxels 0 \
    --bids_folder /data/ds-neuralpriors
```

## Model summary

Models are numbered; the key ones used in the paper:

| Label | Description |
|-------|-------------|
| 0  | No shift (μ_wide = μ_narrow) |
| 3  | Fitted shift ratio, free per voxel ('free-shift' model) |
| 4  | Efficient coding: fixed shift (δ=2) |
| 5  | Efficient coding: shared shift ratio ('participant-specific slopes') |
| 14 | Free width ratio, per voxel ('free-widths' model) |
| 15 | Fitted width scaling, shared across voxels ('participant-specific scaling') |
| **31** | **Fixed width scaling (δ_σ=1.29) — best-fitting model** |
| 32 | Fixed width scaling + free amplitude ratio |
| 33 | Fixed width scaling + shared amplitude ratio |

Model 31 encodes the paper's main finding: μ_wide = 10 + 2(μ_narrow − 10) with σ_wide = 1.29 · σ_narrow, all fixed from the group-level means of models 4 and 15.

See `neural_priors/encoding_model/fit_model.py` for the full list of 35+ model variants.

## Citation

```bibtex
@article{pratcarrabin2025distributed,
  title   = {Distributed range adaptation in human parietal encoding of numbers},
  author  = {Prat-Carrabin, Arthur and de Hollander, Gilles and Bedi, Saurabh
             and Gershman, Samuel J. and Ruff, Christian C.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.09.25.675916}
}
```

## License

Code: MIT. Data: see OpenNeuro dataset page.
