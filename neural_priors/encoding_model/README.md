# encoding_model

Code for fitting population receptive field (PRF) encoding models to fMRI data, decoding stimuli from brain activity, and extracting / summarising fitted parameters.

---

## Core scripts

| Script | Purpose |
|--------|---------|
| `models.py` | Model class definitions (`AlphaDeltaModel`, `LinearScalingModel`). See `MODELS.md` for a full description of all numbered variants. |
| `fit_model.py` | Two-stage fitting pipeline (grid search → gradient descent) for a given subject and model label. |
| `fit_model_cv.py` | Cross-validated fitting: splits data by `(session, run)` pairs and returns per-voxel cross-validated R². |
| `decode.py` | Bayesian decoding: inverts the encoding model to compute a posterior PDF over numerosity for each held-out trial. |
| `extract_pars.py` | Reads fitted NIfTI parameter maps and writes fast-access TSVs to `derivatives/extracted_pars/`. Run this before any group-level analysis. |
| `write_parameters_summary.py` | Collects per-subject TSVs across all main models into a single long-format group summary TSV. |
| `calc_likelihood.py` | Computes voxelwise log-likelihoods under the fitted noise model (input to AIC/BIC). |
| `get_expected_uncertainty.py` | Simulates responses from the fitted generative model and decodes them back: expected variability of an ideal Bayesian decoder (Fig. 5a right, 5b). |
| `get_fisher_info.py` | Fisher information of the encoding model per numerosity and condition, computed analytically via braincoder (Fig. 5a left). |

---

## Analysis notebooks

| Notebook | Contents |
|----------|---------|
| `analyze_decoding_fit_responses.ipynb` | Decoding analysis for the production configuration (model 31, smoothed, spherical noise, fit_responses, n_voxels=0). Prints the "MANUSCRIPT STATS" block (rm-corr, MAE, censored-at-25 comparison). |
| `expected_uncertainty.ipynb` | Aggregates `get_expected_uncertainty.py` outputs into a group table (consumed by `notebooks/behavior_vs_fmri.ipynb`). |
| `get_trialwise_neural_measures.ipynb` | Extracts trial-wise decoded posterior mean/SD, writes `derivatives/decoding2/decoding_pars.tsv` (consumed by the Stan fMRI variants and `behavior_vs_fmri.ipynb`). |

Superseded analysis notebooks (older parameter/decoding sweeps) live in `archive/encoding_model/` at the repository root.

---

## SLURM cluster scripts (`slurm_jobs/`)

Submit jobs as SLURM arrays (one job per subject). See `CLAUDE.md` at the repository root for example `sbatch` commands.

| Script | Purpose |
|--------|---------|
| `fit_model.sh` | Fit encoding model for all subjects |
| `fit_model_cv.sh` | Cross-validated fitting |
| `decode.sh` | Bayesian decoding |
| `extract_pars_uncensored.sh` | Extract parameters (uncensored models) |
| `extract_pars_censored.sh` | Extract parameters (censored models) |
| `extract_pars_115_135.sh` | Extract parameters for the fixed-slope model series (115–135) |
| `fit_model_whole_brain.sh` / `fit_model_cv_whole_brain.sh` | Whole-brain (GPU) fits for the surface maps |
| `calc_likelihood.sh` / `submit_likelihoods.sh` | Voxelwise log-likelihoods, all models × subjects |
| `submit_fixed_sd_scaling_jobs.sh` | Width-scaling sweep (models 115–135) behind the δ_σ = 1.29 choice |
| `submit_decode_model31.sh` | Batch submission of the model-31 decoding variants |

---

## Model documentation

See **`MODELS.md`** for a complete description of all numbered model variants (0–36 and 115–145), their parameter constraints, and a parameter glossary.

The primary production model used in all main analyses is **model 31** (LinearScalingModel, fixed width scaling δ_σ = 1.29).
