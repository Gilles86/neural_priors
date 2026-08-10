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
| `calc_likelihood.py` | Computes trial-wise log-likelihoods under the fitted noise model. |
| `get_expected_uncertainty.py` | Theoretical expected uncertainty from the fitted encoding model parameters. |
| `get_fisher_info.py` | Fisher information of the encoding model as a function of numerosity. |

---

## Analysis notebooks

| Notebook | Contents |
|----------|---------|
| `analyze_results.ipynb` | Main group-level analysis of fitted encoding model parameters. |
| `analyze_decoding.ipynb` | Decoding reliability across a sweep of model/voxel configurations. |
| `analyze_decoding_fit_responses.ipynb` | Decoding analysis for the primary model (model 31, smoothed, spherical noise, fit_responses). Reports rm_corr and MAE statistics for the manuscript. |
| `analyze_sigma.ipynb` | Analysis of tuning width parameters across the group. |
| `analyze_likelihoods.ipynb` | Trial-wise log-likelihood analysis. |
| `compare_decoding_methods.ipynb` | Comparison of decoding configurations (noise model, voxel selection). |
| `compare_decoding_lambda_baseline.ipynb` | Effect of regularisation (λ) and baseline subtraction on decoding. |
| `expected_uncertainty.ipynb` | Plots theoretical expected uncertainty from encoding model parameters. |
| `fisher_info.ipynb` | Fisher information curves from the fitted models. |
| `get_likelihood.ipynb` | Interactive exploration of trial-wise likelihoods. |
| `get_trialwise_neural_measures.ipynb` | Extracts trial-wise neural summary measures for regression analyses. |

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
| `extract_pars_115_135.sh` | Extract parameters for the fixed-slope model series (115–145) |
| `submit_*.sh` | Convenience scripts for batch-submitting multiple configurations |

---

## Model documentation

See **`MODELS.md`** for a complete description of all numbered model variants (0–36 and 115–145), their parameter constraints, and a parameter glossary.

The primary production model used in all main analyses is **model 31** (LinearScalingModel, fixed width scaling δ_σ = 1.29).
