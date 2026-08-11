# Archive

Code kept for provenance that does **not** correspond to any result in the
paper. Nothing in the active pipeline (see the root `README.md`) imports or
reads from this directory. Files were moved here with `git mv`, so their full
history is available via `git log --follow`.

| Folder | Contents | Why archived |
|--------|----------|--------------|
| `figures/` | `figure2/3/4.ipynb`, `amplitude.ipynb`, `shift_dispersion.ipynb`, `test.ipynb` | Original-submission figure notebooks, superseded by `notebooks/fmri_models_analysis.ipynb` (which reads the newer `summary_tsvs` pipeline). Several call APIs that were since renamed and no longer run as-is. |
| `visualize/` | `visualize2.py`, `visualize_cvr2.py`, `model_comparison.py` | Broken one-off pycortex scripts (removed API calls) and a near-duplicate of `visualize_group.py`. |
| `notebooks/` | `behavior.ipynb`, `behavior_gilles.ipynb` | 2024 behavioral-model exploration, superseded by the Stan hierarchical model (`notebooks/stan/`); and the scratch prototype of `figures/figure1.ipynb`. |
| `encoding_model/` | `analyze_results.ipynb`, `analyze_sigma.ipynb`, `fisher_info.ipynb`, `analyze_likelihoods.ipynb`, `get_likelihood.ipynb`, `analyze_decoding.ipynb`, `compare_decoding_methods.ipynb`, `compare_decoding_lambda_baseline.ipynb` (+ their PNG outputs), `slurm_jobs/submit_decode_lambda.sh` | Superseded group analyses (their roles moved to `write_parameters_summary.py` + the `notebooks/` figure track) and decoding-configuration sweeps (λ-regularization, baseline subtraction, voxel-count) that informed — but are not — the production decoding configuration (`--spherical_noise --n_voxels 0`). |
| `gp_prior/` | `fit_gp_prior.py`, `plot_gp_prior_results.py`, `plot_gp_prior_nyu.py`, `gp_prior_results.ipynb`, `GP_EXPERIMENTS.md`, `slurm_jobs/fit_gp_prior.sh` | A self-contained methods experiment (May 2026): hierarchical nPRF fitting with a Gaussian-process prior over cortical distance, as a possible alternative to spatial smoothing. Well-instrumented (per-run manifests with git SHAs; experiment registry in `GP_EXPERIMENTS.md`) but not part of the paper. Note: the two plotting scripts predate the `exp-<tag>` output layout and need their glob updated one level deeper before reuse. |
| `cluster_preproc/` | `submit_alina.sh`, `submit_alina2.sh` | One-off fMRIPrep runs for pilot subjects not in the study sample. |
| `glm/` | `submit_glm_ses-1_smoothed.sh`, `submit_glm_ses-2_smoothed.sh`, `submit_glm-ses1.sh` | Per-session GLMsingle submissions, superseded by the both-sessions scripts (`submit_glm_both*.sh`) used for the paper. |
| `utils/` | `plotting.py`, `make_images_subjects.py` | Unused plotting helper (depended on a removed `Subject` method) and a subject-anatomy GIF generator not used by any analysis. |
