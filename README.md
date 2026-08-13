# Distributed range adaptation in human parietal encoding of numbers

**Arthur Prat-Carrabin\*, Gilles de Hollander\*, Saurabh Bedi, Samuel J. Gershman, Christian C. Ruff**
(\*equal contribution)

bioRxiv preprint: https://doi.org/10.1101/2025.09.25.675916

This repository contains all analysis code for the paper. It is organised so that every figure and reported statistic can be traced to a specific script or notebook (see [From paper result to code](#from-paper-result-to-code)).

---

## Overview

39 participants estimated the number of dots in visual displays while undergoing 3T fMRI, across two contexts that differed in the *range* of possible numerosities (Narrow: 10–25; Wide: 10–40). Using numerical population receptive field (nPRF) models fitted to single-trial BOLD responses, we show that:

1. **Preferred numerosities shift** between conditions following a quantitative efficient-coding prediction: μ_wide = 10 + 2(μ_narrow − 10) for μ_narrow ≥ 10 (slope = ratio of prior widths = 2).
2. **Receptive fields widen** in the Wide condition, with a uniform scaling factor of ~1.3 across participants.
3. **Neural encoding precision decreases** under the wider prior, consistent with dynamic efficient coding.
4. **Individual differences** in neural adaptation correlate with individual differences in behavioral variability.

## Installation

Clone with submodules (the [braincoder](https://github.com/Gilles86/braincoder) nPRF-fitting library is vendored as a submodule):

```bash
git clone --recursive https://github.com/ruffgroup/neural_priors.git
cd neural_priors
```

(`github.com/Gilles86/neural_priors`, the URL given in earlier versions of the paper, is the same repository — kept as a mirror.)

Two conda environments are provided. **These pinned environment files are the environments of record** — the versions listed are the ones that produced the published results.

| File | Purpose |
|------|---------|
| `environment.yml` | Linux / cluster (`neural_priors2`): all model fitting, decoding, GLMsingle. TensorFlow 2.15 (CUDA build on the cluster). |
| `environment_silicon.yml` | Local macOS (Apple Silicon, `neural_priors_silicon`): figure notebooks, revision analyses, small simulations. TF-metal stack. |
| `environments/*.lock.yml` | Full `conda env export` snapshots of the actual environments used (byte-level record). |

```bash
conda env create -f environment.yml        # or environment_silicon.yml on a Mac
conda activate neural_priors2              # or neural_priors_silicon
```

`pip install -e .` (the project package) is included in both environment files. The cluster environment pins braincoder to the exact commit used for the paper fits (`ed48b10`, 2026-03-09); for development, install the submodule editable instead:

```bash
pip install -e libs/braincoder
```

Software *outside* these environments: fMRIPrep 23.2.1 and MRIQC 24.0.0 (run as Singularity containers, see `neural_priors/cluster_preproc/`), FreeSurfer 7.3.2 (bundled inside fMRIPrep), CmdStan ≥ 2.35 (behavioral model, `notebooks/stan/`), and [pycortex](https://github.com/gallantlab/pycortex) (surface visualization only).

## Data

The BIDS dataset will be publicly available on OpenNeuro (contact the authors in the meantime). All scripts take `--bids_folder` (default `/data/ds-neuralpriors`).

- 39 participants (`sub-01` … `sub-41`; `sub-11` and `sub-23` excluded for having only one session — see `neural_priors/data/subjects.yml`)
- 2 sessions × 8 runs × 30 trials = 480 trials per participant (240 per condition; one condition per session block)
- 3T Philips Achieva, 2.5 mm EPI, TR = 2.286 s
- Trial-wise events in `sub-*/ses-*/func/*_events.tsv`; raw behavior in `sourcedata/behavior/`

Key derivatives (each produced by a pipeline stage below):

```
derivatives/
├── fmriprep/                      # Stage 1: preprocessed BOLD (T1w space), FreeSurfer surfaces
├── glm_stim1.denoise.smoothed/    # Stage 2: GLMsingle single-trial betas (desc-stim_pe.nii.gz)
├── ips_masks/                     # Stage 3: NPC ROI masks in T1w space
├── encoding_models/               # Stage 4: per-voxel nPRF parameter maps, model{N}[.cv][...]/
├── extracted_pars/                # Stage 5: per-model group parameter TSVs
├── summary_tsvs/                  # Stage 5: long-format all-model summary TSV
├── decoding2/                     # Stage 6: per-trial posterior PDFs over numerosity
├── fisher_information2/           # Stage 7: Fisher information per numerosity/condition
└── expected_uncertainty/          # Stage 7: simulated ideal-decoder error distributions
```

All data access in analysis code goes through the `Subject` class (`neural_priors/utils/data.py`) — e.g. `Subject(12).get_single_trial_estimates(...)`, `.get_prf_parameters_volume(model_label=31)`, `.get_behavioral_data()`. Scripts never build BIDS paths themselves.

## Analysis pipeline

Stages 1–7 ran on a SLURM cluster (array jobs, one task per subject: `sbatch --array=1-41 <script>`); figures and statistics ran locally. Every stage writes into the derivatives tree above.

### Stage 0 — Raw data → BIDS (provenance only)

`neural_priors/prepare/convert_raw_mri_data.py` (Philips PAR/REC → BIDS, synthesizes opposite-phase-encoding fieldmaps) and `neural_priors/prepare/make_events_files.py` (PsychoPy logs → `_events.tsv`). The OpenNeuro dataset ships already converted; these scripts document how.

### Stage 1 — Preprocessing (fMRIPrep)

```bash
sbatch --array=1-41 neural_priors/cluster_preproc/fmriprep.sh
```

fMRIPrep 23.2.1, output spaces `T1w MNI152NLin2009cAsym fsaverage fsnative`, `--dummy-scans 4`. QC with `mriqc.sh` / `mriqc_group.sh`.

### Stage 2 — Single-trial BOLD amplitudes (GLMsingle)

```bash
# session argument 0 = both sessions concatenated (production setting)
python neural_priors/glm/fit_single_trials_denoise.py 1 0 --smoothed --bids_folder /data/ds-neuralpriors
# cluster: sbatch --array=1-41 neural_priors/glm/cluster_jobs/submit_glm_both_smoothed.sh
```

GLMsingle (HRF library + GLMdenoise + fractional ridge) on 5-mm-smoothed T1w-space BOLD; one regressor per stimulus and per response event. Output: 480 stimulus betas per voxel (`desc-stim_pe.nii.gz`).

### Stage 3 — ROI definition

```bash
python neural_priors/surface/get_npc_mask.py 1 --roi NPC
```

Projects the group-level right NPC surface label (taken from Barretto-García et al. 2023) to each subject's T1w volume via fsaverage → fsnative → volume. All main analyses use the right NPC (`NPCr`).

### Stage 4 — nPRF model fitting

```bash
# The best-fitting 'Efficient shifts, fixed width-scaling' model (31) on NPCr:
python neural_priors/encoding_model/fit_model.py 12 --model_label 31 --smoothed
python neural_priors/encoding_model/fit_model_cv.py 12 --model_label 31 --smoothed   # 8-fold cvR²

# Cluster:
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model.sh 31 --smoothed
sbatch --array=1-41 neural_priors/encoding_model/slurm_jobs/fit_model_cv.sh 31 --smoothed
# Whole-brain (for surface maps): fit_model_whole_brain.sh / fit_model_cv_whole_brain.sh (GPU)
# Width-scaling sweep (models 115–135): submit_fixed_sd_scaling_jobs.sh
```

Two-stage fit per voxel: correlation-based grid search + OLS for amplitude/baseline, then joint ADAM gradient descent on relative R² (see Methods, "nPRF estimation"). Every model is fitted both to the presented numerosity (*ground truth*) and to the participant's estimate (`--fit_responses`). Cross-validation re-indexes runs as `run2 = (run−1) % 4 + 1`, pairing runs *k* and *k+4* of the same session; each of the 8 (session, run2) folds holds out one narrow and one wide run (the two conditions occupy the two halves of each session). Model definitions: `neural_priors/encoding_model/models.py`; the full numbered-variant registry: `neural_priors/encoding_model/MODELS.md`.

### Stage 5 — Parameter extraction and model comparison

```bash
python neural_priors/encoding_model/calc_likelihood.py 12 31 --smoothed     # voxelwise log-likelihood (for BIC)
python neural_priors/encoding_model/extract_pars.py 31 --smoothed           # → derivatives/extracted_pars/
python neural_priors/encoding_model/write_parameters_summary.py --smoothed  # → derivatives/summary_tsvs/
# batch versions: slurm_jobs/submit_likelihoods.sh, slurm_jobs/extract_pars_*.sh
```

The summary TSV (long format: one row per subject × model × voxel, with AIC/BIC) is the single input for all group-level parameter figures.

### Stage 6 — Bayesian decoding

```bash
# Production configuration used for the decoding statistics in the paper:
python neural_priors/encoding_model/decode.py 12 --model_label 31 --smoothed \
    --fit_responses --spherical_noise --n_voxels 0
# cluster: sbatch --array=1-41 slurm_jobs/decode.sh 31 0 --smoothed --fit_responses --spherical_noise
```

Per cross-validation fold: refit the encoding model on training runs, select voxels with cvR² > 0 (`--n_voxels 0`), fit a Student-*t* residual noise model, and evaluate the posterior over the numerosity grid for each held-out trial. Output: one TSV of per-trial posterior PDFs per subject (MultiIndex columns `(n, range)`; read with `pd.read_csv(..., header=[0,1], index_col=[0,1,2,3,4])`).

### Stage 7 — Encoding precision (Fisher information & ideal-decoder simulations)

```bash
python neural_priors/encoding_model/get_fisher_info.py 12 --model_label 31 --smoothed --spherical_noise
python neural_priors/encoding_model/get_expected_uncertainty.py 12 --model_label 31 --smoothed --spherical_noise
```

`get_fisher_info.py` computes the encoding Fisher information per numerosity and condition via braincoder (analytical Gaussian form; Fig. 5a left). `get_expected_uncertainty.py` simulates 20,000 responses per numerosity from the fitted generative model and decodes them back (posterior mean), yielding the expected variability of an ideal Bayesian decoder (Fig. 5a right, 5b).

### Stage 8 — Surface maps (Fig. 2)

```bash
python neural_priors/visualize/import_freesurfer_subject.py 12        # once per subject: pycortex setup
python neural_priors/surface/sample_prf_to_surface_nilearn.py 12 31 --smoothed
# then, group GIfTIs: neural_priors/visualize/combine_subjects.ipynb
cd neural_priors/visualize                                            # scripts use `from utils import ...`
python visualize_group.py 31            # Fig. 2a (group R² on fsaverage)
python visualize_subject_model.py 12 31 # Fig. 2b (individual preferred-numerosity maps)
# colorbars: make_colorbars.ipynb
```

### Stage 9 — Behavioral model (Stan; Fig. 1b)

```bash
# 1. Build Stan input JSONs:   notebooks/stan/make_stan_json_files.ipynb
# 2. Fit notebooks/stan/hmodel.stan with CmdStan (10 chains × 1000 samples, 1000 warmup, HMC-NUTS)
# 3. Post-process posterior:   notebooks/stan_output.ipynb
```

The hierarchical model estimates group- and subject-level response mean/SD per numerosity and condition (Methods Eq. 8–9). **Note:** subjects are re-indexed 1…39 in the Stan data (BIDS IDs with gaps at 11 and 23 are not preserved) — `make_stan_json_files.ipynb` documents the mapping.

## From paper result to code

| Paper result | Produced by | Upstream inputs |
|---|---|---|
| Fig. 1a (task) | `experiment/` (PsychoPy task) | — |
| Fig. 1b (response variability) + behavioral t-tests | `notebooks/stan_output.ipynb`, `neural_priors/figures/figure1.ipynb` | Stan fit (stage 9) |
| Fig. 1c–e (priors, adaptation schematic) | `neural_priors/figures/figure1.ipynb` | — |
| Fig. 2a (group R² surface map) | `neural_priors/visualize/visualize_group.py` | stages 4 (whole-brain) + 8 |
| Fig. 2b (individual μ maps, S12/S20/S26) | `neural_priors/visualize/visualize_subject_model.py` | stages 4 + 8 |
| Fig. 3a–f (efficient shifts of μ; panel f = cvR² comparison of models 0/3/4/5; also the pooled free-shift ρ≈0.77 and the proportional-shift SI models 1/2) | `notebooks/fmri_models_analysis.ipynb` § "Figure - preferred numerosities" (whole composite incl. panel f) and § "Model 3 - Free mus" | summary TSV (stage 5) |
| Fig. 4a–f (width scaling; panel f = cvR² comparison of models 4/14/31/15, "No width change" = model 4) | `notebooks/fmri_models_analysis.ipynb` § "Figure - widths" (whole composite incl. panel f) | summary TSV; note the composite is saved in a ground-truth and a `_gt`-suffixed variant via the `WITH_GROUND_TRUTH` switch — the paper uses the response-fit variant |
| Fig. S1 (extended 14-model cvR² comparison) | `notebooks/fmri_models_analysis.ipynb` § "Figure S1" (`models_fits.pdf`); cross-check in `neural_priors/revision feb 2026/cvr2.ipynb` | summary TSV |
| Amplitude analyses (SI; models 32–35, "no evidence amplitudes change") | `notebooks/fmri_models_analysis.ipynb` (t-tests models 32/33 vs 31) + `notebooks/likelihoods_and_bics.ipynb` (BIC, model 35) | summary TSV, `all_likelihoods.tsv` |
| δ_σ = 1.29 (model 31's fixed width scaling) | `neural_priors/revision feb 2026/fit_sd_scale.ipynb` | model 115–135 sweep (stage 4) |
| Fig. 5a (Fisher info / ideal-decoder imprecision) | `notebooks/behavior_vs_fmri.ipynb` | stage 7 outputs (`--spherical_noise` configuration) |
| Fig. 5b (variability of fMRI-derived estimates) | `notebooks/behavior_vs_fmri.ipynb` | `get_expected_uncertainty.py` (incl. `--wide` control) |
| Fig. 5c–f (behavior ↔ fMRI variability correlations) | `notebooks/behavior_vs_fmri.ipynb` (`fmri_beh_variability.pdf`) | Stan fit + stage 7 |
| Fig. 5g–i (Weber compression ω) | `notebooks/behavior_vs_fmri.ipynb` | same |
| Decoding statistics (Results, rm-corr / MAE / censored-at-25 comparison) | `neural_priors/encoding_model/analyze_decoding_fit_responses.ipynb` (prints a "MANUSCRIPT STATS" block) | stage 6 |
| Trial-wise decoded measures (feeds Stan fMRI variants) | `neural_priors/encoding_model/get_trialwise_neural_measures.ipynb` | stage 6 |
| BIC model comparison (SI) | `notebooks/likelihoods_and_bics.ipynb` (also contains a plain-English glossary of every model label) | stage 5 |
| Example voxel fits (SI) | `neural_priors/revision feb 2026/fit_examples.ipynb` | stages 2 + 4 |
| Model recovery, shift parameter (Fig. S8) | `neural_priors/revision feb 2026/simulate_data.py` + `recovery_bias.ipynb` | model-3 fits |
| Model recovery, width parameter (Fig. S9) | `neural_priors/revision aug 2026/simulate_data_sd.py` + `recovery_bias_sd.ipynb`; final figure layouts in `plot_recovery_figures.py`, per-voxel μ-recovery panel in `plot_pervoxel_mu_figure.py` | model-15 fits |
| Censored-fit robustness checks (SI) | `neural_priors/revision feb 2026/censored_fits_mu.ipynb`, `censored_fits_sd.ipynb` | stage 4 with `--censored` |
| Left-NPC replication (SI) | `neural_priors/revision feb 2026/NPCl.ipynb` | stage 4 on NPCl |
| Retinotopic-confound controls (Discussion + SI) | `neural_priors/revision feb 2026/retinotopic_mus.ipynb`; SI figure (round 2): `neural_priors/revision aug 2026/plot_retinotopy_figure.py` | summary TSV |
| Per-participant behavioral estimate/SD curves (SI) | `notebooks/stan_output.ipynb` § "individual" (`subjs_indiv_estimates.pdf`, `subjs_indiv_sd.pdf`) | Stan fit (stage 9) |
| Median-cvR² model-comparison robustness check | `notebooks/fmri_models_analysis.ipynb` § "Median cvR2" (`models_fits_mediancvr2.pdf`) | summary TSV |
| Participant demographics (Methods) | `notebooks/participant_info.ipynb` | `participants.tsv` |

## Model labels

Models are numbered; `neural_priors/encoding_model/MODELS.md` documents all of them. The ones named in the paper:

| Label | Paper name | Constraint |
|-------|------------|-----------|
| 0  | No shift | μ_wide = μ_narrow |
| 3  | Free shifts | μ_narrow, μ_wide free per voxel |
| 4  | Efficient shifts | μ_wide = 10 + 2(μ_narrow − 10), fixed slope 2 |
| 5  | Participant-specific slopes | as 4, but slope r_μ fitted per participant |
| 14 | Free widths | σ_narrow, σ_wide free per voxel |
| 15 | Participant-specific scaling | σ_wide = r_σ·σ_narrow, r_σ fitted per participant |
| **31** | **Efficient shifts, fixed width-scaling** | **slope 2 and σ_wide = 1.29·σ_narrow, both fixed — best-fitting model** |
| 32/33 | + free / shared amplitude ratio | amplitude controls |
| 34/35 | Fixed tuning, amplitude-only adaptation | amplitude controls |
| 115–145 | Width-scaling sweep | σ scaling fixed at label/100 (1.15 … 1.45) |

## Reproducibility notes

- **Environments**: use the pinned `environment*.yml`; exact snapshots in `environments/*.lock.yml` (exported 2026-08-10 from the environments in which the production fits and figures were run). The paper's cluster fits used braincoder commit `ed48b10` (pinned in `environment.yml`); the submodule tracks a slightly later commit of the same line with build/CI fixes only. The GLMsingle stage ran in a separate cluster env (glmsingle 1.2, declared in `environment.yml` but absent from the fitting-env lock file).
- **Ground truth vs. responses**: tuning analyses (Figs. 3, 4) use the `--fit_responses` variants; brain–behavior analyses (Fig. 5) use ground-truth fits to keep neural and behavioral noise independent (see Methods).
- **Voxel inclusion**: all parameter analyses are restricted to voxels with 8-fold cvR² > 0.
- **Stan outputs**: the Stan-fit summary CSVs read by `stan_output.ipynb` / `behavior_vs_fmri.ipynb` are date-stamped run outputs (`notebooks/stan/outputs/`); rerun stage 9 to regenerate them.
- **Notebook working directories**: notebooks under `notebooks/` use paths relative to that directory; the pycortex scripts in `neural_priors/visualize/` must be run with that directory as the working directory.
- **NPC ROI**: the group surface labels (`derivatives/surface_masks/`) originate from Barretto-García et al. (2023, *Nature Human Behaviour*) and are distributed with the dataset, not this repository.

## Archived code

`archive/` contains superseded or exploratory code that does **not** correspond to any result in the paper (earlier figure versions, decoding-configuration sweeps, a Gaussian-process-prior fitting experiment, one-off pilots). It is kept for provenance; see `archive/README.md`. Nothing in the pipeline above depends on it.

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
