# Revision Aug 2026 — width-parameter model recovery

Addresses Reviewer 3's remaining comment (round 2): extend the model recovery
analysis of Fig. S8 (shift parameter δ_wide) to the **tuning-width scaling
parameter** (`sd_wide_scale`), and document the type/magnitude of the noise
used in the simulations.

- `simulate_data_sd.py` — runs the simulations (see below)
- `recovery_bias_sd.ipynb` — analysis/figures

## How the generating parameters were chosen

The guiding principle is the same as in the δ_wide recovery simulation
(`revision feb 2026/simulate_data.py`): **do not invent parameter values —
resample them from the empirical fits**, so that the simulated population has
the same distribution of preferred numerosities and tuning widths (including
"awkward" voxels, e.g. monotonic response profiles from off-range preferred
numerosities) as the real data.

### Two sampling variants

- **Pooled** (default): each iteration draws 250 voxels from the group-level
  pool described below, over all subjects.
- **Subject-wise** (`--sample_subject`): each iteration draws one random
  subject and simulates **all** of that subject's supra-threshold voxels
  (29–535 across subjects). This exactly mirrors the real model-15 fit, which
  estimates one width-scaling factor per subject over that subject's voxels,
  and lets recovery precision vary with per-subject voxel count as it does in
  the real data.

### Per-voxel parameters: resampled empirical (μ, σ) pairs

The generative pool consists of the model-15 fits in right parietal cortex
(NPCr, ground-truth stimulus space, smoothed data), restricted to voxels with
positive out-of-sample R² (cvr2 > 0): **8,785 voxels from 39 subjects**. On
each simulation iteration, 250 voxels are drawn from this pool without
replacement (seeded by iteration number), and their fitted
(`mu_narrow`, `sd_narrow`) values are used **jointly, as pairs**, so the
empirical joint distribution — including its (weak) dependence between
preferred numerosity and width (Spearman ρ ≈ −0.11) — is preserved.

Empirical distributions of the sampled parameters:

| Parameter | Median | Mean ± SD | 5–95% quantiles |
|---|---|---|---|
| `mu_narrow` (preferred numerosity, narrow) | 10.1 | 11.0 ± 5.6 | 4.3 – 17.0 |
| `sd_narrow` (log-space tuning width, narrow) | 0.55 | 0.61 ± 0.40 | 0.13 – 1.36 |

Notably, ~45% of the pool has a preferred numerosity *below* the presented
range (μ < 10) and ~2% above 25 — exactly the voxels for which the reviewer's
μ/width trade-off concern (monotonic raw responses) is most acute. These are
deliberately kept in the simulation.

### Condition-linking (group) parameters: fixed at their model values

| Parameter | Value | Rationale |
|---|---|---|
| `delta_wide` | 2.0 | Full shift of preferred numerosities in the wide range, as in the production models (15/31) where it is fixed |
| `sd_wide_scale` | **1.0** or **1.287794** | The recovery target. 1.0 = null (no widening). 1.287794 = mean across the 39 subjects of the model-15 shared width-scaling estimate (subject mean 1.288, SD 0.25) — the value hard-coded in production model 31 |
| `lower_bound_range` | 10 | Lower edge of the stimulus range (fixed in all models) |
| `amplitude` | 1.0 | Standardized (see noise section) |
| `baseline` | 0.0 | Standardized |

### Design

Trial counts and stimulus values mirror the δ_wide simulation:

- **Narrow condition**: numerosities 10–24, 10 trials each (150 trials).
- **Wide condition, "full" design**: numerosities 10–39, 5 trials each (150
  trials) — the actual experimental design.
- **Wide condition, "censored" design**: numerosities restricted to 10–24, 10
  trials each — control showing what happens when the wide range is never
  actually sampled.

### Noise

I.i.d. Gaussian noise with standard deviation **0.5** is added to every
simulated single-trial response (`model.simulate(..., noise=0.5)`). With the
standardized amplitude of 1 and baseline of 0, this corresponds to a peak
single-trial SNR of 2. This is the same noise model and magnitude as in the
Fig. S8 δ_wide recovery simulation. For reference, the empirical median fitted
amplitude in the generative pool is 0.66 (5–95%: 0.26–2.03), so the simulated
SNR sits within the empirically observed range.

### Fitting

Each simulated dataset is refit with the exact production pipeline
(`fit_model(15, ...)`): grid search with correlation cost followed by gradient
descent, with per-voxel `mu_narrow`, `sd_narrow`, `amplitude`, `baseline`;
`sd_wide_scale` fitted freely but **shared across voxels** (one value per
simulated "subject", as in model 15); `delta_wide` fixed at 2 and
`lower_bound_range` fixed at 10.

Outputs per iteration (in
`/data/ds-neuralpriors/simulated_recovery_sd/sd_scale_{v}/noise_{n}/design_{d}/`):

- `iteration-{i}_results.csv` — recovered shared `sd_wide_scale`
- `iteration-{i}_pervoxel.csv` — generative vs. recovered per-voxel
  `mu_narrow`/`sd_narrow` (plus recovered amplitude/baseline), for checking
  the reviewer's μ/width trade-off directly at the voxel level

100 iterations were run per cell of the 2 (sampling variant) × 2 (generative
`sd_wide_scale`) × 2 (design) grid, as a SLURM array on the sciencecluster
(`slurm_jobs/submit_recovery_sd.sh`, env `neural_priors_gp`); results were
rsynced back to the local BIDS folder for analysis in
`recovery_bias_sd.ipynb`.
