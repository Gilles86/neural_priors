# Encoding Model Variants

This document describes all numbered model variants used in `fit_model.py`.

---

## Core Response Function

Both model classes predict voxel responses as:

```
response(x) = f(x | μ_cond, σ_cond) × amplitude_cond + baseline_cond
```

where `x` is the stimulus (numerosity), and `_cond` indicates narrow or wide range condition.

### Preferred numerosity shift between conditions

Both models use the same linear shift rule:

```
μ_wide = δ × (μ_narrow − lb) + lb
```

- `δ` (`delta_wide`): shift ratio; δ=1 means no shift, δ=2 means proportional doubling from lb
- `lb` (`lower_bound_range`): reference/boundary point (typically 10)
- **identity_below_range**: if `μ_wide < lb`, clamp to `μ_wide = μ_narrow` (no shift for neurons tuned below the range boundary)

---

## AlphaDeltaModel (models 0–14)

Tuning curve is a **Box-Cox Gaussian** in transformed stimulus space:

```
f(x | μ, σ, α) = exp( −(α(x) − α(μ))² / 2σ² )
  where  α(x) = (x^α − 1) / α   →  log(x) as α → 0
```

- `alpha` (γ) controls the shape of the transformation (Box-Cox power)
- Most variants fix α≈0 (log-normal); models 6–8 fit α as a shared parameter

| Model | δ | α | lb | Width | Amplitude | Notes |
|-------|---|---|----|----|-----------|-------|
| 0  | 1.0 (fixed) | ~0 (fixed) | free | single sd | single | Null model — no shift |
| 1  | 2.0 (fixed) | ~0 (fixed) | fixed | single sd | single | Fixed shift, no range constraint |
| 2  | shared | ~0 (fixed) | fixed | single sd | single | Fitted shift, shared across voxels |
| 3  | free | ~0 (fixed) | free | single sd | single | Fitted shift, free per voxel |
| 4  | 2.0 (fixed) | ~0 (fixed) | fixed | single sd | single | Efficient coding: fixed shift + identity_below_range |
| 5  | shared | ~0 (fixed) | fixed | single sd | single | Efficient coding: shared shift |
| 6  | 2.0 (fixed) | shared | free | single sd | single | Like 1, but α fitted (shared) |
| 7  | 2.0 (fixed) | shared | fixed | single sd | single | Like 4, but α fitted (shared) |
| 8  | shared | shared | fixed | single sd | single | Like 5, but α fitted (shared) |
| 9  | free | ~0 (fixed) | free | single sd | single | Like 3, baseline fixed at 0 |
| 10 | free | ~0 (fixed) | free | single sd | narrow + wide | Like 9, separate amplitudes |
| 11 | free | ~0 (fixed) | free | single sd | narrow + wide | Like 10, baseline rescaled by amplitude ratio (shared ratio) |
| 12 | 2.0 (fixed) | ~0 (fixed) | fixed | single sd | narrow + wide | Like 4, separate amplitudes + separate baselines |
| 13 | 2.0 (fixed) | ~0 (fixed) | fixed | single sd | narrow + wide | Like 4, separate amplitudes + rescale baseline |
| 14 | 2.0 (fixed) | ~0 (fixed) | fixed | narrow + wide | single | Like 4, separate sd per condition |

---

## LinearScalingModel (models 15–36)

Tuning curve is a **log-normal** (α=0 limit of AlphaDeltaModel):

```
f(x | μ, σ) = exp( −(log x − log μ)² / 2σ² )   [default: σ in log space]
```

Three σ parameterisations:
- **Default**: σ is standard deviation in log space
- `sd_natural=True` (models 26–27): σ converted from natural-space std
- `sigma_fwhm=True` (models 28–30): σ derived from FWHM in natural space

Width scaling (when `separate_sds=True`):
```
σ_wide = sd_wide_scale × σ_narrow
```

Amplitude relationship (when `separate_amplitudes=True`):
```
amplitude_wide = amplitude_alpha + amplitude_beta × amplitude_narrow
```
- If `amplitude_alpha` is fixed at 0: pure scaling, no intercept

**identity_below_range is always on for LinearScalingModel.**

### Fitted width scaling (models 15–24)

| Model | δ | σ scaling | Amplitude | Notes |
|-------|---|-----------|-----------|-------|
| 15 | 2.0 (fixed) | shared scale | single | σ_wide = scale × σ_narrow, scale shared |
| 16 | 2.0 (fixed) | single σ | α+β shared, rescale_baseline | Separate amplitudes with shared intercept/slope |
| 17 | 2.0 (fixed) | shared scale | α+β shared, rescale_baseline | Like 16 + separate widths |
| 18 | shared | shared scale | single | Like 15 but δ fitted (shared) |
| 19 | shared | single σ | α+β shared, rescale_baseline | Like 16 but δ fitted |
| 20 | shared | shared scale | α+β shared, rescale_baseline | Like 17 but δ fitted |
| 21 | 2.0 (fixed) | single σ | α+β shared | Like 16, no baseline rescaling |
| 22 | 2.0 (fixed) | shared scale | α+β shared | Like 17, no baseline rescaling |
| 23 | shared | single σ | α+β shared | Like 21 but δ fitted |
| 24 | shared | shared scale | α+β shared | Like 22 but δ fitted |

### Fixed width scaling (models 25–33)

These models fix the `sd_wide_scale` to a specific value rather than fitting it.

| Model | δ | σ scaling | Amplitude | σ parameterisation | Notes |
|-------|---|-----------|-----------|---------------------|-------|
| 25 | 2.0 (fixed) | √2 (fixed) | single | log space | Like 15, scale fixed at √2 |
| 26 | 2.0 (fixed) | shared | single | natural space | Like 15, σ in natural space |
| 27 | 2.0 (fixed) | 2.0 (fixed) | single | natural space | Like 26, scale fixed at 2 |
| 28 | 2.0 (fixed) | shared | single | FWHM natural | Like 15, σ as FWHM |
| 29 | 2.0 (fixed) | 2.0 (fixed) | single | FWHM natural | Like 28, scale fixed at 2 |
| 30 | 2.0 (fixed) | free per voxel | single | FWHM natural | Like 14 (separate per-voxel widths) but FWHM |
| 31 | 2.0 (fixed) | 1.2878 (fixed) | single | log space | **Primary production model.** Scale from mean of model 15 across subjects |
| 32 | 2.0 (fixed) | 1.2878 (fixed) | α=0 fixed, β free per voxel | log space | Like 31 + free per-voxel amplitude scaling (no intercept) |
| 33 | 2.0 (fixed) | 1.2878 (fixed) | α=0 fixed, β shared | log space | Like 32 but amplitude scaling shared |

### Amplitude-only models (models 34–36)

These models hold tuning shape and shift fixed, only fitting amplitude changes between conditions.

| Model | δ | σ scaling | Amplitude | Notes |
|-------|---|-----------|-----------|-------|
| 34 | 2.0 (fixed) | single σ (free per voxel) | α=0 fixed, β free per voxel | Per-voxel amplitude scaling; no intercept |
| 35 | 2.0 (fixed) | single σ (free per voxel) | α shared, β shared | Shared affine amplitude mapping (intercept + slope) |
| 36 | 2.0 (fixed) | single σ (free per voxel) | α=0 fixed, β shared | Like 35 but only a shared scaling factor; no intercept |

---

## Fixed-slope series (models 115–145)

Models 100+ are copies of model 31 where `sd_wide_scale` is fixed at `model_label / 100.0` instead of 1.2878. Used to sweep the width-scaling parameter and identify the best-fitting slope.

| Range | sd_wide_scale | Example use |
|-------|---------------|-------------|
| 115–145 | 1.15–1.45 | Grid over plausible width-scaling values |

---

## Parameter glossary

| Parameter | Description | Scope |
|-----------|-------------|-------|
| `mu_narrow` | Preferred numerosity in narrow-range condition | per voxel |
| `delta_wide` (δ) | Multiplicative shift ratio between conditions | fixed / shared / per voxel |
| `lower_bound_range` (lb) | Reference point for shift (boundary, typically 10) | fixed / per voxel |
| `alpha` (γ) | Box-Cox power (AlphaDeltaModel only; ~0 ≈ log) | shared |
| `sd` / `sd_narrow` | Tuning width (in log or natural space depending on model) | per voxel |
| `sd_wide_scale` | Ratio of wide to narrow tuning width | fixed / shared |
| `amplitude` / `amplitude_narrow` | Response gain in narrow condition | per voxel |
| `amplitude_alpha` | Intercept of wide-vs-narrow amplitude relationship | fixed(0) / shared |
| `amplitude_beta` | Slope of wide-vs-narrow amplitude relationship | shared / per voxel |
| `baseline` | DC offset (same for both conditions unless rescaled) | per voxel |
| `baseline_ratio` | Fractional baseline adjustment (rescale_baseline models) | shared |

---

## Parameter count (AIC/BIC)

Counts per-voxel free parameters + shared parameters (as 1 each) + noise σ.

| Model | k | Free parameters |
|-------|---|-----------------|
| -1 (null) | 2 | mean + σ |
| 0 | 5 | μ, sd, amp, baseline + σ |
| 1 | 5 | same |
| 2 | 6 | + shared δ |
| 3 | 7 | μ, δ, lb, sd, amp, baseline + σ |
| 4 | 5 | same as 1 |
| 5 | 6 | + shared δ |
| 14 | 6 | μ, sd_n, sd_w, amp, baseline + σ |
| 15 | 6 | μ, baseline, sd_n, amp + shared scale + σ |
| 31 | 5 | μ, baseline, sd_n, amp + σ |
| 32 | 6 | μ, baseline, sd_n, amp_n, amp_β + σ |
| 33 | 6 | μ, baseline, sd_n, amp_n + shared amp_β + σ |
| 34 | 7 | μ, lb, baseline, sd, amp_n, amp_β + σ |
| 35 | 8 | μ, lb, baseline, sd, amp_n + shared amp_α, amp_β + σ |
| 36 | 7 | μ, lb, baseline, sd, amp_n + shared amp_β + σ |
