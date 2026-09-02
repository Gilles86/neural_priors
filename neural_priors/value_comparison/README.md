# `value_comparison` — numerosity tuning measured with the value-tuning analysis

Answers one question: **how much stronger is numerosity tuning in this dataset
than value tuning is in `~/git/value_prf` (Le Bouc, Ruff et al.), when both are
measured with exactly the same analysis?**

Every estimator is *imported* from `value_prf` rather than reimplemented
(`fit_grid_ols`, `cv_by_group`, `effective_df`, `build_model('gauss', K=11,
sd=0.075)`), so no difference between the two datasets can come from the code.
The stimulus axis is the only thing that changes: numerosity 10-40 plays the
role of value 0-900, and the basis width and PRF sigma grid are defined as the
same *fractions of the stimulus range*.

## Pipeline

| Step | Script | Where it runs |
|---|---|---|
| 1. Dump ROI-masked single-trial betas to small `.npz` | `extract_roi_trials.py` (+ `slurm_jobs/extract_roi_trials.sh`) | **cluster** — the 4D GLMsingle NIfTIs are ~300 MB/subject and live only on `/shares/zne.uzh/gdehol/ds-neuralpriors` |
| 2. cvR², permutation null, normalised-axis records, example voxels | `run_analysis.py` | local |
| 3. The same statistics on the value data | `value_side.py` | local |
| 4. Permutation null for the held-out centre−flank statistic | `heldout_null.py` | local |
| 5. Does the peak-aligned effect survive interior-μ voxels only? | `interior_mu.py` | local |
| 6. Figure + tidy TSV | `plot_numerosity_vs_value.py` | local |

`matched_prf.py` holds the shared machinery (loading, the numerosity stimulus
grids, the normalised axis, the validated fast ridge).

## Running it

```bash
# 1. cluster (one job per subject; 11 and 23 have a single session and are excluded)
rsync -a neural_priors/value_comparison/ sciencecluster:git/neural_priors/neural_priors/value_comparison/
ssh sciencecluster 'cd ~/git/neural_priors && sbatch --array=1-10,12-22,24-41 \
    neural_priors/value_comparison/slurm_jobs/extract_roi_trials.sh'
rsync -a sciencecluster:/shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/value_comparison/ \
    /data/ds-neuralpriors/derivatives/value_comparison/

# 2-6. local (needs braincoder AND value_prf importable)
export KERAS_BACKEND=tensorflow
export PYTHONPATH=~/git/neural_priors:~/git/value_prf
P=~/mambaforge/envs/braincoder/bin/python
$P -m neural_priors.value_comparison.run_analysis
$P -m neural_priors.value_comparison.value_side
$P -m neural_priors.value_comparison.heldout_null --n-perm 5
$P -m neural_priors.value_comparison.interior_mu
$P -m neural_priors.value_comparison.plot_numerosity_vs_value
```

Outputs land in `~/git/value_prf/notes/{data,figures,analyses}` (that is where
the comparison is being written up).

## The one shortcut, and its validation

braincoder's `WeightFitter(alpha)` solves a 12x12 ridge system in TensorFlow;
the TF round-trip, not the algebra, dominates, so the permutation nulls would
take hours. `matched_prf.cv_wprf_fast` does the identical closed-form solve in
numpy (~300x faster). `matched_prf.check_equivalence` runs the two against each
other on real data and is asserted at the top of every `run_analysis` run —
observed max |ΔcvR²| = 2.3e-6 on both datasets (float32 vs float64 rounding),
and `value_side.py` reproduces value_prf's stored permutation null exactly.

## Traps this code deliberately respects

* **Circularity** — μ, σ, the amplitude/baseline *and* the "well fitting" voxel
  selection all come from a different half of the runs than the trials plotted.
  The circular version is always produced alongside, because it makes a clean
  Gaussian out of pure noise and is what makes the held-out panel readable.
* **Amplitude sign** — unconstrained OLS puts a *trough* at μ for a large
  fraction of voxels; `positive_only=True` throughout.
* **Edge/coverage composition** — only small-σ voxels reach large |x|, so far
  bins are a biased subset; bins where <25% of the selected voxels contribute
  are hidden, and out-of-range points are *dropped*, never clipped into edge bins.
* **Edge μ** — a voxel whose μ sits at the end of the stimulus range turns
  "centre minus flanks" into "one end of the range minus the other", which a
  monotonic response produces with no tuning. `interior_mu.py` exists for that.
* **Trial count** — numerosity has 480 trials to value's 128. Every result is
  also reported for a 128-trial (8 runs x 16 trials) numerosity subsample.
