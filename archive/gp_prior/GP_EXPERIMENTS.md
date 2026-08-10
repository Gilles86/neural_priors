# GP-Prior Experiment Registry

Each row is an experiment tag — the `--tag NAME` value passed to
`fit_gp_prior.py`. Outputs land under

    derivatives/encoding_models/gp_prior_roi-{ROI}[.smoothed]/exp-{tag}/

with a per-subject `sub-{NN}_desc-manifest.json` recording git SHAs,
CLI args, and timestamps. Always **add a row here before submitting a
new tag** so the registry stays the single source of truth.

---

## Conventions

- Tag is short kebab/snake (e.g. `indep_l`, `mu_only`, `joint_4`).
- Within a tag, smoothing × stim_range × ROI vary; everything else is
  fixed by the tag definition below.
- "Status" is updated when SLURM jobs complete. Date is submission date.
- `compared against` lists adjacent tags that share the rest of the
  config so contrasts are clean.

---

## Active experiments

| Tag        | What it tests                          | Priors on               | Stage-2 mode            | Voxel selection      | Decoder ω | Status        | Compared against    | Submitted | Notes |
|------------|----------------------------------------|-------------------------|-------------------------|----------------------|-----------|---------------|---------------------|-----------|-------|
| `indep_l`  | Baseline (current production recipe)   | mu, sd, amplitude, baseline | Per-prior MLE, independent l | per-fold p_signal≥0.5 | plain + distance | **complete** 36/31 (unsm/sm) | `shared_l`, `mu_only`, `joint_4` | 2026-05-18 | bayes consistently 0.02–0.04 r *worse* than classical at decoding. |
| `shared_l` | Tied lengthscales across all 4 priors   | mu, sd, amplitude, baseline | Joint MLE, one shared l, per-prior v, n | per-fold p_signal≥0.5 | plain + distance | **complete** 36/34 | `indep_l`            | 2026-05-18 | Numerically nearly identical to `indep_l` — sharing doesn't rescue the prior. |
| `mu_only`  | Daghlian's actual paper recipe          | mu                      | Per-prior MLE           | per-fold p_signal≥0.5 | plain + distance | running, 30/22 landed | `indep_l`            | 2026-05-18 | Modest Δr ≈ +0.011 on unsm/plain (21/30 positive). Confirms over-applying priors was the main problem. Tied on distance ω. |
| `joint_4`  | Type-II MAP, all 4 priors               | mu, sd, amplitude, baseline | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance | running, 4/4 landed   | `indep_l`            | 2026-05-18 | Preliminary: catastrophic on smoothed (Δr = −0.127, N=4). Joint-MAP gives 4 priors too much freedom to over-regularize jointly. |
| `joint_mu` | Type-II MAP, mu prior only              | mu                      | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance | running, 23/17 landed | `mu_only`, `joint_4` | 2026-05-18 | **Winner so far.** unsm/plain Δr=+0.025, 17/23 positive. Daghlian-faithful priors + joint-MAP stage-2 fix. |
| `joint_mu_noW` | joint_mu × no WᵀW in decoder Ω        | mu                      | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance, **σ²·WᵀW stripped** | running, 10/9 landed | `joint_mu`           | 2026-05-18 | Δr goes *negative* without WᵀW (−0.009 unsm/plain; −0.034 smoothed/plain). σ² is fit honestly; WᵀW was doing real noise modeling, not cancelling the prior. |
| `joint_tuning` | Priors on tuning shape (mu + sd)      | mu, sd                  | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance | running, 0/0 landed   | `joint_mu`, `joint_4` | 2026-05-18 | Tests whether tuning *shape* (peak + width) benefits from smoothing. amplitude+baseline left un-priored — they're voxel-specific scale/offset with no cortical-smoothness rationale. |
| `m15_joint_mu` | Model 15 (LinearScalingModel: narrow+wide joint fit, shared σ scaling) with prior on mu_narrow + joint MAP | mu_narrow            | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance | submitted             | `joint_mu`           | 2026-05-19 | Tests whether the richer model (more data per voxel: narrow + wide jointly; shared σ scaling) lets the prior pay off. Uses `fit_model.fit_model()` verbatim for classical step. Debug fold (sub-02): lengthscale converged to ~11 mm (vs ~22 mm in LogGaussianPRF cells) — joint constraint gives mu more structure to identify. bayes cvR² > classical cvR² in that one fold. |

---

## Findings (running summary)

Last refreshed: 2026-05-18 (interim — `joint_*` cells still landing).

1. **`joint_mu` is the working recipe.** Single-parameter prior on the tuning peak (mu only) plus type-II MAP for hyperparameters. At N=23 unsm subjects, Δr(bayes − classical) = **+0.025 ± 0.012 in plain ω, 17/23 positive** — first reliable bayes-helps result in this project.

2. **Over-applying priors hurts, badly.** All four-parameter cells lose (`indep_l`, `shared_l`, preliminary `joint_4` is catastrophic). Amplitude and baseline are voxel-specific scale/offset — they have no cortical-smoothness rationale and forcing one degrades the encoding model.

3. **Stage-2 MLE on noisy point estimates over-smooths.** Fitted lengthscales reach ~22–26 mm (≈ NPC diameter) in the 2-step cells. The cure (joint-MAP, which adds the `-½ log\|K(ψ)\|` term to the loss) fixes it: `joint_mu` lengthscales are reasonable, and the prior helps.

4. **WᵀW is not cancelling the prior — our hypothesis was wrong.** `joint_mu_noW` (which strips σ²·WᵀW from decoder Ω) is *worse* than `joint_mu`, not better. σ² is fit honestly by the residual fitter, and the σ²·WᵀW term captures real spatial-noise structure aligned with tuning similarity.

5. **Decoder ω is doing huge work regardless of the prior.** Distance-modulated ω adds +0.02 to +0.10 r in every cell. The strongest decoding result is **`joint_mu` / smoothed / distance: bayes r = +0.210, N=17** (just edges out classical at +0.201 in that cell).

6. **Baseline numbers** (paper reports r ≈ 0.082 narrow / 0.136 wide for V1): our classical/distance recipe is already 2–3× better. The GP prior brings a modest additional boost on top, only when applied correctly.

### What's still uncertain
- `joint_tuning` (mu + sd) just submitted; will tell us if sd-smoothing helps or hurts.
- `joint_4` catastrophic result is N=4; needs full N before declaring it dead.
- Wide stimulus range still unrun across the new tags. Once narrow is in for all winners, submit wide-only for the top 2–3 tags.

### Standing decisions
- Voxel selection: per-fold NPC R²-mixture, posterior P(signal | R²) ≥ 0.5.
- Within-fold cross-validation: leave-one-(session, run)-out, all fitting on train, decode on test.
- Decoder ω: always evaluate both `plain` and `distance` head-to-head per fold.

---

## Legacy (untagged) runs — pre-2026-05-18

Before the `--tag` convention was added, outputs landed directly under
`gp_prior_roi-{ROI}[.smoothed]/sub-NN/func/` (no `exp-` level). These
are kept on disk for archaeology but should not be mixed with new
experiments in analyses. Run config:

- Voxel selection: **whole-brain FDR α=0.05** (subject-level threshold from `model15.cv` whole-brain cvR²; falls back to top-100 voxels when threshold > all NPC R²)
- Priors on: mu, sd, amplitude, baseline (independent lengthscales)
- Stage 2: per-prior MLE
- braincoder SHA: ranged across the buggy `clipnorm` era; some folds hit NaN-gradient blow-ups (fixed by [`3392680`](https://github.com/Gilles86/braincoder/commit/3392680))

---

## How to add a new experiment

1. Add a row to **Active experiments** (or **Planned** if not yet submitted).
2. Pick a tag that's distinct from every existing row.
3. Run with `--tag <your_tag>` and any flags that define the variant.
4. After completion, fill in Status with the SLURM job ID + landed subject count.
5. Once analyzed, append a one-line key finding to Notes.
