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
| `indep_l`  | Baseline (current production recipe)   | mu, sd, amplitude, baseline | Per-prior MLE, independent l | per-fold p_signal≥0.5 | plain + distance | running 3091131/3091133 | `shared_l`, `mu_only`, `joint_4` | 2026-05-18 | 31–36 of 39 subj landed so far. bayes consistently 0.02–0.04 r *worse* than classical at decoding. |
| `shared_l` | Tied lengthscales across all 4 priors   | mu, sd, amplitude, baseline | Joint MLE, one shared l, per-prior v, n | per-fold p_signal≥0.5 | plain + distance | running 3091132/3091134 | `indep_l`            | 2026-05-18 | Numerically nearly identical to `indep_l` — sharing doesn't rescue the prior. |
| `mu_only`  | Daghlian's actual paper recipe          | mu                      | Per-prior MLE           | per-fold p_signal≥0.5 | plain + distance | running 3120687/3120688 | `indep_l`            | 2026-05-18 | Paper applies the GP to *one* tuning parameter only; broad/no prior on sd/amplitude/baseline. Tests whether over-applying priors is what's hurting decoding. |
| `joint_4`  | Type-II MAP, all 4 priors               | mu, sd, amplitude, baseline | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance | submitted             | `indep_l`            | 2026-05-18 | Hyperparameters co-optimized with parameters in fit_map. `-½ log\|K(ψ)\|` term in prior log-prob auto-regularizes ψ (long l → big \|K\| → loss penalty). Sidesteps stage-2's over-smoothing bias. |
| `joint_mu` | Type-II MAP, mu prior only              | mu                      | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance | submitted             | `mu_only`, `joint_4` | 2026-05-18 | Combines Daghlian's single-parameter recipe with the joint-MAP fix for stage-2 bias. The cleanest test of "is the prior good in principle, when applied correctly?" |
| `joint_mu_noW` | joint_mu × no WᵀW in decoder Ω        | mu                      | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance, **σ²·WᵀW stripped** | submitted | `joint_mu`           | 2026-05-18 | Tests whether the σ²·WᵀW term in decoder Ω absorbs/cancels the GP prior's spatial-smoothness benefit. Compare bayes Δr (vs classical) in this cell to `joint_mu`. If gain is much larger here, WᵀW was creating cancellation. |
| `joint_tuning` | Priors on tuning shape (mu + sd)      | mu, sd                  | Skipped — hyperparams in Stage 3 trainable set | per-fold p_signal≥0.5 | plain + distance | submitted             | `joint_mu`, `joint_4` | 2026-05-18 | Tests whether the tuning *shape* (peak + width) benefits from smoothing while leaving the easy/linear amplitude+baseline parameters un-priored. Intuition: mu and sd describe topographic structure; amplitude+baseline are voxel-specific scale/offset that have no reason to be cortically smooth. |

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
