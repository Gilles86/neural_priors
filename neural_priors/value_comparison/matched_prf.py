"""Numerosity tuning measured with EXACTLY the analysis used for value tuning
in ~/git/value_prf, so the two effect sizes can be put on one axis.

The value analysis (value_prf/visualize/plot_prf_normalized.py,
value_prf/encoding_model/prf_cv.py, value_prf/visualize/plot_example_voxels.py)
concluded that single-voxel value tuning is at chance and that the population
"peak-aligned" effect is +0.03 to +0.08 z.  This module reproduces every step of
that on the numerosity dataset:

  * the same WEIGHTED PRF -- 11 fixed Gaussian basis tiling the NORMALISED
    stimulus range with sd = 0.075 of that range, plus an intercept, per-voxel
    weights by ridge alpha = 0.1 (braincoder WeightFitter) -- imported from
    value_prf so the code path is literally the same;
  * the same single-peak grid PRF (`fit_grid_ols`, positive_only=True);
  * the same normalised-axis test: mu/sd/amplitude AND voxel selection from one
    half of the runs, held-out trials plotted at x = (stim - mu)/sd, z-scored
    per voxel within the held-out half, scored as centre (|x|<0.5) minus flanks
    (1.5<|x|<3);
  * the same circular control (fit and show on the same trials);
  * the same leave-one-run-out cvR^2 and label-permutation null.

Stimulus mapping.  Numerosity 10-40 plays the role value 0-900 played: the
basis and the sd grid are defined as FRACTIONS of the stimulus range, so
sd = 0.075 * 900 = 67.5 value units becomes 0.075 * 30 = 2.25 numerosity units,
and the whole grid scales the same way.  Nothing else changes.

Assumptions that had to be made about the numerosity data (all flagged in
notes/analyses/numerosity_comparison.md):
  * the two range conditions (narrow 10-25, wide 10-40) are POOLED and a single
    stimulus axis 10-40 is used, because the value analysis has no analogous
    condition factor.  A range-shifting voxel is therefore slightly blurred by
    this analysis -- i.e. it is, if anything, conservative for numerosity.
  * permutation null shuffles numerosity WITHIN range condition, so that the
    narrow/wide block structure (which is confounded with run) survives the
    null and only the trial-level stimulus-response link is destroyed.
"""
import os

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

from pathlib import Path

import numpy as np
import pandas as pd

# The value analysis itself -- imported, not reimplemented, so that any
# difference between the two datasets cannot come from the code.  value_prf is
# a sibling repo that is not installed into this env's site-packages, so make
# it importable if it is not already (override with $VALUE_PRF_DIR).
import sys
try:
    import value_prf  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get(
        'VALUE_PRF_DIR', os.path.expanduser('~/git/value_prf')))

from value_prf.encoding_model.prf_cv import fit_grid_ols, _design
from value_prf.visualize.plot_example_voxels import (
    cv_by_group, effective_df, K_POP, SD_POP, ALPHA, GRID01)

# ── stimulus space ───────────────────────────────────────────────────────────
# Every dataset gets the SAME model, expressed as fractions of its own stimulus
# range: 11 fixed Gaussian basis with sd = 0.075 of the range, and a single-peak
# PRF whose sigma grid is value's [67.5, 100, 150, 225, 300, 450] on 0-900
# rewritten as fractions.  Only the range -- and whether the axis is log --
# changes between datasets.
N_MU = 61                              # same grid resolution as the value fit
SD_FRACTIONS = np.array([0.075, 1 / 9., 1 / 6., 0.25, 1 / 3., 0.5])


class Space:
    """A stimulus axis: where it starts and ends, and whether it is logarithmic.

    ``t()`` maps a raw stimulus onto the axis the model actually works on
    (identity, or natural log).  Fitting, the mu/sigma grids and the normalised
    axis x = (t(stim) - mu)/sigma all live in that transformed space.
    """

    def __init__(self, lo, hi, log=False, name='', unit=''):
        self.lo, self.hi, self.log, self.name, self.unit = lo, hi, log, name, unit
        self.t_lo = float(np.log(lo)) if log else float(lo)
        self.t_hi = float(np.log(hi)) if log else float(hi)
        self.range = self.t_hi - self.t_lo
        self.mu_grid = np.linspace(self.t_lo, self.t_hi, N_MU)
        self.sd_grid = SD_FRACTIONS * self.range

    def t(self, v):
        v = np.asarray(v, float)
        return np.log(v) if self.log else v

    def to01(self, v_t):
        return ((np.asarray(v_t, float) - self.t_lo) / self.range).astype(np.float32)

    def grid01_to_stim(self, g01):
        t = self.t_lo + np.asarray(g01, float) * self.range
        return np.exp(t) if self.log else t

    def __repr__(self):
        return (f'Space({self.name!r}, {self.lo:g}-{self.hi:g}'
                f'{", log" if self.log else ""})')


# The narrow range condition on its own.  Pooling narrow (10-25) and wide
# (10-40) blurs the very range-shift effect neural_priors is about: a voxel's
# preferred numerosity MOVES between conditions, so one mu per voxel is the
# wrong model for the pooled data.
SPACES = {
    'neural_priors_narrow': None,       # filled in below
    'neural_priors_narrow_log': None,
    # neural_priors: numerosity 10-40, only ~2 octaves, so linear is the
    # primary axis (it is also what the value analysis does).  The log variant
    # is kept so the two numerosity datasets can be treated identically.
    'neural_priors': Space(10., 40., log=False, name='Numerosity', unit=''),
    'neural_priors_log': Space(10., 40., log=True, name='Numerosity'),
    # tms_risk: n1 of the risky-choice pair spans 7-86, i.e. 3.6 octaves and
    # log-spaced by design, so a linear axis would put most trials in one
    # corner.  Log is the honest choice here (and the space that repo's own
    # nPRF models use).
    'tmsrisk': Space(7., 86., log=True, name='Numerosity'),
    'tmsrisk_lin': Space(7., 86., log=False, name='Numerosity'),
    # value, for the scripts that need the value axis in the same object
    'value': Space(0., 900., log=False, name='Value rating'),
}

SPACES['neural_priors_narrow'] = Space(10., 25., log=False, name='Numerosity')
# neural_priors' own production model (LinearScalingModel, model 31) is
# LOG-NORMAL over numerosity.  A linear axis here was a choice to keep the
# estimator identical to the value pipeline, not a claim about numerosity
# coding; this space tests what that choice costs.
SPACES['neural_priors_narrow_log'] = Space(10., 25., log=True,
                                           name='Numerosity')

DEFAULT_SPACE = SPACES['neural_priors']

# module-level aliases kept so the original neural_priors call sites still work
STIM_LO, STIM_HI = DEFAULT_SPACE.lo, DEFAULT_SPACE.hi
STIM_RANGE = DEFAULT_SPACE.range
MU_GRID = DEFAULT_SPACE.mu_grid
SD_GRID = DEFAULT_SPACE.sd_grid

# ── normalised axis (identical constants to plot_prf_normalized.py) ──────────
X_LIM = 4.0
N_X_BINS = 32
X_EDGES = np.linspace(-X_LIM, X_LIM, N_X_BINS + 1)
X_CENTERS = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])

# Overridable so the cluster does not need its working tree edited (which
# breaks `git pull` and the sync-by-commit rule).  Set NP_VALUE_COMPARISON_DIR
# in the SLURM script to point at the share.
DATA_DIR = Path(os.environ.get(
    'NP_VALUE_COMPARISON_DIR',
    '/data/ds-neuralpriors/derivatives/value_comparison'))
ROI_LABEL = {'NPCr': 'NPC (R)', 'NPCl': 'NPC (L)',
             'wholebrain': 'Whole brain (outside NPC)'}


# ── data ─────────────────────────────────────────────────────────────────────

def load_subject(subject, roi='NPCr', data_dir=DATA_DIR, smoothed=False,
                 narrow_only=False):
    """(betas, n, run_id, range_wide) for one subject.

    run_id runs 1..16 over the two sessions, so leave-one-run-out and the
    odd/even split-half are both defined on it.  The odd/even split is balanced
    for range condition because the narrow/wide block order is reversed in
    session 2.
    """
    tag = '.smoothed' if smoothed else ''
    fn = Path(data_dir) / f'sub-{subject}_trials{tag}.npz'
    d = np.load(fn)
    B = np.asarray(d[f'betas_{roi}'], dtype=np.float64)
    n = np.asarray(d['n'], dtype=np.float64)
    run_id = (d['session'].astype(int) - 1) * 8 + d['run'].astype(int)
    rw = d['range_wide'].astype(bool)
    if narrow_only:
        # 240 of the 480 trials.  Halving the data costs effect size, so a
        # narrow-log result is only interpretable against narrow-LINEAR, never
        # against the pooled run.
        keep = ~rw
        B, n, run_id, rw = B[keep], n[keep], run_id[keep], rw[keep]
    return B, n, run_id, rw


TMSRISK_DIR = Path(os.environ.get(
    'TMSRISK_VALUE_COMPARISON_DIR',
    '/data/ds-tmsrisk/derivatives/value_comparison'))


def load_subject_tmsrisk(subject, roi='NPCr', data_dir=TMSRISK_DIR, session=1,
                         smoothed=False):
    """(betas, n1, run_id, dummy) for one tms_risk subject, session 1 only.

    Session 1 is the baseline session (no cTBS yet), 6 runs x 20 trials = 120
    trials -- essentially the value dataset's 128, which is the whole point of
    including this dataset.  The returned 4th element is a dummy 'condition'
    array so the permutation null can share one code path with neural_priors
    (tms_risk has no range manipulation, so the permutation is unrestricted).
    """
    tag = '.smoothed' if smoothed else ''
    fn = Path(data_dir) / f'sub-{int(subject):02d}_ses-{session}_trials{tag}.npz'
    d = np.load(fn)
    B = np.asarray(d[f'betas_{roi}'], dtype=np.float64)
    return (B, np.asarray(d['n'], float), d['run'].astype(int),
            np.zeros(len(d['n']), bool))


def subsample_matched(B, n, run_id, range_wide, n_runs=8, n_trials=16, seed=0):
    """Cut the numerosity data down to the value dataset's size: 8 runs x 16
    trials = 128 trials, so the two datasets can be compared at equal power.

    Runs are taken alternately (1, 3, 5, ...) so both range conditions and both
    sessions stay represented; trials within a run are a random subset.
    """
    rng = np.random.default_rng(seed)
    runs = np.unique(run_id)[::2][:n_runs]
    keep = []
    for r in runs:
        idx = np.flatnonzero(run_id == r)
        keep.append(rng.choice(idx, min(n_trials, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return B[keep], n[keep], run_id[keep], range_wide[keep]


# ── weighted PRF on the numerosity axis ──────────────────────────────────────

def to01(v, space=None):
    space = space or DEFAULT_SPACE
    return space.to01(space.t(v))


def _wprf_frames(B, v, space=None):
    return pd.DataFrame(B), pd.DataFrame({'x': to01(v, space)})


def fit_wprf(B_fit, v_fit, B_show, v_show, space=None):
    """mu, scale, prediction for the shown trials, and training R^2.

    Exact port of `_fit_wprf` in value_prf/visualize/plot_prf_normalized.py with
    value/900 replaced by (n-10)/30: mu = argmax of the weighted sum inside the
    stimulus range the fit covered, scale = FWHM / 2.355.
    """
    from braincoder.optimize import WeightFitter
    from braincoder.utils import get_rsq
    from value_prf.decoding.decode import build_model

    space = space or DEFAULT_SPACE
    fit_df, par = _wprf_frames(B_fit, v_fit, space)
    model = build_model('gauss', K=K_POP, sd=SD_POP)
    wf = WeightFitter(model, model.parameters, fit_df, par)
    model.weights = wf.fit(alpha=ALPHA)

    grid01 = np.linspace(0, 1, 200).astype(np.float32)
    curves = np.asarray(model.predict(paradigm=pd.DataFrame({'x': grid01})))
    gstim = space.t_lo + grid01 * space.range          # transformed axis
    vt_fit = space.t(v_fit)
    sup = (gstim >= vt_fit.min()) & (gstim <= vt_fit.max())
    cs = curves[sup]
    mu = gstim[sup][np.nanargmax(cs, axis=0)]
    rng_c = cs.max(0) - cs.min(0)
    half = cs.min(0) + 0.5 * rng_c
    fwhm = ((cs > half[None, :]).sum(0) / sup.sum()
            * (gstim[sup].max() - gstim[sup].min()))
    scale = np.maximum(fwhm / 2.355, 1e-3)

    pred_show = np.asarray(model.predict(
        paradigm=pd.DataFrame({'x': to01(v_show, space)})))
    pred_fit = np.asarray(model.predict(paradigm=par))
    r2 = np.asarray(get_rsq(fit_df, pd.DataFrame(pred_fit)))
    return mu, scale, pred_show, r2


def cv_wprf(B, v, groups, space=None):
    """Leave-one-group-out cvR^2 for the weighted PRF (value_prf's cv_by_group,
    on the normalised stimulus axis)."""
    betas, par = _wprf_frames(B, v, space)
    return cv_by_group(betas, par, np.asarray(groups))


def cv_wprf_curves(B, v, groups):
    betas, par = _wprf_frames(B, v)
    return cv_by_group(betas, par, np.asarray(groups), return_fold_curves=True)


def permutation_null(B, v, range_wide, groups, n_perm=20, seed=0):
    """Leave-one-run-out cvR^2 with the numerosity->trial mapping shuffled.

    Shuffling happens WITHIN range condition, so the narrow/wide block
    structure (confounded with run) is preserved and only the trial-level
    stimulus-response relation is destroyed.  This is the numerosity analogue
    of `permutation_null` in value_prf/visualize/plot_example_voxels.py, which
    shuffles value across items (preserving the repeat structure).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        vperm = np.asarray(v, float).copy()
        for cond in (False, True):
            sel = range_wide == cond
            if sel.sum():
                vperm[sel] = rng.permutation(vperm[sel])
        out.append(cv_wprf(B, vperm, groups))
    return np.stack(out)


# ── single-peak PRF cross-validation ─────────────────────────────────────────

def cv_prf(B, v, groups, positive_only=True, space=None):
    space = space or DEFAULT_SPACE
    vt = space.t(v)
    pred = np.full_like(B, np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        pred[te] = fit_grid_ols(B[tr], vt[tr], B[te], vt[te], space.mu_grid,
                                space.sd_grid, positive_only=positive_only)[0]
    ss_res = ((B - pred) ** 2).sum(0)
    ss_tot = ((B - B.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


# ── normalised axis ──────────────────────────────────────────────────────────

def normalised(B_fit, v_fit, B_show, v_show, top_pct=2.0, model='prf',
               space=None):
    """Held-out trials on x = (stim - mu)/sd, plus the model there.

    Line-for-line the same as `normalised` in
    value_prf/visualize/plot_prf_normalized.py, with the value grids replaced by
    the numerosity grids.  positive-amplitude-only, out-of-range points dropped
    rather than clipped, per-voxel z-scoring inside the SHOWN half.
    """
    space = space or DEFAULT_SPACE
    vt_fit, vt_show = space.t(v_fit), space.t(v_show)
    if model == 'wprf':
        mu, sd, pred_all, r2 = fit_wprf(B_fit, v_fit, B_show, v_show, space)
        frac_neg = 0.0
        keep = r2 >= np.nanpercentile(r2, 100 - top_pct)
        mu, sd = mu[keep], sd[keep]
        pred, S = pred_all[:, keep], B_show[:, keep]
    else:
        _, mu, sd, beta, r2 = fit_grid_ols(B_fit, vt_fit, B_fit[:1], vt_fit[:1],
                                           space.mu_grid, space.sd_grid,
                                           positive_only=True)
        keep = r2 >= np.nanpercentile(r2, 100 - top_pct)
        frac_neg = float((beta[0][keep] < 0).mean())
        mu, sd, beta = mu[keep], sd[keep], beta[:, keep]
        S = B_show[:, keep]
        pred = np.stack([_design(vt_show, m, s) @ beta[:, i]
                         for i, (m, s) in enumerate(zip(mu, sd))], axis=1)

    m0, s0 = S.mean(0), S.std(0, ddof=1) + 1e-12
    Z, P = (S - m0) / s0, (pred - m0) / s0

    x = (vt_show[:, None] - mu[None, :]) / sd[None, :]
    xr, zr, pr = x.ravel(), Z.ravel(), P.ravel()
    ok = (xr >= X_EDGES[0]) & (xr <= X_EDGES[-1])      # drop, never clip
    xr, zr, pr = xr[ok], zr[ok], pr[ok]
    vox_id = np.tile(np.arange(len(mu)), (len(vt_show), 1)).ravel()[ok]

    bidx = np.digitize(xr, X_EDGES[1:-1])
    binned = np.full(N_X_BINS, np.nan)
    bpred = np.full(N_X_BINS, np.nan)
    counts = np.zeros(N_X_BINS)
    n_vox_bin = np.zeros(N_X_BINS)
    for b in range(N_X_BINS):
        sel = bidx == b
        if sel.any():
            binned[b], bpred[b] = zr[sel].mean(), pr[sel].mean()
            counts[b] = sel.sum()
            n_vox_bin[b] = len(np.unique(vox_id[sel]))
    return dict(pts=np.column_stack([xr, zr]), binned=binned, bpred=bpred,
                counts=counts, n_vox_bin=n_vox_bin, n_vox=len(mu),
                frac_neg_amplitude=frac_neg, mu=np.asarray(mu, float),
                sd=np.asarray(sd, float), trial_values=np.asarray(vt_fit, float),
                median_sd=float(np.median(sd)),
                median_r2=float(np.median(r2[keep])))


def combine(halves):
    """Pool the two split-half directions (value_prf's `_combine`)."""
    cnt = sum(h['counts'] for h in halves)
    safe = np.where(cnt, cnt, np.nan)
    with np.errstate(all='ignore'):
        return dict(
            pts=np.vstack([h['pts'] for h in halves]),
            binned=np.nansum([h['binned'] * h['counts'] for h in halves], 0) / safe,
            bpred=np.nansum([h['bpred'] * h['counts'] for h in halves], 0) / safe,
            counts=cnt,
            n_vox_bin=sum(h['n_vox_bin'] for h in halves),
            n_vox=sum(h['n_vox'] for h in halves),
            mu=np.concatenate([h['mu'] for h in halves]),
            sd=np.concatenate([h['sd'] for h in halves]),
            trial_values=np.concatenate([h['trial_values'] for h in halves]),
            median_sd=float(np.mean([h['median_sd'] for h in halves])),
            median_r2=float(np.mean([h['median_r2'] for h in halves])),
            frac_neg_amplitude=float(np.mean(
                [h['frac_neg_amplitude'] for h in halves])))


def subject_normalised(B, v, run_id, top_pct=2.0, models=('prf', 'wprf'),
                       max_points=40000, seed=0, space=None):
    runs = np.asarray(run_id)
    odd = np.isin(runs, np.unique(runs)[::2])
    rng = np.random.default_rng(seed)
    out = {}
    for mdl in models:
        out[(mdl, 'circular')] = normalised(B, v, B, v, top_pct, model=mdl,
                                            space=space)
        halves = [normalised(B[tr], v[tr], B[~tr], v[~tr], top_pct, model=mdl,
                             space=space)
                  for tr in (odd, ~odd)]
        out[(mdl, 'heldout')] = combine(halves)
    for k in out:
        pts = out[k]['pts']
        if len(pts) > max_points:
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
        out[k]['pts'] = pts
    return out


def centre_minus_flank(binned, good):
    cc = np.abs(X_CENTERS) < 0.5
    ff = (np.abs(X_CENTERS) > 1.5) & (np.abs(X_CENTERS) < 3)
    with np.errstate(all='ignore'):
        return (np.nanmean(binned[cc & good]) - np.nanmean(binned[ff & good]))


def effective_df_numerosity(v, space=None):
    """Trace of the ridge hat matrix for the 11-basis + intercept design, on the
    normalised stimulus axis -- directly comparable to the value number."""
    return effective_df(to01(v, space).astype(float))


# ── fast (numpy) weighted-PRF, validated against braincoder ──────────────────
# braincoder's WeightFitter(alpha) solves (A'A + alpha I) w = A' b in TensorFlow
# with A = [11 Gaussian basis | intercept column].  That is a 12x12 linear
# solve, so the TF round-trip dominates: 7.5 s for one leave-one-run-out sweep
# versus 0.02 s in numpy.  Doing 20 permutations x 39 subjects x 3 ROIs through
# TF would be hours; in numpy it is a minute.  `check_equivalence` below runs
# the two against each other on real data and is called once per analysis run,
# so the shortcut is verified rather than assumed (observed max |dcvR^2| = 2e-6
# on both the numerosity and the value data -- float32 vs float64 rounding).

WPRF_MUS = np.linspace(0., 1., K_POP)


def wprf_design(x01):
    A = np.exp(-(np.asarray(x01, float)[:, None] - WPRF_MUS[None, :]) ** 2
               / (2 * SD_POP ** 2))
    return np.hstack([A, np.ones((len(A), 1))])


def wprf_fit(A, B, alpha=ALPHA):
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ B)


def cv_wprf_fast(B, v, groups, alpha=ALPHA, return_fold_curves=False,
                 space=None):
    A = wprf_design(to01(v, space))
    g = np.asarray(groups)
    Agrid = wprf_design(GRID01)
    pred = np.empty_like(B)
    curves = []
    for gg in np.unique(g):
        tr, te = g != gg, g == gg
        w = wprf_fit(A[tr], B[tr], alpha)
        pred[te] = A[te] @ w
        if return_fold_curves:
            curves.append(Agrid @ w)
    r2 = 1 - ((B - pred) ** 2).sum(0) / ((B - B.mean(0)) ** 2).sum(0)
    if return_fold_curves:
        return r2, np.stack(curves)
    return r2


def wprf_full(B, v, alpha=ALPHA, space=None):
    """Full-data weighted-PRF tuning curve per voxel, plus the readouts the
    example panels need (preferred numerosity, effective width, peak height).

    Read out only inside the stimulus range actually presented, exactly as
    plot_example_voxels.py does for value."""
    space = space or DEFAULT_SPACE
    A = wprf_design(to01(v, space))
    w = wprf_fit(A, B, alpha)
    gstim = space.grid01_to_stim(GRID01.astype(float))   # raw stimulus units
    curves = wprf_design(GRID01) @ w
    sup = (gstim >= np.min(v)) & (gstim <= np.max(v))
    cs = curves[sup]
    pref = gstim[sup][np.nanargmax(cs, axis=0)]
    peak_height = cs.max(0) - cs.min(0)
    half = cs.min(0) + 0.5 * peak_height
    eff_width = ((cs > half[None, :]).sum(0) / sup.sum()
                 * (gstim[sup].max() - gstim[sup].min()))
    return dict(weights=w, curves=curves, grid=gstim, pref=pref,
                eff_width=eff_width, peak_height=peak_height)


def permutation_null_fast(B, v, range_wide, groups, n_perm=20, seed=0,
                          space=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        vperm = np.asarray(v, float).copy()
        for cond in (False, True):
            sel = range_wide == cond
            if sel.sum():
                vperm[sel] = rng.permutation(vperm[sel])
        out.append(cv_wprf_fast(B, vperm, groups, space=space))
    return np.stack(out)


def check_equivalence(B, v, groups):
    """max |cvR^2_braincoder - cvR^2_numpy| on the data at hand."""
    return float(np.abs(cv_wprf(B, v, groups) - cv_wprf_fast(B, v, groups)).max())


def load_subject_narrow(subject, roi='NPCr', data_dir=DATA_DIR, smoothed=False):
    """`load_subject` restricted to the NARROW range condition (10-25)."""
    return load_subject(subject, roi=roi, data_dir=data_dir, smoothed=smoothed,
                        narrow_only=True)
