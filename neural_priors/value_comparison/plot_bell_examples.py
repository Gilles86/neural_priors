#!/usr/bin/env python3
"""Numerosity voxels with a genuine BELL, not a ramp.

`plot_numerosity_vs_value.py` shows the best-fitting NPC voxels, and those
prefer LOW numerosities (median mu = 13.5 of a 10-40 range), so over the sampled
range their tuning looks monotonic.  That is honest but it does not show what a
numerosity PRF looks like when its peak is inside the range.

This selects voxels that are non-monotonic by construction and by evidence:

  * preferred numerosity strictly interior (MU_LO < mu < MU_HI), so a peak is
    visible with data on BOTH sides of it;
  * cvR^2 above that subject's own permutation-null p95 -- not merely the best
    of many;
  * at least MIN_AT_PEAK trials within one fitted sigma of the peak, so the peak
    rests on data rather than on the grid;
  * few local maxima and binned points that follow the curve, so the panel is
    legible rather than a scribble.

The curve and the leave-one-run-out band are refitted here with the same
`fit_grid_ols` used everywhere else, so nothing about the estimator changes --
only which voxels are shown.

Run:
  KERAS_BACKEND=tensorflow PYTHONPATH=~/git/neural_priors:~/git/value_prf \\
  ~/mambaforge/envs/braincoder/bin/python \\
      -m neural_priors.value_comparison.plot_bell_examples
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from neural_priors.value_comparison import matched_prf as M
from value_prf.encoding_model.prf_cv import fit_grid_ols
from value_prf.visualize import set_style, panel_letter

OUT_FIG = Path.home() / 'git/value_prf/notes/figures'
OUT_DAT = Path.home() / 'git/value_prf/notes/data'
C_NUM = '#5D8C3F'
MU_LO, MU_HI = 17., 33.          # interior of the 10-40 numerosity range
MIN_AT_PEAK = 20
MAX_PEAKS = 4
MIN_BIN_R = 0.85
N_BINS = 12


def refit_mu(B, nvals, vox, space):
    """The peak this script will actually DRAW, from a full-data refit."""
    _, mu, sd, beta, _ = fit_grid_ols(B[:, [vox]], nvals, B[:1, [vox]],
                                      nvals[:1], space.mu_grid, space.sd_grid,
                                      positive_only=True)
    return float(mu[0]), float(sd[0]), beta[:, 0]


def pick(cand, n=8, space=None, cache=None, roi=None):
    """Interior-peaked, above its own null, data-supported, legible.

    The interior-peak test is applied to the peak this script REFITS and draws,
    not to the candidate table's stored value.  Those two disagree (different
    grids), and selecting on one while displaying the other put four monotonic
    ramps in a figure whose whole point was that they are not ramps.
    """
    if roi is not None:
        # the comparison figure looks up ONE ROI's permutation-null threshold,
        # so mixing NPCl and NPCr here would score voxels against the wrong null
        cand = cand[cand.roi == roi]
    c = cand[(cand.preferred_n > MU_LO - 6) & (cand.preferred_n < MU_HI + 6)
             & (cand.cvr2_loro > cand.null_p95)
             & (cand.n_trials_at_peak >= MIN_AT_PEAK)
             & (cand.n_peaks <= MAX_PEAKS) & (cand.bin_r > MIN_BIN_R)]
    c = c.sort_values('cvr2_loro', ascending=False)
    keep = []
    for _, r in c.iterrows():
        key = (int(r.subject), r.roi)
        if key not in cache:
            cache[key] = M.load_subject(f'{int(r.subject):02d}', roi=r.roi)
        B, nvals, _, _ = cache[key]
        mu, sd, _ = refit_mu(B, nvals, int(r.voxel), space)
        if MU_LO < mu < MU_HI:
            keep.append(dict(r, refit_mu=mu, refit_sd=sd))
        if len(keep) >= 400:
            break
    c = pd.DataFrame(keep)
    n_avail = len(c)
    # spread the shown voxels over preferred numerosity so the panel shows a
    # range of peaks rather than eight copies of the same one
    edges = np.linspace(MU_LO, MU_HI, n + 1)
    picks, used = [], set()
    for i in range(n):
        s = c[(c.refit_mu >= edges[i]) & (c.refit_mu < edges[i + 1])]
        for _, r in s.iterrows():
            k = (r.subject, r.roi, r.voxel)
            if k not in used:
                picks.append(r); used.add(k); break
    for _, r in c.iterrows():
        if len(picks) >= n:
            break
        k = (r.subject, r.roi, r.voxel)
        if k not in used:
            picks.append(r); used.add(k)
    return pd.DataFrame(picks).sort_values('refit_mu'), n_avail


def main(n_examples=8, out_fig=OUT_FIG, out_dat=OUT_DAT, roi='NPCr'):
    set_style()
    cand = pd.read_csv(OUT_DAT / 'numerosity_example_candidates.tsv', sep='\t')
    space = M.SPACES['neural_priors']
    cache = {}
    picks, n_avail = pick(cand, n_examples, space=space, cache=cache,
                          roi=roi)
    print(f'{n_avail} voxels have an interior REFITTED peak, beat their own '
          f'null p95, and are data-supported; showing {len(picks)}')

    grid = np.linspace(space.lo, space.hi, 200)
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 1.95 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    tidy = []

    for ax, (_, r) in zip(axes, picks.iterrows()):
        key = (int(r.subject), r.roi)
        if key not in cache:
            cache[key] = M.load_subject(f'{int(r.subject):02d}', roi=r.roi)
        B, nvals, run_id, _ = cache[key]
        y = B[:, int(r.voxel)]

        # full-data curve and the leave-one-run-out envelope, same estimator
        curves = []
        for g in np.unique(run_id):
            tr = run_id != g
            _, mu_g, sd_g, beta_g, _ = fit_grid_ols(
                B[tr][:, [int(r.voxel)]], nvals[tr], B[:1, [int(r.voxel)]],
                nvals[:1], space.mu_grid, space.sd_grid, positive_only=True)
            curves.append(beta_g[0, 0] * np.exp(-0.5 * ((grid - mu_g[0])
                                                        / sd_g[0]) ** 2)
                          + beta_g[1, 0])
        _, mu_f, sd_f, beta_f, _ = fit_grid_ols(
            B[:, [int(r.voxel)]], nvals, B[:1, [int(r.voxel)]], nvals[:1],
            space.mu_grid, space.sd_grid, positive_only=True)
        curve = (beta_f[0, 0] * np.exp(-0.5 * ((grid - mu_f[0]) / sd_f[0]) ** 2)
                 + beta_f[1, 0])
        lo, hi = np.min(curves, 0), np.max(curves, 0)

        edges = np.unique(np.quantile(nvals, np.linspace(0, 1, N_BINS + 1)))
        idx = np.digitize(nvals, edges[1:-1])
        bx, bm, bs = [], [], []
        for b in range(len(edges) - 1):
            s = idx == b
            if s.sum() > 1:
                bx.append(nvals[s].mean()); bm.append(y[s].mean())
                bs.append(y[s].std(ddof=1) / np.sqrt(s.sum()))
        bx, bm, bs = map(np.array, (bx, bm, bs))

        ax.scatter(nvals, y, s=3.5, color='0.78', alpha=0.55, lw=0, zorder=1,
                   rasterized=True)
        ax.fill_between(grid, lo, hi, color=C_NUM, alpha=0.22, lw=0, zorder=2)
        ax.plot(grid, curve, color=C_NUM, lw=1.6, zorder=3)
        ax.errorbar(bx, bm, yerr=bs, fmt='o', color='0.12', ms=3.0, lw=0.9,
                    capsize=0, zorder=4)
        ax.axvline(mu_f[0], color='0.55', lw=0.7, ls='--', zorder=1)
        lo_y, hi_y = np.percentile(y, [3, 97])
        pad = 0.12 * (hi_y - lo_y)
        ax.set_ylim(min(lo_y, (bm - bs).min()) - pad,
                    max(hi_y, (bm + bs).max()) + pad)
        ax.set_xticks([10, 20, 30, 40])
        ax.tick_params(labelsize=6.5)
        ax.set_title(f'sub-{int(r.subject):02d} {r.roi}  ·  peak n = '
                     f'{mu_f[0]:.0f}\ncvR² = {r.cvr2_loro:.3f} '
                     f'(null p95 = {r.null_p95:.3f})', fontsize=6.2)
        meta = dict(subject=int(r.subject), roi=r.roi, voxel=int(r.voxel),
                    preferred_n=float(mu_f[0]), sigma=float(sd_f[0]),
                    cvr2_loro=float(r.cvr2_loro), null_p95=float(r.null_p95))
        tidy += [dict(**meta, kind='trial', numerosity=float(a), y=float(b),
                      run=int(c)) for a, b, c in zip(nvals, y, run_id)]
        tidy += [dict(**meta, kind='binned', numerosity=float(a), y=float(b),
                      yerr=float(c)) for a, b, c in zip(bx, bm, bs)]
        tidy += [dict(**meta, kind='model', numerosity=float(a), y=float(b),
                      band_lo=float(c), band_hi=float(d))
                 for a, b, c, d in zip(grid, curve, lo, hi)]

    for k, ax in enumerate(axes):
        if k >= len(picks):
            ax.set_visible(False); continue
        if k % ncol == 0:
            ax.set_ylabel('Response (a.u.)', fontsize=7)
        if k // ncol == nrow - 1:
            ax.set_xlabel('Numerosity', fontsize=7)

    fig.suptitle('Numerosity voxels with a peak INSIDE the sampled range — '
                 'genuinely bell-shaped, not a ramp', fontsize=9,
                 fontweight='bold')
    fig.text(0.5, -0.035,
             f'Selected on: preferred n strictly inside {MU_LO:.0f}-{MU_HI:.0f} '
             f'(so there is data on both sides of the peak); cvR² above that '
             f'subject’s OWN permutation-null p95;\n'
             f'≥ {MIN_AT_PEAK} trials within one fitted σ of the peak. '
             f'At least {n_avail} voxels qualify. Grey: every single trial. Points: '
             f'{N_BINS} equal-N bins ± SEM. Band: leave-one-run-out refits.\n'
             f'Dashed line: the fitted peak. Same estimator as everywhere else '
             f'— only the SELECTION differs.',
             ha='center', va='top', fontsize=6.3, color='0.25', linespacing=1.4)
    sns.despine(offset=2, trim=False)
    out_fig = Path(out_fig)
    out_fig.mkdir(parents=True, exist_ok=True)
    stem = out_fig / 'numerosity_bell_examples'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', bbox_inches='tight', pad_inches=0.02,
                    dpi=300 if ext == 'png' else None)
    print(f'saved {stem}.pdf / .png')
    fn = Path(out_dat) / 'data_numerosity_bell_examples.tsv'
    td = pd.DataFrame(tidy)
    td.to_csv(fn, sep='\t', index=False)
    print(f'saved {fn}')

    # Also emit the three files in the schema `plot_numerosity_vs_value.py`
    # reads, so its panel a can show these bells instead of the best-fitting
    # (and therefore edge-preferring, ramp-looking) voxels.
    meta_cols = ['subject', 'roi', 'voxel', 'preferred_n', 'cvr2_loro']
    base = td[meta_cols].drop_duplicates()
    nn = pd.read_csv(Path(out_dat) / 'numerosity_voxel_stats.tsv', sep='\t')
    p99 = (nn[nn['sample'] == 'full'].groupby('subject')['null_max_p99'].first()
           if 'null_max_p99' in nn else None)
    for kind, extra, name in (
            ('trial', {'stimulus': 'numerosity', 'measured_beta': 'y'},
             'numerosity_bell_trials'),
            ('binned', {'stimulus_bin_center': 'numerosity',
                        'measured_mean': 'y', 'measured_sem': 'yerr'},
             'numerosity_bell_bins'),
            ('model', {'stimulus_grid': 'numerosity',
                       'model_prediction': 'y', 'model_loro_lo': 'band_lo',
                       'model_loro_hi': 'band_hi'}, 'numerosity_bell_curves')):
        sub = td[td['kind'] == kind].copy()
        out = sub[meta_cols].copy()
        out.insert(0, 'dataset', 'numerosity')
        out['eff_width'] = sub['sigma'].values * 2.355
        out['null_p99'] = (out.subject.map(p99).values if p99 is not None
                           else np.nan)
        out['edf'] = np.nan
        for dest, srccol in extra.items():
            out[dest] = sub[srccol].values
        if kind == 'trial':
            out['run'] = sub['run'].values.astype(int)
        if kind == 'binned':
            out['n_trials_in_bin'] = np.nan
        fn2 = Path(out_dat) / f'{name}.tsv'
        out.to_csv(fn2, sep='\t', index=False)
        print(f'saved {fn2}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-examples', type=int, default=8)
    p.add_argument('--roi', default='NPCr')
    a = p.parse_args()
    main(n_examples=a.n_examples, roi=a.roi)
