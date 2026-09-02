"""Is the peak-aligned effect actually a PEAK, or just a monotonic ramp?

If a voxel's fitted mu sits at the edge of the stimulus range, the normalised
axis x = (stimulus - mu)/sigma only ever covers one side, and "centre minus
flanks" reduces to "one end of the stimulus range minus the other" -- which a
purely monotonic response produces with no tuning at all.  The best-fitting
numerosity voxels look monotonic by eye, so this has to be checked rather than
assumed.

Two things are computed here, for both datasets, on the held-out split only:
  1. the distribution of fitted mu among the SELECTED voxels;
  2. the held-out centre-minus-flank restricted to INTERIOR-mu voxels, i.e.
     voxels whose preferred stimulus is at least 1 basis-sd away from both ends
     of the range, so that both flanks are actually sampled.

Run:
  KERAS_BACKEND=tensorflow PYTHONPATH=~/git/neural_priors:~/git/value_prf \
      ~/mambaforge/envs/braincoder/bin/python \
      -m neural_priors.value_comparison.interior_mu
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.environ.get('VALUE_PRF_DIR',
                                  os.path.expanduser('~/git/value_prf')))

from neural_priors.value_comparison import matched_prf as M         # noqa: E402
from neural_priors.utils.data import get_all_subject_ids            # noqa: E402
from value_prf.utils.data import Subject, get_all_subjects          # noqa: E402
from value_prf.encoding_model.prf_cv import (                       # noqa: E402
    fit_grid_ols, SD_GRIDS, VALUE_MAX, N_MU)

VAL_MU_GRID = np.linspace(0., VALUE_MAX, N_MU)
VAL_SD_GRID = SD_GRIDS['bounded']
MIN_VOX_FRAC = 0.25
TOP_PCT = 2.0
# "interior" = at least one basis sd (0.075 of the range) from either end
VAL_INT = (0.075 * VALUE_MAX, VALUE_MAX - 0.075 * VALUE_MAX)          # 67.5-832.5
NUM_INT = (M.STIM_LO + 0.075 * M.STIM_RANGE, M.STIM_HI - 0.075 * M.STIM_RANGE)


def half_record(B_fit, v_fit, B_show, v_show, mus, sds, interior):
    _, mu, sd, beta, r2 = fit_grid_ols(B_fit, v_fit, B_fit[:1], v_fit[:1],
                                       mus, sds, positive_only=True)
    keep = r2 >= np.nanpercentile(r2, 100 - TOP_PCT)
    mu, sd = mu[keep], sd[keep]
    S = B_show[:, keep]
    if interior is not None:
        sel = (mu >= interior[0]) & (mu <= interior[1])
        if sel.sum() < 5:
            return None
        mu, sd, S = mu[sel], sd[sel], S[:, sel]
    Z = (S - S.mean(0)) / (S.std(0, ddof=1) + 1e-12)
    x = (v_show[:, None] - mu[None, :]) / sd[None, :]
    xr, zr = x.ravel(), Z.ravel()
    ok = (xr >= M.X_EDGES[0]) & (xr <= M.X_EDGES[-1])
    xr, zr = xr[ok], zr[ok]
    vox = np.tile(np.arange(len(mu)), (len(v_show), 1)).ravel()[ok]
    bidx = np.digitize(xr, M.X_EDGES[1:-1])
    binned = np.full(M.N_X_BINS, np.nan)
    nvb = np.zeros(M.N_X_BINS)
    cnt = np.zeros(M.N_X_BINS)
    for b in range(M.N_X_BINS):
        s = bidx == b
        if s.any():
            binned[b] = zr[s].mean()
            nvb[b] = len(np.unique(vox[s]))
            cnt[b] = s.sum()
    return binned, nvb, len(mu), cnt, mu


def heldout(B, v, runs, mus, sds, interior=None):
    odd = np.isin(runs, np.unique(runs)[::2])
    recs = [half_record(B[tr], v[tr], B[~tr], v[~tr], mus, sds, interior)
            for tr in (odd, ~odd)]
    recs = [r for r in recs if r is not None]
    if not recs:
        return np.nan, np.array([])
    cnt = np.sum([r[3] for r in recs], 0)
    safe = np.where(cnt, cnt, np.nan)
    with np.errstate(all='ignore'):
        binned = np.nansum([r[0] * r[3] for r in recs], 0) / safe
    frac = np.sum([r[1] for r in recs], 0) / max(sum(r[2] for r in recs), 1)
    return (M.centre_minus_flank(binned, frac >= MIN_VOX_FRAC),
            np.concatenate([r[4] for r in recs]))


def main(datadir):
    rows, mus_rows = [], []
    print('== numerosity ==', flush=True)
    for s in get_all_subject_ids():
        B, n, run, rw = M.load_subject(s, 'NPCr')
        allcf, mu_all = heldout(B, n, run, M.MU_GRID, M.SD_GRID)
        intcf, mu_int = heldout(B, n, run, M.MU_GRID, M.SD_GRID, NUM_INT)
        rows.append(dict(dataset='numerosity', roi='NPCr', subject=s,
                         cf_all=allcf, cf_interior=intcf,
                         frac_interior=float(np.mean((mu_all >= NUM_INT[0])
                                                     & (mu_all <= NUM_INT[1]))),
                         median_mu=float(np.median(mu_all))))
        for m in mu_all:
            mus_rows.append(dict(dataset='numerosity', roi='NPCr', subject=s,
                                 mu=float(m)))
        print(f'  sub-{s}: all {allcf:+.3f}  interior {intcf:+.3f}  '
              f'({100*rows[-1]["frac_interior"]:.0f}% interior)', flush=True)

    print('== value ==', flush=True)
    for ds in (1, 2):
        for roi in ('vmpfc', 'v1'):
            for s in get_all_subjects(ds):
                sub = Subject(s, dataset=ds)
                try:
                    mk = sub.get_roi_masker(roi)
                    beh = sub.get_behavior()
                    bet = (sub.get_single_trial_betas(roi, denoise=True,
                                                      masker=mk)
                           .loc[beh.index].astype(np.float32))
                except (FileNotFoundError, ValueError, KeyError):
                    continue
                B = np.asarray(bet, float)
                v = beh['value'].values.astype(float) * VALUE_MAX
                runs = bet.index.get_level_values('run').values
                allcf, mu_all = heldout(B, v, runs, VAL_MU_GRID, VAL_SD_GRID)
                intcf, _ = heldout(B, v, runs, VAL_MU_GRID, VAL_SD_GRID, VAL_INT)
                rows.append(dict(dataset='value', value_dataset=ds,
                                 roi={'vmpfc': 'vmPFC', 'v1': 'V1'}[roi],
                                 subject=s, cf_all=allcf, cf_interior=intcf,
                                 frac_interior=float(np.mean(
                                     (mu_all >= VAL_INT[0]) & (mu_all <= VAL_INT[1]))),
                                 median_mu=float(np.median(mu_all))))
                for m in mu_all:
                    mus_rows.append(dict(dataset='value', roi=roi, subject=s,
                                         mu=float(m)))
                print(f'  ds{ds} {roi} sub-{s:05d}: all {allcf:+.3f}  '
                      f'interior {intcf:+.3f}', flush=True)

    df = pd.DataFrame(rows)
    Path(datadir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(datadir) / 'heldout_interior_mu.tsv', sep='\t', index=False)
    pd.DataFrame(mus_rows).to_csv(Path(datadir) / 'heldout_fitted_mu.tsv',
                                  sep='\t', index=False)
    print(f'\nsaved {datadir}/heldout_interior_mu.tsv')
    for k, g in df.groupby([c for c in ('dataset', 'roi') if c in df]):
        for col in ('cf_all', 'cf_interior'):
            v = g[col].dropna()
            t, p = stats.ttest_1samp(v, 0)
            print(f'  {k} {col:12s}: {v.mean():+.4f}  t({len(v)-1})={t:.2f} '
                  f'p={p:.3g}  n={len(v)}')
        print(f'  {k} interior fraction: {g.frac_interior.mean():.2f}, '
              f'median mu {g.median_mu.median():.1f}')
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datadir',
                   default=str(Path.home() / 'git/value_prf/notes/data'))
    a = p.parse_args()
    main(a.datadir)
