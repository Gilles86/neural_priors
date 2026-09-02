"""What does the held-out peak-aligned statistic score when there is no tuning?

The held-out normalised-axis test takes mu, sigma and the voxel selection from
one half of the runs and scores the OTHER half, so it is not circular.  That
does not by itself prove the resulting centre-minus-flank is zero under the
null: split-half structure shared between the two halves (scanner drift, a
voxel's own noise autocorrelation, the stimulus distribution) could in principle
produce a positive value with no stimulus tuning at all.

So: destroy the stimulus-response link with the same label permutation used for
the cvR^2 null, re-run the whole held-out pipeline, and see what centre-minus-
flank comes out.  Anything the real data scores above THIS is real.

Both datasets are run through their own pipeline (numerosity: shuffle n within
range condition; value: shuffle value over items) so the two nulls are directly
comparable to the two real numbers.

Run:
  KERAS_BACKEND=tensorflow PYTHONPATH=~/git/neural_priors:~/git/value_prf \
      ~/mambaforge/envs/braincoder/bin/python \
      -m neural_priors.value_comparison.heldout_null --n-perm 5
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

from neural_priors.value_comparison import matched_prf as M        # noqa: E402
from neural_priors.utils.data import get_all_subject_ids           # noqa: E402
from value_prf.utils.data import Subject, get_all_subjects         # noqa: E402
from value_prf.encoding_model.prf_cv import (                      # noqa: E402
    fit_grid_ols, _design, SD_GRIDS, VALUE_MAX, N_MU)

VAL_MU_GRID = np.linspace(0., VALUE_MAX, N_MU)
VAL_SD_GRID = SD_GRIDS['bounded']
MIN_VOX_FRAC = 0.25
TOP_PCT = 2.0


def normalised_generic(B_fit, v_fit, B_show, v_show, mus, sds, top_pct=TOP_PCT):
    """The held-out normalised-axis record, for either stimulus space."""
    _, mu, sd, beta, r2 = fit_grid_ols(B_fit, v_fit, B_fit[:1], v_fit[:1],
                                       mus, sds, positive_only=True)
    keep = r2 >= np.nanpercentile(r2, 100 - top_pct)
    mu, sd, beta = mu[keep], sd[keep], beta[:, keep]
    S = B_show[:, keep]
    m0, s0 = S.mean(0), S.std(0, ddof=1) + 1e-12
    Z = (S - m0) / s0
    x = (v_show[:, None] - mu[None, :]) / sd[None, :]
    xr, zr = x.ravel(), Z.ravel()
    ok = (xr >= M.X_EDGES[0]) & (xr <= M.X_EDGES[-1])
    xr, zr = xr[ok], zr[ok]
    vox = np.tile(np.arange(len(mu)), (len(v_show), 1)).ravel()[ok]
    bidx = np.digitize(xr, M.X_EDGES[1:-1])
    binned = np.full(M.N_X_BINS, np.nan)
    nvb = np.zeros(M.N_X_BINS)
    counts = np.zeros(M.N_X_BINS)
    for b in range(M.N_X_BINS):
        sel = bidx == b
        if sel.any():
            binned[b] = zr[sel].mean()
            nvb[b] = len(np.unique(vox[sel]))
            counts[b] = sel.sum()
    return binned, nvb, len(mu), counts


def heldout_cf(B, v, runs, mus, sds):
    """Split-half centre-minus-flank, the two directions pooled exactly as
    matched_prf.combine() pools them (count-weighted, not a plain mean), so this
    null is on the same scale as the real numbers."""
    odd = np.isin(runs, np.unique(runs)[::2])
    bs, ns, nv, cs = [], [], [], []
    for tr in (odd, ~odd):
        b, n, k, c = normalised_generic(B[tr], v[tr], B[~tr], v[~tr], mus, sds)
        bs.append(b)
        ns.append(n)
        nv.append(k)
        cs.append(c)
    cnt = np.sum(cs, 0)
    safe = np.where(cnt, cnt, np.nan)
    with np.errstate(all='ignore'):
        binned = np.nansum([b * c for b, c in zip(bs, cs)], 0) / safe
    frac = np.sum(ns, 0) / max(np.sum(nv), 1)
    return M.centre_minus_flank(binned, frac >= MIN_VOX_FRAC)


def numerosity_null(n_perm, subjects, roi='NPCr', samples=('full', 'matched128'),
                    dataset='neural_priors'):
    """Held-out centre-minus-flank, real vs permuted labels, for either
    numerosity dataset.  The stimulus space (linear 10-40 for neural_priors,
    log 7-86 for tms_risk) comes from matched_prf.SPACES."""
    from neural_priors.value_comparison.run_analysis import DATASETS
    cfg = DATASETS[dataset]
    space = M.SPACES[cfg['space']]
    load = getattr(M, cfg['loader'])
    rows = []
    for sample in samples:
        for s in subjects:
            B, n, run, rw = load(s, roi)
            if sample == 'matched128':
                B, n, run, rw = M.subsample_matched(B, n, run, rw, seed=int(s))
            nt = space.t(n)
            real = heldout_cf(B, nt, run, space.mu_grid, space.sd_grid)
            rng = np.random.default_rng(int(s))
            null = []
            for _ in range(n_perm):
                vp = nt.copy()
                for cond in (False, True):
                    sel = rw == cond
                    if sel.sum():
                        vp[sel] = rng.permutation(vp[sel])
                null.append(heldout_cf(B, vp, run, space.mu_grid, space.sd_grid))
            rows.append(dict(dataset=cfg['prefix'], roi=roi, sample=sample,
                             subject=s, real_cf=real, null_cf_mean=np.mean(null),
                             null_cf_sd=np.std(null, ddof=1),
                             p=float(np.mean(np.asarray(null) >= real))))
            print(f"  {dataset} {roi}/{sample} sub-{s}: real {real:+.3f}  null "
                  f"{np.mean(null):+.3f}", flush=True)
    return rows


def value_null(n_perm, datasets=(1, 2), rois=('vmpfc', 'v1')):
    rows = []
    for ds in datasets:
        for roi in rois:
            for s in get_all_subjects(ds):
                sub = Subject(s, dataset=ds)
                try:
                    masker = sub.get_roi_masker(roi)
                    beh = sub.get_behavior()
                    betas = (sub.get_single_trial_betas(roi, denoise=True,
                                                        masker=masker)
                             .loc[beh.index].astype(np.float32))
                except (FileNotFoundError, ValueError, KeyError) as e:
                    print(f'  ds{ds} {roi} sub-{s:05d}: skip ({e})', flush=True)
                    continue
                B = np.asarray(betas, float)
                v900 = beh['value'].values.astype(float) * VALUE_MAX
                runs = betas.index.get_level_values('run').values
                items = beh['item'].values
                real = heldout_cf(B, v900, runs, VAL_MU_GRID, VAL_SD_GRID)
                rng = np.random.default_rng(s)
                uitems = np.unique(items)
                item_value = pd.Series(v900, index=items).groupby(
                    level=0).first()
                null = []
                for _ in range(n_perm):
                    perm = pd.Series(rng.permutation(item_value.values),
                                     index=uitems)
                    null.append(heldout_cf(B, perm.loc[items].values.astype(float),
                                           runs, VAL_MU_GRID, VAL_SD_GRID))
                rows.append(dict(dataset='value', value_dataset=ds, roi=roi,
                                 sample='full', subject=s, real_cf=real,
                                 null_cf_mean=np.mean(null),
                                 null_cf_sd=np.std(null, ddof=1),
                                 p=float(np.mean(np.asarray(null) >= real))))
                print(f"  ds{ds} {roi} sub-{s:05d}: real {real:+.3f}  null "
                      f"{np.mean(null):+.3f}", flush=True)
    return rows


def main(datadir, n_perm=5, do_value=True, n_subjects=None,
         datasets=('neural_priors',), out='heldout_centreflank_null.tsv'):
    from neural_priors.value_comparison.run_analysis import get_subjects_for
    rows = []
    for ds in datasets:
        subs = get_subjects_for(ds)
        if n_subjects:
            subs = subs[:n_subjects]
        print(f'== {ds} ({len(subs)} subjects) ==', flush=True)
        samples = ('full', 'matched128') if ds == 'neural_priors' else ('full',)
        rows += numerosity_null(n_perm, subs, 'NPCr', samples, dataset=ds)
        rows += numerosity_null(n_perm, subs, 'wholebrain', ('full',),
                                dataset=ds)
    if do_value:
        print('== value ==', flush=True)
        rows += value_null(n_perm)
    df = pd.DataFrame(rows)
    Path(datadir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(datadir) / out, sep='\t', index=False)
    print(f'\nsaved {datadir}/{out}')
    key = ['dataset', 'roi', 'sample'] + (
        ['value_dataset'] if 'value_dataset' in df else [])
    for k, g in df.groupby([c for c in key if c in df]):
        t, p = stats.ttest_rel(g.real_cf, g.null_cf_mean)
        print(f'  {k}: real {g.real_cf.mean():+.4f}  null '
              f'{g.null_cf_mean.mean():+.4f}  diff '
              f'{(g.real_cf - g.null_cf_mean).mean():+.4f}  '
              f't({len(g)-1})={t:.2f} p={p:.3g}')
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datadir',
                   default=str(Path.home() / 'git/value_prf/notes/data'))
    p.add_argument('--n-perm', type=int, default=5)
    p.add_argument('--n-subjects', type=int, default=None)
    p.add_argument('--no-value', action='store_true')
    p.add_argument('--datasets', default='neural_priors')
    p.add_argument('--out', default='heldout_centreflank_null.tsv')
    a = p.parse_args()
    main(a.datadir, n_perm=a.n_perm, do_value=not a.no_value,
         n_subjects=a.n_subjects, datasets=tuple(a.datasets.split(',')),
         out=a.out)
