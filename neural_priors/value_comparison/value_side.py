"""The value-tuning statistics, recomputed with the SAME estimator used for
numerosity, so the two columns of the comparison table cannot differ because of
the code that produced them.

value_prf already reports these numbers (notes/data/data_example_voxel_cvr2_null.tsv,
notes/data/data_prf_normalized_per_subject_ds{1,2}.tsv).  This script reproduces
the single-voxel part for BOTH datasets with the validated numpy ridge, adds the
effective-df / trials-per-parameter bookkeeping, and writes one tidy file with a
`dataset` column so it can be concatenated with the numerosity output.

Only the permutation scheme differs from the numerosity side, because the
designs differ: value permutes the value labels over ITEMS (each item is shown
twice, so this preserves the repeat structure), numerosity permutes numerosity
over trials WITHIN range condition (there are no repeated items, and the
narrow/wide blocks are confounded with run).  Both destroy exactly the
trial-level stimulus-response relation and nothing else.

Run:
  KERAS_BACKEND=tensorflow ~/mambaforge/envs/braincoder/bin/python \
      -m neural_priors.value_comparison.value_side
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.environ.get('VALUE_PRF_DIR',
                                  os.path.expanduser('~/git/value_prf')))

from value_prf.utils.data import Subject, get_all_subjects          # noqa: E402
from value_prf.visualize.plot_example_voxels import (               # noqa: E402
    effective_df, K_POP, SD_POP, ALPHA)
from value_prf.encoding_model.prf_cv import (                       # noqa: E402
    fit_grid_ols, SD_GRIDS, VALUE_MAX, N_MU)

CV_THRESHOLDS = (0.05, 0.1, 0.15)
MUS = np.linspace(0., 1., K_POP)
MU_GRID = np.linspace(0., VALUE_MAX, N_MU)


def design(x01):
    A = np.exp(-(np.asarray(x01, float)[:, None] - MUS[None, :]) ** 2
               / (2 * SD_POP ** 2))
    return np.hstack([A, np.ones((len(A), 1))])


def cv_wprf_fast(B, x01, groups, alpha=ALPHA):
    A = design(x01)
    g = np.asarray(groups)
    pred = np.empty_like(B)
    I = alpha * np.eye(A.shape[1])
    for gg in np.unique(g):
        tr, te = g != gg, g == gg
        w = np.linalg.solve(A[tr].T @ A[tr] + I, A[tr].T @ B[tr])
        pred[te] = A[te] @ w
    return 1 - ((B - pred) ** 2).sum(0) / ((B - B.mean(0)) ** 2).sum(0)


def cv_prf(B, v900, groups):
    pred = np.full_like(B, np.nan)
    for gg in np.unique(groups):
        tr, te = groups != gg, groups == gg
        pred[te] = fit_grid_ols(B[tr], v900[tr], B[te], v900[te], MU_GRID,
                                SD_GRIDS['bounded'], positive_only=True)[0]
    return 1 - ((B - pred) ** 2).sum(0) / ((B - B.mean(0)) ** 2).sum(0)


def permutation_null(B, x01, items, runs, n_perm=20, seed=0):
    """Value labels shuffled over items (value_prf's own scheme)."""
    rng = np.random.default_rng(seed)
    uitems = np.unique(items)
    item_value = pd.Series(np.asarray(x01).ravel(),
                           index=items).groupby(level=0).first()
    out = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(item_value.values), index=uitems)
        out.append(cv_wprf_fast(B, perm.loc[items].values, runs))
    return np.stack(out)


def main(datadir, datasets=(1, 2), rois=('vmpfc', 'v1'), n_perm=20,
         n_subjects=None):
    rows = []
    for ds in datasets:
        subjects = get_all_subjects(ds)
        if n_subjects:
            subjects = subjects[:n_subjects]
        for roi in rois:
            print(f'== ds{ds} {roi} ==', flush=True)
            for s in subjects:
                sub = Subject(s, dataset=ds)
                try:
                    masker = sub.get_roi_masker(roi)
                    beh = sub.get_behavior()
                    betas = (sub.get_single_trial_betas(roi, denoise=True,
                                                        masker=masker)
                             .loc[beh.index].astype(np.float32))
                except (FileNotFoundError, ValueError, KeyError) as e:
                    print(f'  sub-{s:05d}: skip ({e})', flush=True)
                    continue
                B = np.asarray(betas, float)
                x01 = beh['value'].values.astype(float)
                v900 = x01 * VALUE_MAX
                runs = betas.index.get_level_values('run').values
                items = beh['item'].values

                edf = effective_df(x01)
                r2_w = cv_wprf_fast(B, x01, runs)
                r2_p = cv_prf(B, v900, runs)
                null = permutation_null(B, x01, items, runs, n_perm=n_perm,
                                        seed=s)
                row = dict(dataset='value', value_dataset=ds, subject=s,
                           roi=roi, sample='full', n_vox=B.shape[1],
                           n_trials=len(x01), n_runs=len(np.unique(runs)),
                           edf=edf, trials_per_par=len(x01) / edf,
                           wprf_cvr2_median=float(np.median(r2_w)),
                           wprf_cvr2_p999=float(np.percentile(r2_w, 99.9)),
                           wprf_cvr2_max=float(np.max(r2_w)),
                           prf_cvr2_median=float(np.median(r2_p)),
                           prf_cvr2_max=float(np.max(r2_p)),
                           real_max=float(np.max(r2_w)),
                           null_max_mean=float(np.mean(null.max(1))),
                           null_max_p95=float(np.percentile(null.max(1), 95)),
                           p_max=float(np.mean(null.max(1) >= np.max(r2_w))))
                for thr in CV_THRESHOLDS:
                    real_n = int((r2_w > thr).sum())
                    null_n = (null > thr).sum(1)
                    row[f'real_n_gt{thr}'] = real_n
                    row[f'null_n_gt{thr}_mean'] = float(null_n.mean())
                    row[f'p_n_gt{thr}'] = float(np.mean(null_n >= real_n))
                rows.append(row)
                print(f"  sub-{s:05d}: {row['n_vox']:5d} vox  cvR2max="
                      f"{row['wprf_cvr2_max']:.3f} (null p95="
                      f"{row['null_max_p95']:.3f}, p={row['p_max']:.2f})",
                      flush=True)
    df = pd.DataFrame(rows)
    Path(datadir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(datadir) / 'value_voxel_stats.tsv', sep='\t', index=False)
    print(f'saved {datadir}/value_voxel_stats.tsv')
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datadir',
                   default=str(Path.home() / 'git/value_prf/notes/data'))
    p.add_argument('--datasets', default='1,2')
    p.add_argument('--rois', default='vmpfc,v1')
    p.add_argument('--n-perm', type=int, default=20)
    p.add_argument('--n-subjects', type=int, default=None)
    a = p.parse_args()
    main(a.datadir, datasets=tuple(int(x) for x in a.datasets.split(',')),
         rois=tuple(a.rois.split(',')), n_perm=a.n_perm,
         n_subjects=a.n_subjects)
