"""Run the value-matched numerosity analysis over all subjects and write the
source data the figure and the write-up read.

Outputs (default --datadir ~/git/value_prf/notes/data):
  numerosity_voxel_stats.tsv        one row per (subject, roi, sample):
                                    leave-one-run-out cvR^2 of the weighted PRF
                                    and of the single-peak PRF, the
                                    label-permutation null, effective df.
  numerosity_normalized_binned.tsv  one row per (subject, roi, model,
                                    alignment, x-bin) on the normalised axis.
  numerosity_normalized_points.npz  the single-trial cloud for the panels.
  numerosity_examples_{trials,bins,curves}.tsv   example voxels.

Run:
  KERAS_BACKEND=tensorflow ~/mambaforge/envs/braincoder/bin/python \
      -m neural_priors.value_comparison.run_analysis
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neural_priors.value_comparison import matched_prf as M          # noqa: E402
from neural_priors.utils.data import get_all_subject_ids             # noqa: E402

ROIS = ('NPCr', 'NPCl', 'wholebrain')
SAMPLES = ('full', 'matched128')

# tms_risk is the independent low-trial-count numerosity dataset: session 1
# (baseline, pre-cTBS) is 6 runs x 20 = 120 trials, essentially the value
# dataset's 128.  Its n1 spans 7-86, so its axis is logarithmic.
DATASETS = {
    'neural_priors': dict(loader='load_subject', space='neural_priors',
                          rois=('NPCr', 'NPCl', 'wholebrain'),
                          samples=('full', 'matched128'), prefix='numerosity'),
    'tmsrisk': dict(loader='load_subject_tmsrisk', space='tmsrisk',
                    rois=('NPCr', 'NPC12r', 'wholebrain'),
                    samples=('full',), prefix='tmsrisk'),
    # Narrow range only, linear vs log.  The pair isolates the axis at matched
    # trial count; comparing either against the pooled run would confound the
    # axis with the range-shift AND with halving the trials.
    'neural_priors_narrow': dict(loader='load_subject_narrow',
                                 space='neural_priors_narrow',
                                 rois=('NPCr', 'NPCl'), samples=('full',),
                                 prefix='numerosity_narrow'),
    'neural_priors_narrow_log': dict(loader='load_subject_narrow',
                                     space='neural_priors_narrow_log',
                                     rois=('NPCr', 'NPCl'), samples=('full',),
                                     prefix='numerosity_narrowlog'),
}
CV_THRESHOLDS = (0.05, 0.1, 0.15)
N_VALUE_BINS = 16          # equal-N display bins, as in plot_example_voxels.py
MIN_TRIALS_AT_PEAK = 8


def value_bin_edges(v, n_bins=N_VALUE_BINS):
    e = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    e[0] -= 1e-6
    e[-1] += 1e-6
    return e


def binned_response(y, v, edges):
    idx = np.digitize(v, edges[1:-1])
    cx, m, sem, ns = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        cx.append(v[sel].mean())
        m.append(y[sel].mean())
        sem.append(y[sel].std(ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else 0.)
        ns.append(int(sel.sum()))
    return map(np.array, (cx, m, sem, ns))


def display_quality(B, v, curves, grid, sup):
    """bin_r / n_peaks / bin_snr, as in value_prf/visualize/plot_example_voxels.py."""
    edges = value_bin_edges(v)
    bidx = np.digitize(v, edges[1:-1])
    keep = [b for b in range(len(edges) - 1) if (bidx == b).sum() > 1]
    bm = np.stack([B[bidx == b].mean(0) for b in keep])
    bsem = np.stack([B[bidx == b].std(0, ddof=1) / np.sqrt((bidx == b).sum())
                     for b in keep])
    bc = np.array([v[bidx == b].mean() for b in keep])
    gi = np.abs(bc[:, None] - grid[None, :]).argmin(1)
    cm = curves[gi]
    a = bm - bm.mean(0, keepdims=True)
    b = cm - cm.mean(0, keepdims=True)
    bin_r = (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)
    cs = curves[sup]
    d = np.diff(cs, axis=0)
    n_peaks = ((d[:-1] > 0) & (d[1:] <= 0)).sum(0)
    bin_snr = (cs.max(0) - cs.min(0)) / (bsem.mean(0) + 1e-12)
    return bin_r, n_peaks, bin_snr


def analyse_subject_roi(s, roi, sample, n_perm, store_examples,
                        dataset='neural_priors'):
    cfg = DATASETS[dataset]
    space = M.SPACES[cfg['space']]
    B, n, run, rw = getattr(M, cfg['loader'])(s, roi)
    if sample == 'matched128':
        B, n, run, rw = M.subsample_matched(B, n, run, rw, seed=int(s))

    edf = M.effective_df_numerosity(n, space)
    r2_w, fold_curves = M.cv_wprf_fast(B, n, run, return_fold_curves=True,
                                       space=space)
    r2_p = M.cv_prf(B, n, run, space=space)
    null = M.permutation_null_fast(B, n, rw, run, n_perm=n_perm, seed=int(s),
                                   space=space)

    row = dict(dataset=cfg['prefix'], subject=s, roi=roi, sample=sample,
               n_vox=B.shape[1], n_trials=len(n), n_runs=len(np.unique(run)),
               edf=edf, trials_per_par=len(n) / edf,
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

    recs = M.subject_normalised(B, n, run, seed=int(s), space=space)

    ex = None
    if store_examples:
        full = M.wprf_full(B, n, space=space)
        sup = (full['grid'] >= n.min()) & (full['grid'] <= n.max())
        bin_r, n_peaks, bin_snr = display_quality(B, n, full['curves'],
                                                  full['grid'], sup)
        # "how many trials constrain this peak" must be measured on the axis
        # the model works on, otherwise it is meaningless for a log space
        nt = space.t(n)
        n_at_peak = np.array([(np.abs(nt - space.t(p)) <
                               M.SD_POP * space.range).sum()
                              for p in full['pref']])
        ex = dict(
            cand=pd.DataFrame({
                'subject': s, 'roi': roi, 'voxel': np.arange(B.shape[1]),
                'preferred_n': full['pref'], 'eff_width': full['eff_width'],
                'peak_height': full['peak_height'], 'cvr2_loro': r2_w,
                'bin_r': bin_r, 'n_peaks': n_peaks, 'bin_snr': bin_snr,
                'n_trials_at_peak': n_at_peak, 'edf': edf,
                'null_p95': float(np.percentile(null, 95)),
                'null_p99': float(np.percentile(null, 99))}),
            B=B, n=n, run=run, curves=full['curves'], grid=full['grid'],
            fold_lo=fold_curves.min(0), fold_hi=fold_curves.max(0))
    return row, recs, ex


def get_subjects_for(dataset, data_dir=None):
    if dataset == 'neural_priors':
        return list(get_all_subject_ids())
    # tms_risk hardcodes subjects 22 and 49 as outliers (tms_risk/utils/data.py)
    files = sorted(Path(M.TMSRISK_DIR).glob('sub-*_ses-1_trials.npz'))
    subs = [f.name.split('_')[0].replace('sub-', '') for f in files]
    return [s for s in subs if int(s) not in (22, 49)]


def main(datadir, rois=None, samples=None, n_perm=20, subjects=None,
         n_subjects=None, dataset='neural_priors'):
    cfg = DATASETS[dataset]
    space = M.SPACES[cfg['space']]
    rois = rois or cfg['rois']
    samples = samples or cfg['samples']
    prefix = cfg['prefix']
    datadir = Path(datadir)
    datadir.mkdir(parents=True, exist_ok=True)
    subjects = list(subjects or get_subjects_for(dataset))
    if n_subjects:
        subjects = subjects[:n_subjects]
    print(f'== dataset {dataset}: {len(subjects)} subjects, {space} ==',
          flush=True)

    # verify the numpy shortcut against braincoder's own WeightFitter, once,
    # on real data -- never assume the two agree
    B0, n0, run0, _ = getattr(M, cfg['loader'])(subjects[0], rois[0])
    dmax = M.check_equivalence(B0, n0, run0)
    print(f'braincoder vs numpy weighted-PRF cvR^2: max |diff| = {dmax:.2e}',
          flush=True)
    assert dmax < 1e-4, 'numpy weighted-PRF does not match braincoder'

    rows, binned_rows, points, cands, example_store = [], [], {}, [], {}
    for roi in rois:
        for sample in samples:
            print(f'== {roi} / {sample} ==', flush=True)
            for s in subjects:
                store_ex = (sample == 'full')
                try:
                    row, recs, ex = analyse_subject_roi(s, roi, sample, n_perm,
                                                        store_ex, dataset)
                except FileNotFoundError as e:
                    print(f'  sub-{s}: skip ({e})', flush=True)
                    continue
                rows.append(row)
                for (mdl, mode), r in recs.items():
                    frac = r['n_vox_bin'] / max(r['n_vox'], 1)
                    for b in range(M.N_X_BINS):
                        binned_rows.append(dict(
                            dataset=prefix, subject=s, roi=roi,
                            sample=sample, model=mdl, alignment=mode,
                            x_bin=b, x_norm=M.X_CENTERS[b],
                            measured_z=r['binned'][b], model_z=r['bpred'][b],
                            n_voxel_trials=int(r['counts'][b]),
                            frac_voxels=float(frac[b]), n_vox=r['n_vox']))
                    key = (roi, sample, mdl, mode)
                    points.setdefault(key, []).append(r['pts'])
                if ex is not None and roi != 'wholebrain':
                    cands.append(ex['cand'])
                    example_store[(s, roi)] = ex
                print(f"  sub-{s}: {row['n_vox']:5d} vox  cvR2max="
                      f"{row['wprf_cvr2_max']:.3f} (null p95="
                      f"{row['null_max_p95']:.3f}, p={row['p_max']:.2f})  "
                      f"heldout PRF c-f="
                      f"{M.centre_minus_flank(recs[('prf','heldout')]['binned'], recs[('prf','heldout')]['n_vox_bin'] / max(recs[('prf','heldout')]['n_vox'],1) >= 0.25):+.3f}",
                      flush=True)

    pd.DataFrame(rows).to_csv(datadir / f'{prefix}_voxel_stats.tsv',
                              sep='\t', index=False)
    pd.DataFrame(binned_rows).to_csv(
        datadir / f'{prefix}_normalized_binned.tsv', sep='\t', index=False)
    rng = np.random.default_rng(0)
    packed = {}
    for k, v in points.items():
        arr = np.vstack(v)
        if len(arr) > 60000:
            arr = arr[rng.choice(len(arr), 60000, replace=False)]
        packed['|'.join(k)] = arr.astype(np.float32)
    np.savez_compressed(datadir / f'{prefix}_normalized_points.npz', **packed)

    if cands:
        write_examples(pd.concat(cands, ignore_index=True), example_store,
                       datadir, prefix)
    print(f'saved to {datadir}', flush=True)


def pick_examples(cand, n=8):
    """Best cross-validated voxels that also make a legible panel.

    Same idea as value_prf's `pick_nice`: rank on cross-validated fit, filter on
    legibility (the binned points must follow the curve, the curve must not be a
    scribble, the peak must be supported by trials), and spread the picks over
    subjects so one subject cannot supply the whole figure.
    """
    ok = cand[(cand.bin_r > 0.8) & (cand.n_peaks <= 2)
              & (cand.n_trials_at_peak >= MIN_TRIALS_AT_PEAK)
              & (cand.cvr2_loro > cand.null_p99)]
    if len(ok) == 0:
        ok = cand.copy()
    ok = ok.sort_values('cvr2_loro', ascending=False)
    picks, seen = [], {}
    for _, r in ok.iterrows():
        if seen.get(r.subject, 0) >= 2:
            continue
        picks.append(r)
        seen[r.subject] = seen.get(r.subject, 0) + 1
        if len(picks) == n:
            break
    return pd.DataFrame(picks)


def write_examples(cand, store, datadir, prefix='numerosity'):
    cand.to_csv(datadir / f'{prefix}_example_candidates.tsv', sep='\t',
                index=False)
    picks = pick_examples(cand)
    trials, bins, curves = [], [], []
    for _, r in picks.iterrows():
        ex = store[(r.subject, r.roi)]
        v = int(r.voxel)
        y = ex['B'][:, v]
        meta = dict(dataset=prefix, subject=r.subject, roi=r.roi,
                    voxel=v, preferred_n=r.preferred_n,
                    eff_width=r.eff_width, cvr2_loro=r.cvr2_loro,
                    null_p99=r.null_p99, edf=r.edf)
        for run_i, x_i, y_i in zip(ex['run'], ex['n'], y):
            trials.append(dict(**meta, run=int(run_i), stimulus=float(x_i),
                               measured_beta=float(y_i)))
        cx, m, sem, ns = binned_response(y, ex['n'], value_bin_edges(ex['n']))
        for a, b_, c, d in zip(cx, m, sem, ns):
            bins.append(dict(**meta, stimulus_bin_center=float(a),
                             measured_mean=float(b_), measured_sem=float(c),
                             n_trials_in_bin=int(d)))
        for g, p, lo, hi in zip(ex['grid'], ex['curves'][:, v],
                                ex['fold_lo'][:, v], ex['fold_hi'][:, v]):
            curves.append(dict(**meta, stimulus_grid=float(g),
                               model_prediction=float(p),
                               model_loro_lo=float(lo), model_loro_hi=float(hi)))
    pd.DataFrame(trials).to_csv(datadir / f'{prefix}_example_trials.tsv',
                                sep='\t', index=False)
    pd.DataFrame(bins).to_csv(datadir / f'{prefix}_example_bins.tsv',
                              sep='\t', index=False)
    pd.DataFrame(curves).to_csv(datadir / f'{prefix}_example_curves.tsv',
                                sep='\t', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datadir',
                   default=str(Path.home() / 'git/value_prf/notes/data'))
    p.add_argument('--rois', default=None)
    p.add_argument('--samples', default=None)
    p.add_argument('--n-perm', type=int, default=20)
    p.add_argument('--n-subjects', type=int, default=None)
    p.add_argument('--dataset', default='neural_priors', choices=list(DATASETS))
    a = p.parse_args()
    main(a.datadir, rois=tuple(a.rois.split(',')) if a.rois else None,
         samples=tuple(a.samples.split(',')) if a.samples else None,
         n_perm=a.n_perm, n_subjects=a.n_subjects, dataset=a.dataset)
