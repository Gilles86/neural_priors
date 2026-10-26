"""Posterior predictive checks of the Bayesian-observer model, per range condition.

Each fit's 200 posterior-predictive datasets (``<tag>_ppc.parquet``: one simulated
report per trial per posterior draw, same trials) are summarised exactly like the data:
every statistic is computed per simulated dataset first, then the 95% band is taken
across datasets. Censored trials are compared like with like: the replicate is
censored at the data's own censoring point (min(rep, final) when the slider was still
moving up, max(rep, final) when moving down).

Figure 1 (group, per condition): mean report vs n, report s.d. vs n, response-option
histogram (slider ends), per-participant slope, bend and residual s.d. observed vs
predicted. Figure 2: every participant, mean report and s.d. per fifth of each range.

    PYTHONPATH=$HOME/git/bauer-estimation:$PWD ~/mambaforge/envs/bauer/bin/python -m \\
        neural_priors.cogmodels.behaviour_model_ppc power_truncated_cens

Outputs (git-ignored): notes/figures/behaviour_model/ppc_<tag>{,_subjects}.{pdf,png},
notes/data/behaviour_model/ppc_<tag>_summary.tsv, ppc_<tag>_subjects_summary.tsv.
"""
import argparse
import os.path as op

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_priors.cogmodels.behaviour_model_fit import DER, load_table
from neural_priors.cogmodels.observer_model import RANGES, LEVELS

REPO = op.abspath(op.join(op.dirname(__file__), '..', '..'))
FIG = op.join(REPO, 'notes', 'figures', 'behaviour_model')
OUT = op.join(REPO, 'notes', 'data', 'behaviour_model')
INK = '#2b2b2b'
COL = {'narrow': '#3B5BA5', 'wide': '#C8873A'}
FLAG = '#C44E52'
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .6, 'pdf.fonttype': 42,
                     'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial']})


def fifth(n, lo, hi):
    return np.minimum(((n - lo) / (hi - lo + 1) * 5).astype(int), 4)


class Stats:
    """Statistics of one condition; X is (n_trials, k) (k = 1 for the data)."""

    def __init__(self, D, cond):
        self.lo, self.hi = RANGES[cond]
        self.subj = D.index.get_level_values('subject').to_numpy()
        self.subjects = np.unique(self.subj)
        self.n = D['n'].to_numpy(float)
        self.ns = np.arange(self.lo, self.hi + 1)
        self.bin = fifth(self.n, self.lo, self.hi)
        self.x = (self.n - (self.lo + self.hi) / 2) / ((self.hi - self.lo) / 2)

    def _cell(self, X, key, values, fun):
        return np.array([[fun(X[(self.subj == s) & (key == v)]) for v in values]
                         for s in self.subjects])                         # (S, V, k)

    def all(self, X):
        sd = lambda x: x.std(0, ddof=1) if len(x) > 1 else np.full(x.shape[1], np.nan)
        mean_n = self._cell(X, self.n, self.ns, lambda x: x.mean(0) if len(x) else
                            np.full(X.shape[1], np.nan))
        sd_n = self._cell(X, self.n, self.ns, sd)
        slope, bend, sdres = [], [], []
        for s in self.subjects:
            m = self.subj == s
            A = np.column_stack([np.ones(m.sum()), self.n[m]])
            b = np.linalg.lstsq(A, X[m], rcond=None)[0]
            slope.append(b[1])
            sdres.append((X[m] - A @ b).std(0, ddof=1))
            A2 = np.column_stack([np.ones(m.sum()), self.x[m], self.x[m] ** 2])
            bend.append(np.linalg.lstsq(A2, X[m], rcond=None)[0][2])
        opts = np.arange(self.lo, self.hi + 1)
        return dict(mean_n=np.nanmean(mean_n, 0), sd_n=np.nanmean(sd_n, 0),
                    mean_bin_s=self._cell(X, self.bin, range(5), lambda x: x.mean(0)),
                    sd_bin_s=self._cell(X, self.bin, range(5), sd),
                    slope=np.array(slope), bend=np.array(bend), sdres=np.array(sdres),
                    hist=np.stack([(X == o).mean(0) for o in opts]))


def band(a):
    return np.percentile(a, 2.5, -1), np.median(a, -1), np.percentile(a, 97.5, -1)


def coverage(obs, a):
    lo, _, hi = band(a)
    o = obs[..., 0]
    return float(((o >= lo) & (o <= hi)).mean())


def replicate_coverage(a):
    lo, _, hi = band(a)
    return float(((a >= lo[..., None]) & (a <= hi[..., None])).mean())


def mahalanobis(obs, a):
    mu = a.mean(-1)
    Ci = np.linalg.pinv(np.cov(a) + 1e-9 * np.eye(a.shape[0]))
    d = lambda v: float((v - mu) @ Ci @ (v - mu))
    d_obs = d(obs[:, 0])
    d_rep = np.array([d(a[:, k]) for k in range(a.shape[1])])
    return d_obs, float(np.mean(d_rep >= d_obs)), float(np.percentile(d_rep, 95))


def load_ppc(tag, D, fits_dir):
    p = pd.read_parquet(op.join(fits_dir, f'{tag}_ppc.parquet'))
    p['subject'] = p['subject'].astype(str)
    keys = ['subject', 'session', 'run', 'trial_nr']
    W = p.pivot(index=keys, columns='ppc_sample', values='simulated_response').loc[D.index]
    assert not W.isna().any().any()
    W = W.to_numpy(float)
    if 'censor' in D:
        c = D['censor'].to_numpy()[:, None]
        fin = D['response'].to_numpy(float)[:, None]
        W = np.where(c > 0, np.minimum(W, fin), W)
        W = np.where(c < 0, np.maximum(W, fin), W)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag', help='fit tag, e.g. power_truncated_cens')
    ap.add_argument('--fits-dir', default=op.join(DER, 'fits'))
    ap.add_argument('--data', default=op.join(DER, 'behaviour_trials.tsv'))
    args = ap.parse_args()
    censor = '_cens' in args.tag
    D_all = load_table(args.data, censor=censor)
    W_all = load_ppc(args.tag, D_all, args.fits_dir)
    obs, sim, st = {}, {}, {}
    rows = []
    for c in LEVELS:
        m = (D_all['range'] == c).to_numpy()
        D = D_all[m]
        st[c] = Stats(D, c)
        obs[c] = st[c].all(D['response'].to_numpy(float)[:, None])
        sim[c] = st[c].all(W_all[m])
        for k in ('mean_n', 'sd_n', 'slope', 'bend', 'sdres', 'mean_bin_s', 'sd_bin_s', 'hist'):
            rows.append(dict(range=c, statistic=k, coverage95=coverage(obs[c][k], sim[c][k]),
                             replicate_coverage95=replicate_coverage(sim[c][k])))
        for k in ('mean_n', 'sd_n', 'bend'):
            d_obs, p, d95 = mahalanobis(obs[c][k], sim[c][k])
            rows.append(dict(range=c, statistic=f'mahalanobis_{k}', observed=d_obs,
                             pred_95=d95, ppp=p))
        for k in ('slope', 'bend', 'sdres'):
            lo, med, hi = band(sim[c][k])
            o = obs[c][k][:, 0]
            rows.append(dict(range=c, statistic=f'{k}_group_mean', observed=o.mean(),
                             pred_median=np.median(sim[c][k].mean(0)),
                             pred_lo=np.percentile(sim[c][k].mean(0), 2.5),
                             pred_hi=np.percentile(sim[c][k].mean(0), 97.5),
                             r_obs_pred=np.corrcoef(o, med)[0, 1],
                             n_inside=int(((o >= lo) & (o <= hi)).sum()), n=len(o)))
        sd_flat = sim[c]['sd_n']
        rows.append(dict(range=c, statistic='sd_n_range_obs_vs_pred',
                         observed=np.ptp(obs[c]['sd_n'][:, 0]), pred_median=np.median(np.ptp(sd_flat, 0))))
    R = pd.DataFrame(rows)
    R.to_csv(op.join(OUT, f'ppc_{args.tag}_summary.tsv'), sep='\t', index=False, float_format='%.4f')
    pd.set_option('display.width', 220)
    print(R.round(3).to_string())

    # ---------------- figure 1: group ----------------
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 4.3), constrained_layout=True)
    a = axes[0, 0]
    a.plot([10, 40], [10, 40], color='0.8', lw=.6, ls='--', zorder=0)
    for c in LEVELS:
        x = st[c].ns
        lo, med, hi = band(sim[c]['mean_n'])
        a.fill_between(x, lo, hi, color=COL[c], alpha=.25, lw=0)
        a.plot(x, med, color=COL[c], lw=1)
        a.plot(x, obs[c]['mean_n'][:, 0], 'o', color=COL[c], mec=INK, mew=.3, ms=2.6)
    a.set(xlabel='Presented number', ylabel='Mean report', title='Mean report')
    a = axes[0, 1]
    for c in LEVELS:
        x = st[c].ns
        lo, med, hi = band(sim[c]['sd_n'])
        a.fill_between(x, lo, hi, color=COL[c], alpha=.25, lw=0)
        a.plot(x, med, color=COL[c], lw=1)
        a.plot(x, obs[c]['sd_n'][:, 0], 'o', color=COL[c], mec=INK, mew=.3, ms=2.6)
    a.set(xlabel='Presented number', ylabel='Report s.d. (items)', title='Report s.d.', ylim=(0, None))
    a = axes[0, 2]
    for c in LEVELS:
        x = np.arange(*RANGES[c]) if False else np.arange(RANGES[c][0], RANGES[c][1] + 1)
        lo, med, hi = band(sim[c]['hist'])
        a.fill_between(x, lo, hi, color=COL[c], alpha=.25, lw=0, step='mid')
        a.step(x, med, where='mid', color=COL[c], lw=.8)
        a.plot(x, obs[c]['hist'][:, 0], 'o', color=COL[c], mec=INK, mew=.3, ms=2.2)
    a.set(xlabel='Report', ylabel='Proportion of trials', title='Report distribution')
    for j, (k, lab) in enumerate((('slope', 'Slope'), ('bend', 'Bend (items)'),
                                  ('sdres', 'Residual s.d. (items)'))):
        a = axes[1, j]
        allv = np.concatenate([obs[c][k][:, 0] for c in LEVELS] + [band(sim[c][k])[1] for c in LEVELS])
        lims = (allv.min() - .1 * np.ptp(allv), allv.max() + .1 * np.ptp(allv))
        a.plot(lims, lims, color='0.75', lw=.6, ls='--', zorder=0)
        txt = []
        for c in LEVELS:
            lo, med, hi = band(sim[c][k])
            o = obs[c][k][:, 0]
            a.vlines(o, lo, hi, color=COL[c], lw=.6, alpha=.6)
            a.plot(o, med, 'o', color=COL[c], ms=2.6, mew=0)
            txt.append(f'{c.capitalize()}: r = {np.corrcoef(o, med)[0, 1]:.2f}, '
                       f'{int(((o >= lo) & (o <= hi)).sum())}/{len(o)} inside')
        a.set(xlim=lims, ylim=lims, xlabel=f'{lab}, observed', ylabel=f'{lab}, predicted')
        a.text(.03, .97, '\n'.join(txt), transform=a.transAxes, va='top', fontsize=5.8, color='0.3')
    for c, y in zip(LEVELS, (.95, .87)):
        axes[0, 0].text(.05, y, f'{c.capitalize()} ({RANGES[c][0]}–{RANGES[c][1]})',
                        color=COL[c], transform=axes[0, 0].transAxes, fontsize=6.5, va='top')
    for c in LEVELS:
        cov_m = R.query('range == @c and statistic == "mean_n"').coverage95.iloc[0]
        cov_s = R.query('range == @c and statistic == "sd_n"').coverage95.iloc[0]
        pm_ = R.query('range == @c and statistic == "mahalanobis_mean_n"').ppp.iloc[0]
        ps_ = R.query('range == @c and statistic == "mahalanobis_sd_n"').ppp.iloc[0]
        y = .2 if c == 'narrow' else .08
        axes[0, 0].text(.97, y, f'{cov_m:.0%} in band, joint p = {pm_:.2f}', color=COL[c],
                        transform=axes[0, 0].transAxes, ha='right', fontsize=5.8)
        axes[0, 1].text(.97, y, f'{cov_s:.0%} in band, joint p = {ps_:.2f}', color=COL[c],
                        transform=axes[0, 1].transAxes, ha='right', fontsize=5.8)
    fig.suptitle(f'Posterior predictive check ({args.tag}); dots data, band 95% of 200 '
                 'simulated datasets', fontsize=7.5)
    for ext in ('pdf', 'png'):
        fig.savefig(op.join(FIG, f'ppc_{args.tag}.{ext}'), dpi=200)
    plt.close(fig)

    # ---------------- figure 2: every participant ----------------
    post = az.from_netcdf(op.join(args.fits_dir, f'{args.tag}.nc')).posterior
    subjects = st['narrow'].subjects
    srows = []
    for i, s in enumerate(subjects):
        out = 0
        for c in LEVELS:
            for k in ('mean_bin_s', 'sd_bin_s'):
                lo, _, hi = band(sim[c][k][i])
                o = obs[c][k][i, :, 0]
                out += int(((o < lo) | (o > hi)).sum())
        srows.append(dict(subject=s, points_outside=out, n_points=20,
                          slope_narrow=obs['narrow']['slope'][i, 0],
                          slope_wide=obs['wide']['slope'][i, 0]))
    S = pd.DataFrame(srows)
    S.to_csv(op.join(OUT, f'ppc_{args.tag}_subjects_summary.tsv'), sep='\t', index=False,
             float_format='%.4f')
    print('participant points outside:', S.points_outside.sum(), 'of', S.n_points.sum(),
          f'({S.points_outside.sum() / S.n_points.sum():.1%}); flagged (>= 3 of 20):',
          list(S.subject[S.points_outside >= 3]))

    def par(name, s, c):
        k = f'{name}_by_condition'
        if k in post:
            return float(post[k].sel(subject=s, condition=c).mean())
        return float(post[name].sel(subject=s).mean()) if name in post else np.nan

    ncol = 5
    nrow = int(np.ceil((len(subjects) + 1) / ncol))
    fig = plt.figure(figsize=(6.85, 9.6))
    gs = fig.add_gridspec(nrow, ncol, left=.07, right=.99, top=.955, bottom=.04, hspace=.45,
                          wspace=.34)
    for j, s in enumerate(subjects):
        i = j
        sub = gs[j // ncol, j % ncol].subgridspec(2, 1, hspace=.1)
        a1, a2 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        a1.plot([10, 40], [10, 40], color='0.85', lw=.5, ls='--', zorder=0)
        for c in LEVELS:
            lo_, hi_ = RANGES[c]
            xb = np.array([st[c].ns[fifth(st[c].ns, lo_, hi_) == b].mean() for b in range(5)])
            for a, k in ((a1, 'mean_bin_s'), (a2, 'sd_bin_s')):
                lo, med, hi = band(sim[c][k][i])
                a.fill_between(xb, lo, hi, color=COL[c], alpha=.25, lw=0)
                a.plot(xb, med, color=COL[c], lw=.8)
                o = obs[c][k][i, :, 0]
                outside = (o < lo) | (o > hi)
                a.plot(xb[~outside], o[~outside], 'o', color=COL[c], mec=INK, mew=.25, ms=2.3)
                a.plot(xb[outside], o[outside], 'o', mfc='white', mec=FLAG, ms=2.8, mew=.8)
        for a in (a1, a2):
            a.set_xlim(8, 42)
            a.set_xticks([10, 25, 40])
        a1.set_ylim(8, 42)
        a1.set_yticks([10, 25, 40])
        a1.set_xticklabels([])
        a2.set_ylim(0, 7.5)
        a2.set_yticks([0, 3, 6])
        r = S.iloc[i]
        a1.set_title(f'sub-{s}', fontsize=6.8, pad=2, color=FLAG if r.points_outside >= 3 else INK)
        lines = []
        for c in LEVELS:
            al = par('alpha', s, c)
            lines.append(f"{c[0].upper()}: {'α %.2f ' % al if np.isfinite(al) else ''}"
                         f"ν {par('nu', s, c):.2f} σr {par('sigma_r', s, c):.1f}")
        a1.text(.03, .97, '\n'.join(lines), transform=a1.transAxes, fontsize=5.2, va='top',
                color='0.3', linespacing=1.1)
        if j % ncol == 0:
            a1.set_ylabel('Mean report')
            a2.set_ylabel('Report s.d.')
        if j + ncol >= len(subjects):
            a2.set_xlabel('Presented number')
    ax = fig.add_subplot(gs[nrow - 1, ncol - 1])
    ax.axis('off')
    tf = ax.transAxes
    for y, c in ((.95, 'narrow'), (.83, 'wide')):
        ax.plot([.05], [y], 'o', color=COL[c], mec=INK, mew=.25, ms=3, transform=tf)
        ax.text(.14, y, f'{c.capitalize()} ({RANGES[c][0]}–{RANGES[c][1]}), fifths of the range',
                fontsize=6, va='center', transform=tf)
    ax.plot([.05], [.71], 'o', mfc='white', mec=FLAG, ms=3.2, mew=.9, transform=tf)
    ax.text(.14, .71, 'Outside the 95% band', fontsize=6, va='center', transform=tf)
    ax.text(0, .58, 'Band: 95% of 200 simulated\ndatasets. Red name: ≥ 3 of 20\npoints outside '
            '(chance ~1).\nPosterior means per condition\n(N narrow, W wide): α exponent,\n'
            'ν noise (log units at 12),\nσr response noise (items)', fontsize=5.8, va='top',
            color='0.3', transform=tf)
    fig.suptitle(f'Every participant, data vs model ({args.tag})', fontsize=7.5, y=.99)
    for ext in ('pdf', 'png'):
        fig.savefig(op.join(FIG, f'ppc_{args.tag}_subjects.{ext}'), dpi=200)
    print('wrote', op.join(FIG, f'ppc_{args.tag}.pdf'), op.join(FIG, f'ppc_{args.tag}_subjects.pdf'))


if __name__ == '__main__':
    main()
