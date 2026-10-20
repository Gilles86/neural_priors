"""
Local stats + figure for the LowRankNoise decoding study
(notes/decoding_lowrank_noise.md). Input: the cluster reduce of
decoding_lowrankcv_collect.py (rsync'ed to notes/data/).

  python decoding_lowrankcv_compare.py --data_dir notes/data \
      --fig figures/prototypes/decoding_lowrank_noise.png
"""
import argparse
import os.path as op
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

METHODS = ['production', 'reference', 'lowrank_T1', 'lowrank']
NARROW_UPPER = 25.0


def subject_metrics(t):
    """One row per (config, subject, method, range, truth)."""
    rows = []
    for (cfg, sub, m, rg), d in t.groupby(['config', 'subject', 'method', 'range']):
        truths = ['truth', 'n'] if cfg == 'responses' else ['truth']
        for tc in truths:
            ok = d['E'].notna() & d[tc].notna()
            dd = d[ok]
            row = dict(config=cfg, subject=sub, method=m, range=rg, truth=tc,
                       n_trials=len(d), n_nan=int((~ok).sum()),
                       r=np.corrcoef(dd['E'], dd[tc])[0, 1] if len(dd) > 2 else np.nan,
                       mae=(dd['E'] - dd[tc]).abs().mean(),
                       post_sd=dd['sd'].mean(),
                       cov50=(dd[f'hpd_{tc}'] < .5).mean(),
                       cov95=(dd[f'hpd_{tc}'] < .95).mean(),
                       pit_sd=dd[f'pit_{tc}'].std(),
                       logscore=dd[f'logscore_{tc}'].replace(-np.inf, np.nan).mean(),
                       logscore_ninf=int(np.isneginf(dd[f'logscore_{tc}']).sum()),
                       frac_flat=dd['flat'].mean())
            # the paper's convention: wide restricted to x <= 25, censored posterior
            if rg == 0.0:
                pe = dd.dropna(subset=['E_paper'])
                row.update(r_paper=pe[['E_paper', tc]].corr().iloc[0, 1],
                           mae_paper=(pe['E_paper'] - pe[tc]).abs().mean(),
                           mad_paper=pe['mad_paper'].mean())
            else:
                pe = dd[dd[tc] <= NARROW_UPPER].dropna(subset=['E_cens'])
                row.update(r_paper=pe[['E_cens', tc]].corr().iloc[0, 1],
                           mae_paper=(pe['E_cens'] - pe[tc]).abs().mean(),
                           mad_paper=pe['mad_cens'].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def paper_stats(t, cfg='responses'):
    """The analysis notebook's MANUSCRIPT STATS block, per method."""
    out = []
    for m, d in t[t.config == cfg].groupby('method'):
        d = d.dropna(subset=['E_paper'])
        res = dict(config=cfg, method=m)
        rm = pg.rm_corr(d, x='E_paper', y='truth', subject='subject')
        res['rmcorr_all'] = rm['r'].iloc[0]
        for rg, lab in [(0.0, 'narrow'), (1.0, 'wide')]:
            dd = d[d['range'] == rg]
            res[f'rmcorr_{lab}'] = pg.rm_corr(dd, x='E_paper', y='truth', subject='subject')['r'].iloc[0]
        nar = d[d['range'] == 0.0].groupby('subject')
        wid = t[(t.config == cfg) & (t.method == m) & (t['range'] == 1.0) & (t['truth'] <= NARROW_UPPER)].dropna(subset=['E_cens']).groupby('subject')
        mae_n = nar.apply(lambda x: (x['E_paper'] - x['truth']).abs().mean())
        mae_w = wid.apply(lambda x: (x['E_cens'] - x['truth']).abs().mean())
        mad_n = nar['mad_paper'].mean()
        mad_w = wid['mad_cens'].mean()
        res.update(mae_narrow=mae_n.mean(), mae_wide25=mae_w.mean(),
                   mae_t=stats.ttest_rel(mae_w, mae_n.loc[mae_w.index]).statistic,
                   mad_narrow=mad_n.mean(), mad_wide25=mad_w.mean(),
                   mad_t=stats.ttest_rel(mad_w, mad_n.loc[mad_w.index]).statistic,
                   mad_p_one=stats.ttest_rel(mad_w, mad_n.loc[mad_w.index], alternative='greater').pvalue)
        out.append(res)
    return pd.DataFrame(out)


def paired(sm, a='reference', b='lowrank', truth='truth'):
    rows = []
    s = sm[sm.truth == truth]
    for cfg in s.config.unique():
        for metric in ['r', 'r_paper', 'mae', 'post_sd', 'cov50', 'cov95', 'pit_sd', 'logscore']:
            w = (s[s.config == cfg].groupby(['subject', 'method'])[metric].mean()
                 .unstack('method'))
            if a not in w or b not in w:
                continue
            w = w[[a, b]].dropna()
            d = w[b] - w[a]
            tt = stats.ttest_rel(w[b], w[a])
            rows.append(dict(config=cfg, truth=truth, metric=metric, n=len(w),
                             **{a: w[a].mean(), b: w[b].mean()}, diff=d.mean(),
                             t=tt.statistic, p=tt.pvalue,
                             p_wilcoxon=stats.wilcoxon(d).pvalue if len(w) > 5 else np.nan,
                             n_improved=int((d > 0).sum())))
    return pd.DataFrame(rows)


def behaviour_table(t):
    """Behavioural variability per subject: SD of responses around per-(n, range) means."""
    b = t[(t.method == 'reference')].drop_duplicates(['config', 'subject', 'session', 'run', 'trial_nr'])
    b = b[b.config == b.config.iloc[0]].dropna(subset=['response'])
    b['resid'] = b['response'] - b.groupby(['subject', 'n', 'range'])['response'].transform('mean')
    beh = b.groupby(['subject', 'range'])['resid'].std().unstack('range')
    beh.columns = ['beh_sd_narrow', 'beh_sd_wide']
    beh['beh_sd'] = b.groupby('subject')['resid'].std()
    return beh


def brain_behaviour(sm, beh):
    rows = []
    s = sm[sm.truth == 'truth']
    for (cfg, m), d in s.groupby(['config', 'method']):
        w = d.groupby('subject')[['r', 'post_sd', 'mae']].mean().join(beh)
        for pred in ['r', 'post_sd', 'mae']:
            rho, p = stats.spearmanr(w[pred], w['beh_sd'], nan_policy='omit')
            rows.append(dict(config=cfg, method=m, predictor=pred, outcome='beh_sd',
                             rho=rho, p=p, n=int(w[[pred, 'beh_sd']].dropna().shape[0])))
        # range effect: Δ decoded SD (wide - narrow) vs Δ behavioural SD
        dsd = d.pivot_table(index='subject', columns='range', values='post_sd')
        dd = (dsd[1.0] - dsd[0.0]).rename('d_post_sd').to_frame().join(beh)
        dd['d_beh'] = dd['beh_sd_wide'] - dd['beh_sd_narrow']
        rho, p = stats.spearmanr(dd['d_post_sd'], dd['d_beh'], nan_policy='omit')
        rows.append(dict(config=cfg, method=m, predictor='d_post_sd', outcome='d_beh_sd',
                         rho=rho, p=p, n=int(dd.dropna().shape[0])))
    return pd.DataFrame(rows)


def trialwise(t, cfg='groundtruth'):
    """Per-subject OLS, group one-sided t (declared tests 3 of notes §1.5)."""
    rows = []
    d0 = t[(t.config == cfg)].dropna(subset=['response', 'E', 'sd'])
    for m, d in d0.groupby('method'):
        b_sd, b_E = [], []
        for sub, x in d.groupby('subject'):
            x = x.copy()
            x['abs_err'] = (x['response'] - x['n']).abs()
            for c in ['n', 'range', 'sd', 'E']:
                x[c] = x[c] - x[c].mean()
            if x['sd'].std() == 0:
                continue
            X = np.c_[np.ones(len(x)), x['n'], x['range'], x['sd']]
            b_sd.append(np.linalg.lstsq(X, x['abs_err'], rcond=None)[0][3] / x['sd'].std())
            X = np.c_[np.ones(len(x)), x['n'], x['range'], x['E']]
            b_E.append(np.linalg.lstsq(X, x['response'], rcond=None)[0][3] * x['E'].std())
        for lab, b in [('abs_err~post_sd (per SD)', b_sd), ('response~E (per SD)', b_E)]:
            b = np.asarray(b)
            tt = stats.ttest_1samp(b, 0, alternative='greater')
            rows.append(dict(config=cfg, method=m, test=lab, n=len(b), mean_beta=b.mean(),
                             t=tt.statistic, p_one=tt.pvalue))
    return pd.DataFrame(rows)


def make_figure(sm, nf, fig_fn):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('paper')
    s = sm[sm.truth == 'truth']
    w = s.groupby(['config', 'subject', 'method'])[['r', 'mae', 'cov95', 'logscore', 'post_sd']].mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    pal = {'responses': '#1f4e79', 'groundtruth': '#c05000'}
    lab = {'responses': 'Paper config (truth = response)', 'groundtruth': 'Ground truth (truth = n)'}
    for ax, metric, title in [(axes[0, 0], 'r', 'Decoding r (posterior mean)'),
                              (axes[0, 1], 'cov95', '95% HDI coverage'),
                              (axes[0, 2], 'logscore', 'Log score vs flat (nats/trial)')]:
        p = w.pivot_table(index=['config', 'subject'], columns='method', values=metric).reset_index()
        for cfg, d in p.groupby('config'):
            ax.scatter(d['reference'], d['lowrank'], s=14, color=pal[cfg], alpha=.8, label=lab[cfg])
        lo = np.nanmin(p[['reference', 'lowrank']].values); hi = np.nanmax(p[['reference', 'lowrank']].values)
        ax.plot([lo, hi], [lo, hi], color='0.6', lw=.8, ls='--')
        if metric == 'cov95':
            ax.axhline(.95, color='0.6', lw=.8); ax.axvline(.95, color='0.6', lw=.8)
        if metric == 'logscore':
            ax.axhline(0, color='0.6', lw=.8); ax.axvline(0, color='0.6', lw=.8)
        ax.set_xlabel('Production noise model (ResidualFitter, spherical)')
        ax.set_ylabel('LowRankNoise (CV k, T)')
        ax.set_title(title, loc='left', fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7)
    # chosen T and k
    ax = axes[1, 0]
    T = nf['temperature'].replace(np.inf, 2 ** 13)
    for cfg, d in nf.groupby('config'):
        ax.hist(np.log2(d['temperature'].replace(np.inf, 2 ** 13)), bins=np.arange(-2, 14, .5),
                histtype='step', color=pal[cfg], lw=1.2, label=lab[cfg])
    ax.set_xlabel('Chosen temperature, log2 T (13 = flat)')
    ax.set_ylabel('Folds')
    ax.set_title('Chosen temperature', loc='left', fontsize=9)
    ax = axes[1, 1]
    kc = nf.groupby(['config', 'k']).size().unstack('config').fillna(0)
    kc.plot.bar(ax=ax, color=[pal[c] for c in kc.columns], legend=False)
    ax.set_xlabel('Chosen rank k'); ax.set_ylabel('Folds')
    ax.set_title('Chosen rank', loc='left', fontsize=9)
    # posterior SD by range
    ax = axes[1, 2]
    s2 = s[s.config == 'responses'].groupby(['subject', 'method', 'range'])['mad_paper'].mean().reset_index()
    s2 = s2[s2.method.isin(['reference', 'lowrank'])]
    s2['Range'] = s2['range'].map({0.0: 'Narrow', 1.0: 'Wide (x ≤ 25)'})
    sns.pointplot(data=s2, x='Range', y='mad_paper', hue='method', ax=ax, dodge=.2,
                  palette={'reference': '0.4', 'lowrank': '#1f4e79'}, errorbar=('ci', 95))
    ax.set_ylabel('Posterior MAD (items)'); ax.set_xlabel('')
    ax.set_title('Decoded uncertainty by prior range', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(fig_fn, dpi=200)
    print('wrote', fig_fn)


def main(data_dir, fig_fn, suffix=''):
    t = pd.read_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_trials.tsv.gz'), sep='\t',
                    dtype={'subject': str})
    nf = pd.read_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_noisefit.tsv'), sep='\t',
                     dtype={'subject': str})
    sm = subject_metrics(t)
    sm.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_subject.tsv'), sep='\t', index=False)
    n_sub = t.groupby('config').subject.nunique()
    print('participants per config:\n', n_sub.to_string())
    if (n_sub < 39).any():
        print('WARNING: fewer than 39 participants in', n_sub[n_sub < 39].to_dict())

    grp = sm.groupby(['config', 'truth', 'method', 'range'])[
        ['r', 'r_paper', 'mae', 'mae_paper', 'post_sd', 'mad_paper', 'cov50', 'cov95', 'pit_sd',
         'logscore', 'logscore_ninf', 'n_nan', 'frac_flat']].mean()
    grp.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_group.tsv'), sep='\t')
    pd.set_option('display.width', 250)
    print(grp.round(3).to_string())

    pr = pd.concat([paired(sm, truth='truth'), paired(sm, truth='n'),
                    paired(sm, a='production', b='reference', truth='truth'),
                    paired(sm, a='reference', b='lowrank_T1', truth='truth')], ignore_index=True)
    pr.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_paired.tsv'), sep='\t', index=False)
    print(pr.round(4).to_string())

    ps = pd.concat([paper_stats(t, 'responses')] +
                   ([paper_stats(t, 'groundtruth')] if 'groundtruth' in t.config.unique() else []))
    ps.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_paperstats.tsv'), sep='\t', index=False)
    print(ps.round(3).to_string())

    # noise model choices
    nfs = nf.groupby('config').agg(k_median=('k', 'median'), k0_frac=('k', lambda x: (x == 0).mean()),
                                   T_median=('temperature', 'median'),
                                   T_inf_frac=('temperature', lambda x: np.isinf(x).mean()),
                                   dof_median=('dof', 'median'), ref_dof_median=('ref_dof', 'median'),
                                   n_vox_median=('n_voxels', 'median'),
                                   fit_s_median=('fit_seconds', 'median'))
    print(nfs.to_string())
    nfs.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_noisefit_summary.tsv'), sep='\t')

    # common-mode vs specific: Δr vs production r and vs chosen k
    s = sm[sm.truth == 'truth'].groupby(['config', 'subject', 'method'])['r'].mean().unstack('method')
    ks = nf.groupby(['config', 'subject'])['k'].mean().rename('k_mean')
    lT = nf.assign(lT=np.log2(nf.temperature.replace(np.inf, 2 ** 13))).groupby(['config', 'subject'])['lT'].mean()
    cm = s.join(ks).join(lT)
    cm['dr'] = cm['lowrank'] - cm['reference']
    rows = []
    for cfg, d in cm.groupby(level='config'):
        for v in ['reference', 'k_mean', 'lT']:
            rho, p = stats.spearmanr(d['dr'], d[v], nan_policy='omit')
            rows.append(dict(config=cfg, dr_vs=v, rho=rho, p=p))
        rows.append(dict(config=cfg, dr_vs='sd_of_dr', rho=d['dr'].std(), p=np.nan))
    cmt = pd.DataFrame(rows)
    print(cmt.round(3).to_string())
    cm.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_dr.tsv'), sep='\t')

    if t.response.notna().any():
        beh = behaviour_table(t)
        bb = brain_behaviour(sm, beh)
        print(bb.round(3).to_string())
        bb.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_brain_behaviour.tsv'), sep='\t', index=False)
        if 'groundtruth' in t.config.unique():
            tw = trialwise(t)
            print(tw.round(4).to_string())
            tw.to_csv(op.join(data_dir, f'decoding_lowrankcv{suffix}_trialwise.tsv'), sep='\t', index=False)
    if fig_fn:
        make_figure(sm, nf, fig_fn)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--fig', default='figures/prototypes/decoding_lowrank_noise.png')
    p.add_argument('--suffix', default='')
    a = p.parse_args()
    main(a.data_dir, a.fig, a.suffix)
