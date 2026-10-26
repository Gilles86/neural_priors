"""Model-free summaries of the estimation reports per participant and range condition.

Per participant x condition (trials with a report; censored timeouts included at their
final position, so these are the reports as given):
- slope: OLS of report on n;
- bend: quadratic coefficient of report on x = (n - centre) / half-width (x in [-1, 1]
  in both conditions; items), and bend / half-width (range-normalised);
- s.d. profile: report s.d. per fifth of the range (normalised position) and the slope
  of the per-n s.d. on n (items per item); Weber noise predicts s.d. growing with n;
- mean bias: mean(report - n); also the bias at the range ends and centre;
- compression relative to the range: slope, and the mean report at the range centre
  minus the centre.
Narrow vs wide: paired t-tests and Wilcoxon over participants. Retinonumeral
(5..24, half-width 9.5, centre 14.5) for reference, same definitions, from its trial table.

    ~/mambaforge/envs/bauer/bin/python -m neural_priors.cogmodels.behaviour_model_free
"""
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats

REPO = op.abspath(op.join(op.dirname(__file__), '..', '..'))
OUT = op.join(REPO, 'notes', 'data', 'behaviour_model')
DATA = '/data/ds-neuralpriors/derivatives/behaviour_model/behaviour_trials.tsv'
RN_DATA = '/data/ds-retinonumeral/derivatives/behaviour_model/behaviour_trials.tsv'
RANGES = {'narrow': (10, 25), 'wide': (10, 40), 'retinonumeral': (5, 24)}


def summarise(g, lo, hi):
    n = g.n.to_numpy(float)
    y = g.response.to_numpy(float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    x = (n - c) / h
    b1 = np.polyfit(n, y, 1)
    b2 = np.polyfit(x, y, 2)
    sd_n = g.groupby('n').response.std(ddof=1).dropna()
    fifth = np.minimum(((n - lo) / (hi - lo + 1) * 5).astype(int), 4)
    sd5 = pd.Series(y).groupby(fifth).std(ddof=1)
    out = dict(trials=len(g), slope=b1[0], bend=b2[0], bend_norm=b2[0] / h,
               sd=np.mean(sd_n), sd_slope=np.polyfit(sd_n.index, sd_n.values, 1)[0],
               sd_rel_slope=np.polyfit((sd_n.index - c) / h, sd_n.values, 1)[0] / np.mean(sd_n),
               bias=np.mean(y - n), bias_norm=np.mean(y - n) / h,
               bias_lo=np.mean((y - n)[x < -.6]), bias_hi=np.mean((y - n)[x > .6]),
               bias_centre=np.mean((y - n)[np.abs(x) < .2]))
    for k in range(5):
        out[f'sd_q{k + 1}'] = sd5.get(k, np.nan)
    out['cv'] = np.mean(sd_n.values / sd_n.index.values)
    return pd.Series(out)


def main():
    d = pd.read_csv(DATA, sep='\t', dtype={'subject': str})
    d = d[d.response.notna()]
    S = (d.groupby(['subject', 'range'])
         .apply(lambda g: summarise(g, *RANGES[g.name[1]]), include_groups=False)
         .reset_index())
    S.to_csv(op.join(OUT, 'model_free_by_subject_summary.tsv'), sep='\t', index=False,
             float_format='%.4f')
    rows = []
    stats_cols = [c for c in S.columns if c not in ('subject', 'range', 'trials')]
    W = S.pivot(index='subject', columns='range')
    for k in stats_cols:
        a, b = W[(k, 'narrow')], W[(k, 'wide')]
        t = stats.ttest_rel(b, a)
        rows.append(dict(statistic=k, narrow_mean=a.mean(), narrow_sd=a.std(),
                         wide_mean=b.mean(), wide_sd=b.std(), diff_wide_minus_narrow=(b - a).mean(),
                         diff_ci_lo=(b - a).mean() - 1.96 * (b - a).std() / np.sqrt(len(a)),
                         diff_ci_hi=(b - a).mean() + 1.96 * (b - a).std() / np.sqrt(len(a)),
                         t=t.statistic, p=t.pvalue, p_wilcoxon=stats.wilcoxon(b - a).pvalue,
                         n_wide_greater=int((b > a).sum()), n=len(a)))
    if op.exists(RN_DATA):
        r = pd.read_csv(RN_DATA, sep='\t', dtype={'subject': str})
        R = r.groupby('subject').apply(lambda g: summarise(g, 5, 24), include_groups=False)
        R.reset_index().to_csv(op.join(OUT, 'model_free_retinonumeral_by_subject_summary.tsv'),
                               sep='\t', index=False, float_format='%.4f')
        for row in rows:
            k = row['statistic']
            row.update(retinonumeral_mean=R[k].mean(), retinonumeral_sd=R[k].std())
    T = pd.DataFrame(rows)
    T.to_csv(op.join(OUT, 'model_free_narrow_vs_wide_summary.tsv'), sep='\t', index=False,
             float_format='%.4g')
    pd.set_option('display.width', 250)
    print(T.round(3).to_string())


if __name__ == '__main__':
    main()
