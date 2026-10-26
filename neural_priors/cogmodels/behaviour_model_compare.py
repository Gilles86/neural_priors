"""Model comparison, parameters per range condition, narrow-wide differences, and the
comparison with the retinonumeral fit of the same observer model.

1. Convergence gate per fit (r_hat <= 1.01, bulk/tail ESS >= 400, divergences) and LOO
   (az.compare) of the encodings (alpha free / log / linear; plus alpha shared across
   conditions).
2. Winner: group-level parameters per condition on the natural scale,
   g(mu +- delta/2) for c = narrow / wide (g = softplus or identity), with 95% HDIs, and
   the wide-minus-narrow difference (HDI, P(> 0)); participant means per condition.
3. Range-relative prior: centre position (mu_p - lo) / (hi - lo), width in items
   (prior_sd * mu_p; prior_sd is in log units at the centre) and relative to the range
   width (hi - lo).
4. Retinonumeral (power_truncated+motor_cens, 5..24) through the same summaries.

    PYTHONPATH=<bauer 4f90289>:$PWD ~/mambaforge/envs/bauer/bin/python -m \\
        neural_priors.cogmodels.behaviour_model_compare
Outputs: notes/data/behaviour_model/{convergence,loo_compare,parameters_by_condition,
parameters_retinonumeral_vs_neuralpriors,participant_parameters}_summary.tsv
"""
import argparse
import glob
import json
import os.path as op
import pickle

import arviz as az
import numpy as np
import pandas as pd

from neural_priors.cogmodels.behaviour_model_fit import DER
from neural_priors.cogmodels.observer_model import RANGES, LEVELS

REPO = op.abspath(op.join(op.dirname(__file__), '..', '..'))
OUT = op.join(REPO, 'notes', 'data', 'behaviour_model')
RN_FIT = ('/data/ds-retinonumeral/derivatives/behaviour_model/fits_v2/'
          'power_truncated+motor_cens.nc')
RN_RANGE = (5, 24)
PARS = ['alpha', 'nu', 'prior_mu', 'prior_sd', 'sigma_r', 'p_lapse']
SOFTPLUS = {'nu', 'prior_mu', 'prior_sd', 'sigma_r'}


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def hdi(x):
    return az.hdi(np.asarray(x).ravel(), hdi_prob=.95)


def group_draws(post, p):
    """Group-level natural-scale draws per condition: dict narrow/wide -> (draws,)."""
    if p == 'p_lapse':
        v = post['p_lapse_mu'].values.ravel() if 'p_lapse_mu' in post else \
            post['p_lapse'].mean('subject').values.ravel()
        return {c: v for c in LEVELS}
    base = post[f'{p}_mu_untransformed' if p in SOFTPLUS else f'{p}_mu'].values.ravel() \
        if (f'{p}_mu_untransformed' in post or f'{p}_mu' in post) else None
    if base is None:
        return None
    d = post[f'{p}_cond_mu'].values.ravel() if f'{p}_cond_mu' in post else 0.0
    g = softplus if p in SOFTPLUS else (lambda x: x)
    return {'narrow': g(base + d / 2), 'wide': g(base - d / 2)}


def derived(draws, lo, hi):
    """Range-relative prior summaries from group draws of prior_mu / prior_sd."""
    mu, sd = draws['prior_mu'], draws['prior_sd']
    return {'prior_centre_rel': (mu - lo) / (hi - lo), 'prior_width_items': sd * mu,
            'prior_width_rel': sd * mu / (hi - lo)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fits-dir', default=op.join(DER, 'fits'))
    ap.add_argument('--winner', default='power_truncated_cens')
    args = ap.parse_args()
    pd.set_option('display.width', 250)

    # 1. convergence + LOO
    conv, loos = [], {}
    for js in sorted(glob.glob(op.join(args.fits_dir, '*_summary.json'))):
        r = json.load(open(js))
        if '_rec' in r['model']:
            continue
        r['gate_pass'] = (r['max_rhat'] <= 1.01 and r['min_ess_bulk'] >= 400
                          and r['min_ess_tail'] >= 400 and r['divergences'] == 0)
        conv.append(r)
        lp = op.join(args.fits_dir, f"{r['model']}_loo.pkl")
        if op.exists(lp):
            loos[r['model']] = pickle.load(open(lp, 'rb'))
    C = pd.DataFrame(conv)
    C.to_csv(op.join(OUT, 'convergence_summary.tsv'), sep='\t', index=False, float_format='%.4g')
    print(C[['model', 'divergences', 'max_rhat', 'min_ess_bulk', 'min_ess_tail',
             'worst_rhat_param', 'frac_maxdepth', 'sample_minutes', 'elpd_loo', 'p_loo',
             'n_pareto_k_gt_07', 'gate_pass']].round(3).to_string())
    if len(loos) > 1:
        cmp_ = az.compare(loos, ic='loo', method='stacking')
        cmp_.to_csv(op.join(OUT, 'loo_compare_summary.tsv'), sep='\t', float_format='%.4f')
        print(cmp_[['rank', 'elpd_loo', 'p_loo', 'elpd_diff', 'dse', 'weight']].round(2).to_string())

    # 2. winner parameters per condition
    post = az.from_netcdf(op.join(args.fits_dir, f'{args.winner}.nc')).posterior
    rows, gd = [], {}
    for p in PARS:
        g = group_draws(post, p)
        if g is None:
            continue
        gd[p] = g
        for c in LEVELS:
            h = hdi(g[c])
            rows.append(dict(parameter=p, range=c, group_mean=g[c].mean(), hdi_lo=h[0], hdi_hi=h[1]))
        if p != 'p_lapse':
            dlt = g['wide'] - g['narrow']
            h = hdi(dlt)
            rows.append(dict(parameter=p, range='wide_minus_narrow', group_mean=dlt.mean(),
                             hdi_lo=h[0], hdi_hi=h[1], p_gt_0=float((dlt > 0).mean())))
        k = f'{p}_by_condition'
        for c in LEVELS:
            v = (post[k].sel(condition=c) if k in post else post[p]).mean(('chain', 'draw'))
            rows[[i for i, r in enumerate(rows) if r['parameter'] == p and r['range'] == c][0]] \
                .update(participant_mean=float(v.mean()), participant_sd=float(v.std()),
                        participant_min=float(v.min()), participant_max=float(v.max()))
        if k in post:
            dv = (post[k].sel(condition='wide') - post[k].sel(condition='narrow'))
            lo_ = dv.quantile(.025, ('chain', 'draw'))
            hi_ = dv.quantile(.975, ('chain', 'draw'))
            rows[-1 if p != 'p_lapse' else -1].update(
                n_participants_delta_gt0=int((lo_ > 0).sum()),
                n_participants_delta_lt0=int((hi_ < 0).sum()))
    for c in LEVELS:
        dd = derived({k: gd[k][c] for k in ('prior_mu', 'prior_sd')}, *RANGES[c])
        for k, v in dd.items():
            h = hdi(v)
            rows.append(dict(parameter=k, range=c, group_mean=v.mean(), hdi_lo=h[0], hdi_hi=h[1]))
    dn = derived({k: gd[k]['narrow'] for k in ('prior_mu', 'prior_sd')}, *RANGES['narrow'])
    dw = derived({k: gd[k]['wide'] for k in ('prior_mu', 'prior_sd')}, *RANGES['wide'])
    for k in dn:
        v = dw[k] - dn[k]
        h = hdi(v)
        rows.append(dict(parameter=k, range='wide_minus_narrow', group_mean=v.mean(),
                         hdi_lo=h[0], hdi_hi=h[1], p_gt_0=float((v > 0).mean())))
    if 'alpha' in gd:
        for c in LEVELS:
            v = (post['alpha_by_condition'].sel(condition=c) if 'alpha_by_condition' in post
                 else post['alpha']).mean(('chain', 'draw'))
            lo_ = (post['alpha_by_condition'].sel(condition=c) if 'alpha_by_condition' in post
                   else post['alpha']).quantile(.025, ('chain', 'draw'))
            hi_ = (post['alpha_by_condition'].sel(condition=c) if 'alpha_by_condition' in post
                   else post['alpha']).quantile(.975, ('chain', 'draw'))
            rows.append(dict(parameter='alpha_participant_intervals', range=c,
                             n_excl_log=int(((lo_ > 0) | (hi_ < 0)).sum()),
                             n_excl_linear=int(((lo_ > 1) | (hi_ < 1)).sum()),
                             n_below_0=int((hi_ < 0).sum())))
    P = pd.DataFrame(rows)
    P.to_csv(op.join(OUT, 'parameters_by_condition_summary.tsv'), sep='\t', index=False,
             float_format='%.4f')
    print(P.round(3).to_string())

    # participant table
    sub = {}
    for p in PARS:
        k = f'{p}_by_condition'
        if k in post:
            for c in LEVELS:
                sub[f'{p}_{c}'] = post[k].sel(condition=c).mean(('chain', 'draw')).to_series()
        elif p in post:
            sub[p] = post[p].mean(('chain', 'draw')).to_series()
    S = pd.DataFrame(sub)
    S.to_csv(op.join(OUT, f'participant_parameters_{args.winner}_summary.tsv'), sep='\t',
             float_format='%.4f')
    # within-participant posterior correlations (pooled draws, averaged over participants)
    cors = {}
    names = [n for n in ('alpha', 'nu', 'prior_mu', 'prior_sd', 'sigma_r')
             if f'{n}_by_condition' in post]
    for c in LEVELS:
        X = {n: post[f'{n}_by_condition'].sel(condition=c).stack(s=('chain', 'draw')).values
             for n in names}                                   # (subject, draws)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cors[(c, a, b)] = np.mean([np.corrcoef(X[a][j], X[b][j])[0, 1]
                                           for j in range(X[a].shape[0])])
    Cr = pd.Series(cors).rename('mean_within_participant_r')
    Cr.to_csv(op.join(OUT, f'posterior_correlations_{args.winner}_summary.tsv'), sep='\t',
              float_format='%.3f')
    print(Cr.round(2).to_string())

    # 4. retinonumeral
    if op.exists(RN_FIT):
        rn = az.from_netcdf(RN_FIT).posterior
        rows = []
        rd = {}
        for p in PARS:
            if p == 'p_lapse':
                v = rn['p_lapse_mu'].values.ravel() if 'p_lapse_mu' in rn else \
                    rn['p_lapse'].mean('subject').values.ravel()
            else:
                key = f'{p}_mu_untransformed' if p in SOFTPLUS else f'{p}_mu'
                v = rn[key].values.ravel()
                v = softplus(v) if p in SOFTPLUS else v
            rd[p] = v
        rd.update(derived(rd, *RN_RANGE))
        for p, v in rd.items():
            h = hdi(v)
            part = rn[p].mean(('chain', 'draw')) if p in rn else None
            row = dict(parameter=p, experiment='retinonumeral (5-24)', group_mean=v.mean(),
                       hdi_lo=h[0], hdi_hi=h[1])
            if part is not None:
                row.update(participant_mean=float(part.mean()), participant_sd=float(part.std()))
            rows.append(row)
            for c in LEVELS:
                src = gd.get(p, {}).get(c) if p in gd else \
                    derived({k: gd[k][c] for k in ('prior_mu', 'prior_sd')}, *RANGES[c]).get(p)
                if src is None:
                    continue
                h = hdi(src)
                dd = src - np.random.default_rng(0).permutation(np.resize(v, src.shape))
                hd = hdi(dd)
                rows.append(dict(parameter=p, experiment=f'neural_priors {c} {RANGES[c]}',
                                 group_mean=src.mean(), hdi_lo=h[0], hdi_hi=h[1],
                                 diff_vs_retinonumeral=dd.mean(), diff_hdi_lo=hd[0],
                                 diff_hdi_hi=hd[1]))
        R = pd.DataFrame(rows)
        R.to_csv(op.join(OUT, 'parameters_retinonumeral_vs_neuralpriors_summary.tsv'), sep='\t',
                 index=False, float_format='%.4f')
        print(R.round(3).to_string())


if __name__ == '__main__':
    main()
