"""Parameter recovery for the Bayesian-observer model with narrow/wide conditions.

simulate  Participant x condition posterior means of a fit (``<tag>.nc``) -> one
          simulated dataset on the SAME trials (intended reports, no censoring) written
          as a trial table the fit script reads (--data), plus the truth table.
collect   True vs recovered (refit posterior means) per parameter and condition, the
          narrow-wide difference per participant, and the group difference (true: mean
          of the participants' true differences; recovered: posterior of the group
          offset, transformed to the natural scale at the group mean).

    PY="PYTHONPATH=<bauer 4f90289>:$PWD ~/mambaforge/envs/bauer/bin/python -m"
    $PY neural_priors.cogmodels.behaviour_model_recovery simulate power_truncated_cens
    # refit on the cluster (job with DATA=<sim tsv>): power_truncated --tag _rec
    $PY neural_priors.cogmodels.behaviour_model_recovery collect power_truncated_cens
"""
import argparse
import os.path as op

import arviz as az
import numpy as np
import pandas as pd

from neural_priors.cogmodels.behaviour_model_fit import DER, load_table, make_model
from neural_priors.cogmodels.observer_model import LEVELS, posterior_by_condition

REPO = op.abspath(op.join(op.dirname(__file__), '..', '..'))
OUT = op.join(REPO, 'notes', 'data', 'behaviour_model')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['simulate', 'collect'])
    ap.add_argument('tag')
    ap.add_argument('--fits-dir', default=op.join(DER, 'fits'))
    ap.add_argument('--data', default=op.join(DER, 'behaviour_trials.tsv'))
    ap.add_argument('--rec-tag', default=None, help='tag of the refit (default <model>_rec)')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out-dir', default=DER, help='where the simulated trials / truth go')
    args = ap.parse_args()
    model_name = args.tag.split('_cens')[0].split('_rec')[0]
    sim_path = op.join(args.out_dir, f'recovery_{args.tag}_trials.tsv')
    truth_path = op.join(args.out_dir, f'recovery_{args.tag}_truth.tsv')

    if args.mode == 'simulate':
        idata = az.from_netcdf(op.join(args.fits_dir, f'{args.tag}.nc'))
        data = load_table(args.data, censor=False)
        cp = idata.attrs.get('condition_params', '').split()
        model = make_model(model_name, data, cp)
        truth = posterior_by_condition(idata, model)
        y = model.simulate_responses(data, truth, n_samples=1, rng=args.seed)[:, 0]
        T = pd.read_csv(args.data, sep='\t', dtype={'subject': str})
        T = T[T.response.notna()].set_index(['subject', 'session', 'run', 'trial_nr']).loc[data.index]
        T['response'] = y.astype(int)
        T['censor'] = 0
        T.reset_index().to_csv(sim_path, sep='\t', index=False)
        truth.to_csv(truth_path, sep='\t')
        print(truth.groupby(level='range').describe().T.round(3).to_string())
        print('wrote', sim_path)
        return

    truth = pd.read_csv(truth_path, sep='\t', dtype={'subject': str}).set_index(['subject', 'range'])
    rec_tag = args.rec_tag or f'{model_name}_rec'
    idata = az.from_netcdf(op.join(args.fits_dir, f'{rec_tag}.nc'))
    data = load_table(sim_path, censor=False)
    model = make_model(model_name, data, idata.attrs.get('condition_params', '').split())
    rec = posterior_by_condition(idata, model).loc[truth.index]
    rows = []
    for p in truth.columns:
        for c in LEVELS:
            t, r = truth[p].xs(c, level='range'), rec[p].xs(c, level='range')
            rows.append(dict(parameter=p, range=c, r=np.corrcoef(t, r)[0, 1],
                             slope=np.polyfit(t, r, 1)[0], mean_true=t.mean(),
                             mean_recovered=r.mean()))
        dt = truth[p].xs('wide', level='range') - truth[p].xs('narrow', level='range')
        dr = rec[p].xs('wide', level='range') - rec[p].xs('narrow', level='range')
        row = dict(parameter=p, range='wide_minus_narrow', mean_true=dt.mean(),
                   mean_recovered=dr.mean())
        if dt.std() > 1e-9:
            row.update(r=np.corrcoef(dt, dr)[0, 1], slope=np.polyfit(dt, dr, 1)[0])
        k = f'{p}_by_condition'
        if k in idata.posterior:
            g = idata.posterior[k].mean('subject')
            gd = (g.sel(condition='wide') - g.sel(condition='narrow')).values.ravel()
            row.update(group_delta_rec_lo=np.percentile(gd, 2.5),
                       group_delta_rec_hi=np.percentile(gd, 97.5),
                       group_delta_true_in_95=bool(np.percentile(gd, 2.5) <= dt.mean()
                                                   <= np.percentile(gd, 97.5)))
        rows.append(row)
    R = pd.DataFrame(rows)
    # cross-parameter matrix (pooled over conditions)
    cross = pd.DataFrame({a: {b: np.corrcoef(truth[a], rec[b])[0, 1] for b in truth.columns}
                          for a in truth.columns})
    cross.index.name = 'recovered'
    R.to_csv(op.join(OUT, f'recovery_{args.tag}_summary.tsv'), sep='\t', index=False,
             float_format='%.4f')
    cross.to_csv(op.join(OUT, f'recovery_{args.tag}_cross_summary.tsv'), sep='\t',
                 float_format='%.3f')
    pd.concat([truth.add_suffix('_true'), rec.add_suffix('_recovered')], axis=1).to_csv(
        op.join(OUT, f'recovery_{args.tag}_subjects.tsv'), sep='\t', float_format='%.4f')
    pd.set_option('display.width', 220)
    print(R.round(3).to_string())
    print('cross (columns true, rows recovered):\n', cross.round(2).to_string())
    ss = idata.sample_stats
    print('refit divergences', int(ss.diverging.sum()))


if __name__ == '__main__':
    main()
