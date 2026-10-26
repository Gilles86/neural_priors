"""Fit one hierarchical Bayesian-observer model of the estimation reports (bauer).

Model: ``RangeEstimationModel`` (cogmodels/observer_model.py), an extension of
``bauer.models.EstimationTaskModel`` (bauer branch ``estimation-model`` at 4f90289,
read-only on PYTHONPATH) with condition-specific prior support, slider ends and lapse
range for the narrow (10..25) and wide (10..40) conditions, and hierarchical
narrow-minus-wide offsets on ``--condition-params``. Recipe as in retinonumeral
(notes/behaviour_bayesian_observer.md there): discrete truncated Gaussian prior on the
internal scale, posterior-mean readout, Gaussian motor noise, lapse, censoring of
unfinished timeout reports; nutpie, 4 chains x 1500 draws (1500 tune), target_accept .95.

Models: power_truncated (alpha free), log_truncated, linear_truncated.

Writes to ``--out-dir``: <tag>.nc, <tag>_loo.pkl, <tag>_summary.json, <tag>_ppc.parquet.

    PYTHONPATH=$HOME/git/bauer-estimation:$PWD ~/mambaforge/envs/bauer/bin/python -m \\
        neural_priors.cogmodels.behaviour_model_fit power_truncated --censor
Cluster: neural_priors/cogmodels/slurm_jobs/behaviour_model_fit.sh
"""
import argparse
import json
import os
import os.path as op
import pickle
import subprocess
import time

import numpy as np
import pandas as pd

DER = '/data/ds-neuralpriors/derivatives/behaviour_model'
N_SUBJECTS = 39
MODELS = {'power_truncated': 'power', 'log_truncated': 'log', 'linear_truncated': 'linear'}
DEFAULT_COND = {'power': ['alpha', 'nu', 'prior_mu', 'prior_sd', 'sigma_r'],
                'log': ['nu', 'prior_mu', 'prior_sd', 'sigma_r'],
                'linear': ['nu', 'prior_mu', 'prior_sd', 'sigma_r']}


def load_table(path, censor=True, confirmed_only=False):
    """Trials with a report (no-movement timeouts dropped). ``censor`` keeps the censor
    column (unfinished timeout reports); otherwise every report is treated as observed."""
    d = pd.read_csv(path, sep='\t', dtype={'subject': str})
    d = d[d['response'].notna()].copy()
    if confirmed_only:
        d = d[d.confirmed.astype(str) == 'True']
    if not censor:
        d = d.drop(columns='censor', errors='ignore')
    if d.subject.nunique() != N_SUBJECTS:
        print('\n' + '*' * 70 + f'\n*** WARNING: n = {d.subject.nunique()} participants, '
              f'expected {N_SUBJECTS}\n' + '*' * 70)
    d['response'] = d['response'].round().astype(int)
    d['n'] = d['n'].astype(float)
    return d.set_index(['subject', 'session', 'run', 'trial_nr']).sort_index()


def make_model(name, data, condition_params=None):
    from neural_priors.cogmodels.observer_model import RangeEstimationModel, space_for
    space = MODELS[name.split('+')[0]]
    cp = DEFAULT_COND[space] if condition_params is None else condition_params
    return RangeEstimationModel(data, encoding_space=space_for(space), condition_parameters=cp)


def git_rev(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=list(MODELS))
    ap.add_argument('--data', default=op.join(DER, 'behaviour_trials.tsv'))
    ap.add_argument('--out-dir', default=op.join(DER, 'fits'))
    ap.add_argument('--draws', type=int, default=1500)
    ap.add_argument('--tune', type=int, default=1500)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--target-accept', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--censor', action='store_true')
    ap.add_argument('--confirmed-only', action='store_true')
    ap.add_argument('--condition-params', nargs='*', default=None,
                    help='parameters with narrow-wide offsets (default: all but the lapse)')
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--low-rank-mass', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-ppc', type=int, default=200)
    args = ap.parse_args()

    import arviz as az
    import nutpie
    import pymc as pm
    import bauer

    tag = args.model + ('_cens' if args.censor else '') + args.tag
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_table(args.data, censor=args.censor, confirmed_only=args.confirmed_only)
    if args.subjects:
        data = data[data.index.get_level_values('subject').isin(args.subjects)]
    model = make_model(args.model, data, args.condition_params)
    print(f'{tag}: {len(data)} trials, {data.index.unique("subject").size} participants, '
          f'condition params {model.condition_parameters}; censored '
          f'{int((data.get("censor", pd.Series(0)) != 0).sum())}; bauer {op.dirname(bauer.__file__)}',
          flush=True)
    model.build_estimation_model(data, hierarchical=True)

    t0 = time.time()
    compiled = nutpie.compile_pymc_model(model.estimation_model)
    idata = nutpie.sample(compiled, draws=args.draws, tune=args.tune, chains=args.chains,
                          cores=args.chains, target_accept=args.target_accept,
                          seed=args.seed, progress_bar=False,
                          **({'low_rank_modified_mass_matrix': True} if args.low_rank_mass else {}))
    t_sample = time.time() - t0
    idata.attrs.update(model_name=args.model, censor=int(args.censor),
                       condition_params=' '.join(model.condition_parameters),
                       confirmed_only=int(args.confirmed_only), sampler='nutpie',
                       bauer_commit=git_rev(op.dirname(op.dirname(bauer.__file__))),
                       n_trials=len(data), sample_seconds=t_sample)
    keep = [g for g in idata.groups() if g in ('posterior', 'sample_stats')]
    az.InferenceData(**{g: idata[g] for g in keep}, attrs=idata.attrs) \
        .to_netcdf(op.join(args.out_dir, f'{tag}.nc'))

    ll = pm.compute_log_likelihood(idata, model=model.estimation_model,
                                   extend_inferencedata=False, progressbar=False)
    loo = az.loo(az.InferenceData(posterior=idata.posterior, log_likelihood=ll), pointwise=True)
    with open(op.join(args.out_dir, f'{tag}_loo.pkl'), 'wb') as f:
        pickle.dump(loo, f)

    pars = list(model.free_parameters)
    group = []
    for p in pars:
        group += [v for v in (f'{p}_mu', f'{p}_sd', f'{p}_kappa', f'{p}_cond_mu', f'{p}_cond_sd')
                  if v in idata.posterior]
    subj = [f'{p}_by_condition' if f'{p}_by_condition' in idata.posterior else p for p in pars]
    summ = az.summary(idata, var_names=group + subj, round_to=None)
    ss = idata.sample_stats
    row = dict(model=tag, n_trials=len(data), sample_minutes=t_sample / 60,
               divergences=int(ss.diverging.sum()),
               frac_maxdepth=float(ss.maxdepth_reached.mean()) if 'maxdepth_reached' in ss else np.nan,
               mean_steps=float(ss.n_steps.mean()) if 'n_steps' in ss else np.nan,
               max_rhat=float(summ.r_hat.max()), min_ess_bulk=float(summ.ess_bulk.min()),
               min_ess_tail=float(summ.ess_tail.min()),
               worst_rhat_param=str(summ.r_hat.idxmax()),
               elpd_loo=float(loo.elpd_loo), se=float(loo.se), p_loo=float(loo.p_loo),
               n_pareto_k_gt_07=int((loo.pareto_k.values > 0.7).sum()),
               censor=bool(args.censor), condition_params=' '.join(model.condition_parameters),
               bauer_commit=idata.attrs['bauer_commit'])
    with open(op.join(args.out_dir, f'{tag}_summary.json'), 'w') as f:
        json.dump(row, f, indent=1)
    print(json.dumps(row, indent=1), flush=True)

    if args.n_ppc:
        model.idata = idata
        ppc = model.posterior_predictive(idata, n_posterior_samples=args.n_ppc,
                                         random_seed=args.seed)
        ppc.reset_index().to_parquet(op.join(args.out_dir, f'{tag}_ppc.parquet'))
    print(f'done {tag} in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
