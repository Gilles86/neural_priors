"""
Cluster-side reduce for the LowRankNoise decoding study
(notes/decoding_lowrank_noise.md; opt-in `decode.py --noise_model lowrank_cv`).

For every participant and configuration it reads

  * the production posterior   derivatives/decoding2/<key>/...nvoxels-0_pars.tsv
  * the paired reference       <key>.<tag>/..._desc-refnoise_loglik.tsv
    (ResidualFitter, production settings, same fold parameters)
  * LowRankNoise               <key>.<tag>/..._desc-lowrank_loglik.tsv
    evaluated untempered (T = 1) and at the fold's CV temperature

and writes one row per (participant, trial, method) with posterior summaries
and calibration quantities, all computed on the posterior over x conditional
on the trial's true range (the production convention):

  E, sd           discrete posterior mean / SD on the full conditional grid
  E_paper, mad_paper, E_cens, mad_cens
                  the paper notebook's compute_E (trapz; narrow truncated at
                  25; *_cens additionally truncates wide at 25)
  logscore        log p(truth) - log(1 / n_grid)  (> 0 = beats flat)
  hpd, pit        randomised HPD level of the truth and randomised PIT
  ...and the same for the presented numerosity when the truth is the response.

Usage (cluster, via srun):
  python decoding_lowrankcv_collect.py --tag lowrankcv --out_dir notes/data
"""
import argparse
import os.path as op
import numpy as np
import pandas as pd
from neural_priors.utils.data import Subject, get_all_subject_ids

NARROW_UPPER = 25.0
CONFIGS = {'responses': 'model31.smoothed.spherical_noise.fit_responses',
           'groundtruth': 'model31.smoothed.spherical_noise'}


def read_grid_table(fn):
    t = pd.read_csv(fn, sep='\t', index_col=[0, 1, 2, 3, 4], header=[0, 1])
    t.columns = pd.MultiIndex.from_arrays(
        [t.columns.get_level_values(0).astype(float),
         t.columns.get_level_values(1).astype(float)], names=['n', 'range'])
    return t


def softmax_rows(a):
    a = a - np.nanmax(np.where(np.isfinite(a), a, -np.inf), 1, keepdims=True)
    p = np.exp(a)
    s = p.sum(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return p / s


def paper_E(p, n, upper=None):
    """The analysis notebook's _integrate on (possibly truncated) columns."""
    if upper is not None:
        m = n <= upper
        p, n = p[:, m], n[m]
    with np.errstate(invalid='ignore', divide='ignore'):
        p = p / np.trapz(p, n, axis=1)[:, None]
        E = np.trapz(p * n[None, :], n, axis=1)
        mad = np.trapz(np.abs(E[:, None] - n[None, :]) * p, n, axis=1)
    return E, mad


def calib(p, n, truth, rng):
    """log score, randomised HPD level and PIT of `truth` under p (rows sum 1)."""
    j = np.abs(n[None, :] - truth[:, None]).argmin(1)
    rows = np.arange(len(p))
    pt = p[rows, j]
    with np.errstate(divide='ignore'):
        logscore = np.log(pt) + np.log(len(n))
    u = rng.uniform(size=len(p))
    hpd = (p * (p > pt[:, None])).sum(1) \
        + u * (p * np.isclose(p, pt[:, None], rtol=1e-12, atol=0)).sum(1)
    cdf_below = (p * (np.arange(len(n))[None, :] < j[:, None])).sum(1)
    pit = cdf_below + rng.uniform(size=len(p)) * pt
    return logscore, hpd, pit


def summarise(logp_joint, trials, truth_cols, temperature=1.0, seed=0):
    """Per-trial summaries of a joint (x, range) log-likelihood/log-posterior table."""
    rng = np.random.default_rng(seed)
    cols = logp_joint.columns
    n_all = cols.get_level_values('n').values
    r_all = cols.get_level_values('range').values
    rng_trial = trials['range'].values
    out = []
    T = np.broadcast_to(np.asarray(temperature, dtype=float), (len(trials),))
    L = logp_joint.values.astype(np.float64)
    for cond in (0.0, 1.0):
        it = np.where(rng_trial == cond)[0]
        if len(it) == 0:
            continue
        ic = np.where(r_all == cond)[0]
        order = np.argsort(n_all[ic])
        ic = ic[order]
        n = n_all[ic]
        a = L[np.ix_(it, ic)]
        Tt = T[it][:, None]
        flat = np.isinf(Tt[:, 0])
        with np.errstate(invalid='ignore'):
            a = np.where(np.isinf(Tt), 0.0, a / np.where(np.isinf(Tt), 1.0, Tt))
        p = softmax_rows(a)
        d = pd.DataFrame(index=trials.index[it])
        d['E'] = p @ n
        d['sd'] = np.sqrt(np.clip(p @ n ** 2 - d['E'].values ** 2, 0, None))
        d['p_max'] = p.max(1)
        d['flat'] = flat
        # the paper's compute_E (trapz, narrow truncated at 25; *_cens: wide too)
        E1, m1 = paper_E(p, n, upper=NARROW_UPPER if cond == 0.0 else None)
        E2, m2 = paper_E(p, n, upper=NARROW_UPPER)
        d['E_paper'], d['mad_paper'], d['E_cens'], d['mad_cens'] = E1, m1, E2, m2
        for tc in truth_cols:
            ls, hpd, pit = calib(p, n, trials[tc].values[it].astype(float), rng)
            d[f'logscore_{tc}'], d[f'hpd_{tc}'], d[f'pit_{tc}'] = ls, hpd, pit
        d['n_grid'] = len(n)
        out.append(d)
    return pd.concat(out).loc[trials.index]


def fold_temperature(trials, noise_log):
    T = noise_log.set_index(['session', 'run2'])['temperature']
    run2 = ((trials.index.get_level_values('run').astype(int) - 1) % 4 + 1)
    keys = list(zip(trials.index.get_level_values('session').astype(int), run2))
    return np.array([T.loc[k] for k in keys], dtype=float)


def collect_one(subject, bids_folder, config, tag):
    key = CONFIGS[config]
    d0 = op.join(bids_folder, 'derivatives', 'decoding2', key, f'sub-{subject}', 'func')
    d1 = op.join(bids_folder, 'derivatives', 'decoding2', f'{key}.{tag}', f'sub-{subject}', 'func')
    base = f'sub-{subject}_mask-NPCr_nvoxels-0'
    fn_lr = op.join(d1, base + '_desc-lowrank_loglik.tsv')
    if not op.exists(fn_lr):
        return None, None
    lr = read_grid_table(fn_lr)
    ref = read_grid_table(op.join(d1, base + '_desc-refnoise_loglik.tsv'))
    noise_log = pd.read_csv(op.join(d1, base + '_desc-noisefit_log.tsv'), sep='\t')
    prod = read_grid_table(op.join(d0, base + '_pars.tsv'))
    with np.errstate(divide='ignore'):
        prod_log = np.log(prod)
    prod_log = prod_log.reindex(lr.index)
    ref = ref.reindex(lr.index)

    trials = lr.index.to_frame(index=False)
    trials.index = lr.index
    trials = trials.rename(columns={'x': 'truth'})
    beh = Subject(subject, bids_folder=bids_folder).get_behavioral_data(session=None)
    beh = beh.reset_index()
    beh = beh[['session', 'run', 'trial_nr', 'n', 'response']].astype(float)
    t2 = trials.reset_index(drop=True).astype({'session': float, 'run': float, 'trial_nr': float})
    t2 = t2.merge(beh, on=['session', 'run', 'trial_nr'], how='left', validate='1:1')
    t2.index = trials.index
    trials = t2
    truth_cols = ['truth', 'n'] if config == 'responses' else ['truth']

    T = fold_temperature(trials, noise_log)
    methods = {'production': (prod_log, 1.0), 'reference': (ref, 1.0),
               'lowrank_T1': (lr, 1.0), 'lowrank': (lr, T)}
    rows = []
    for m, (tab, temp) in methods.items():
        s = summarise(tab, trials, truth_cols, temperature=temp)
        s = pd.concat([trials[['truth', 'n', 'response']], s], axis=1)
        s['method'] = m
        s['temperature'] = temp
        rows.append(s.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, 'config', config)
    out.insert(0, 'subject', subject)
    noise_log.insert(0, 'config', config)
    noise_log.insert(0, 'subject', subject)
    return out, noise_log


def main(bids_folder, tag, out_dir, subjects=None, configs=('responses', 'groundtruth'), suffix=''):
    subjects = subjects or get_all_subject_ids()
    trials, logs, missing = [], [], []
    for config in configs:
        for sub in subjects:
            t, l = collect_one(sub, bids_folder, config, tag)
            if t is None:
                missing.append((config, sub))
                continue
            trials.append(t)
            logs.append(l)
            print(config, sub, len(t))
    if missing:
        print(f'WARNING: {len(missing)} (config, subject) missing: {missing}')
    trials = pd.concat(trials, ignore_index=True)
    logs = pd.concat(logs, ignore_index=True)
    trials.to_csv(op.join(out_dir, f'decoding_lowrankcv{suffix}_trials.tsv.gz'), sep='\t', index=False)
    logs.to_csv(op.join(out_dir, f'decoding_lowrankcv{suffix}_noisefit.tsv'), sep='\t', index=False)
    print('wrote', len(trials), 'trial rows;', len(logs), 'fold rows')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-neuralpriors')
    p.add_argument('--tag', default='lowrankcv')
    p.add_argument('--out_dir', default='notes/data')
    p.add_argument('--subjects', nargs='*', default=None)
    p.add_argument('--configs', nargs='*', default=['responses', 'groundtruth'])
    p.add_argument('--suffix', default='')
    a = p.parse_args()
    main(a.bids_folder, a.tag, a.out_dir, a.subjects, a.configs, a.suffix)
