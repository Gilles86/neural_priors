"""
Decoding accuracy per noise model x voxel count, for a figure next to the
abstract_values one (spherical vs van Bergen & Jehee vs LowRankNoise).

Reads the stored production-style posteriors (no new decoding):

  spherical  derivatives/decoding2/model31.smoothed.spherical_noise[.fit_responses]/
  jehee      derivatives/decoding2/model31.smoothed[.fit_responses]/
             (ResidualFitter, spherical=False: rho, tau, sigma^2 WW')
  ...each at nvoxels-0 (cvR^2 > 0 in the training runs, ~90-570 voxels),
  100, 200, 500 where they exist;

and the low-rank study's per-trial table (notes/data/decoding_lowrankcv_refit_
trials.tsv.gz, method 'lowrank', cvR^2 > 0 voxels only).

Posterior over n conditional on the trial's range (production convention),
full grid; per participant x range r(E, truth) and MAE, and the paper's
convention (r_paper, mae_paper: narrow posterior truncated at 25) (range = narrow /
wide), plus their mean over the two ranges (range = both). truth = response in the paper configuration, the presented
numerosity in the ground-truth configuration.

  python decoding_nvoxels_collect.py --bids_folder /data/ds-neuralpriors
"""
import argparse
import os.path as op
import numpy as np
import pandas as pd
from neural_priors.utils.data import Subject, get_all_subject_ids
from neural_priors.encoding_model.decoding_lowrankcv_collect import read_grid_table, summarise

DIRS = {('responses', 'spherical'): 'model31.smoothed.spherical_noise.fit_responses',
        ('responses', 'jehee'): 'model31.smoothed.fit_responses',
        ('groundtruth', 'spherical'): 'model31.smoothed.spherical_noise',
        ('groundtruth', 'jehee'): 'model31.smoothed'}
NVOX = ['0', '100', '200', '500']


def per_subject(d, truth):
    rows = []
    for keys, g in d.groupby(['config', 'noise', 'n_voxels', 'subject', 'range']):
        ok = g.E.notna() & g[truth].notna()
        g = g[ok]
        p = g.dropna(subset=['E_paper'])
        rows.append(dict(zip(['config', 'noise', 'n_voxels', 'subject', 'range'], keys),
                         r=np.corrcoef(g.E, g[truth])[0, 1], mae=(g.E - g[truth]).abs().mean(),
                         r_paper=np.corrcoef(p.E_paper, p[truth])[0, 1],
                         mae_paper=(p.E_paper - p[truth]).abs().mean(),
                         post_sd=g.sd.mean(), n_trials=len(g)))
    out = pd.DataFrame(rows)
    out['range'] = out['range'].map({0.0: 'narrow', 1.0: 'wide'})
    both = (out.groupby(['config', 'noise', 'n_voxels', 'subject'])[['r', 'mae', 'r_paper', 'mae_paper', 'post_sd']]
            .mean().reset_index().assign(range='both'))
    return pd.concat([out.drop(columns='n_trials'), both], ignore_index=True)


def main(bids_folder, lowrank_trials, out):
    rows = []
    for sub in get_all_subject_ids():
        beh = Subject(sub, bids_folder=bids_folder).get_behavioral_data(session=None).reset_index()
        beh = beh[['session', 'run', 'trial_nr', 'n', 'response']].astype(float)
        for (config, noise), key in DIRS.items():
            for nv in NVOX:
                fn = op.join(bids_folder, 'derivatives', 'decoding2', key, f'sub-{sub}', 'func',
                             f'sub-{sub}_mask-NPCr_nvoxels-{nv}_pars.tsv')
                if not op.exists(fn):
                    continue
                p = read_grid_table(fn)
                with np.errstate(divide='ignore'):
                    lp = np.log(p)
                tr = lp.index.to_frame(index=False)[['session', 'run', 'trial_nr', 'range']]
                tr = tr.astype(float).merge(beh, on=['session', 'run', 'trial_nr'], how='left',
                                            validate='1:1')
                tr.index = lp.index
                s = summarise(lp, tr, [], temperature=1.0)
                s = pd.concat([tr[['range', 'n', 'response']], s[['E', 'sd', 'E_paper']]], axis=1)
                s = s.assign(subject=sub, config=config, noise=noise, n_voxels=nv)
                rows.append(s.reset_index(drop=True))
        print(sub, flush=True)
    d = pd.concat(rows, ignore_index=True)
    lr = pd.read_csv(lowrank_trials, sep='\t', dtype={'subject': str})
    lr = lr[lr.method == 'lowrank'].assign(noise='lowrank', n_voxels='0')
    lr['subject'] = lr.subject.str.zfill(2)
    d = pd.concat([d, lr[d.columns.intersection(lr.columns)]], ignore_index=True)
    res = []
    for config, truth in (('responses', 'response'), ('groundtruth', 'n')):
        res.append(per_subject(d[d.config == config], truth))
    res = pd.concat(res, ignore_index=True)
    res.to_csv(out, sep='\t', index=False, float_format='%.5g')
    print(res.groupby(['config', 'range', 'noise', 'n_voxels'])[['r', 'mae']]
          .agg(['mean', 'count']).round(3).to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    p.add_argument('--lowrank_trials', default='notes/data/decoding_lowrankcv_refit_trials.tsv.gz')
    p.add_argument('--out', default='notes/data/decoding_nvoxels_by_subject.tsv')
    a = p.parse_args()
    main(a.bids_folder, a.lowrank_trials, a.out)
