"""
Collect the noise-model x voxel-set comparison (decode.py --noise_model compare,
slurm_jobs/decode_wholeroi.sh): spherical, Jehee and LowRankNoise (fixed inner
parameters, evaluated at the fold's CV temperature) on the same folds, encoding
parameters and voxel sets (cvR^2 > 0 = '0', top 500, top 1000, whole ROI = 'all').

Metrics as in decoding_nvoxels_collect.py (posterior over n conditional on the
trial's range, full grid; per participant x range r(E, truth), MAE, paper
convention; range = narrow / wide / both). truth = response (paper config) or
presented n (ground truth).

Cluster (reduce):
  python decoding_wholeroi_collect.py --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
      --out notes/data/decoding_nvoxels_wholeroi_by_subject.tsv
Local (append the cells missing from the main table; writes a backup first):
  python decoding_wholeroi_collect.py --append_only \
      --out notes/data/decoding_nvoxels_wholeroi_by_subject.tsv \
      --append_to notes/data/decoding_nvoxels_by_subject.tsv
"""
import argparse
import os.path as op
import shutil
import time
import numpy as np
import pandas as pd

CONFIGS = {'responses': 'model31.smoothed.spherical_noise.fit_responses',
           'groundtruth': 'model31.smoothed.spherical_noise'}
SETS = ['0', '500', '1000', 'all']


def collect(bids_folder, tag, subjects=None):
    from neural_priors.utils.data import Subject, get_all_subject_ids
    from neural_priors.encoding_model.decoding_lowrankcv_collect import (
        read_grid_table, summarise, fold_temperature)
    rows, logs = [], []
    for sub in subjects or get_all_subject_ids():
        beh = Subject(sub, bids_folder=bids_folder).get_behavioral_data(session=None).reset_index()
        beh = beh[['session', 'run', 'trial_nr', 'n', 'response']].astype(float)
        for config, key in CONFIGS.items():
            d = op.join(bids_folder, 'derivatives', 'decoding2', f'{key}.{tag}', f'sub-{sub}', 'func')
            log_fn = op.join(d, f'sub-{sub}_mask-NPCr_desc-noisefit_log.tsv')
            if not op.exists(log_fn):
                print('missing', config, sub)
                continue
            log = pd.read_csv(log_fn, sep='\t', dtype={'n_voxels': str})
            logs.append(log.assign(subject=sub, config=config))
            for nv in SETS:
                for noise in ('spherical', 'jehee', 'lowrank'):
                    fn = op.join(d, f'sub-{sub}_mask-NPCr_nvoxels-{nv}_desc-{noise}_loglik.tsv')
                    if not op.exists(fn):
                        continue
                    ll = read_grid_table(fn)
                    tr = ll.index.to_frame(index=False)[['session', 'run', 'trial_nr', 'range']]
                    tr = tr.astype(float).merge(beh, on=['session', 'run', 'trial_nr'], how='left',
                                                validate='1:1')
                    tr.index = ll.index
                    T = fold_temperature(tr, log[log.n_voxels == nv]) if noise == 'lowrank' else 1.0
                    s = summarise(ll, tr, [], temperature=T)
                    s = pd.concat([tr[['range', 'n', 'response']], s[['E', 'sd', 'E_paper']]], axis=1)
                    rows.append(s.assign(subject=sub, config=config, noise=noise,
                                         n_voxels=nv).reset_index(drop=True))
        print(sub, flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(logs, ignore_index=True)


def append_missing(new_fn, main_fn):
    new = pd.read_csv(new_fn, sep='\t', dtype={'subject': str, 'n_voxels': str})
    main = pd.read_csv(main_fn, sep='\t', dtype={'subject': str, 'n_voxels': str})
    bak = main_fn + time.strftime('.bak-%Y%m%d-%H%M%S')
    shutil.copy(main_fn, bak)
    # the main table's lowrank cvR2 > 0 cell is the `refit` study; the fixed-inner series of
    # this run is also added in full as 'lowrank_fixed' (like-for-like across voxel sets)
    new = pd.concat([new, new[new.noise == 'lowrank'].assign(noise='lowrank_fixed')],
                    ignore_index=True)
    have = set(map(tuple, main[['config', 'noise', 'n_voxels']].drop_duplicates().values))
    add = new[[(c, n, v) not in have for c, n, v in new[['config', 'noise', 'n_voxels']].values]]
    out = pd.concat([main, add[main.columns.intersection(add.columns)]], ignore_index=True)
    out.to_csv(main_fn, sep='\t', index=False, float_format='%.5g')
    print(f'backup {bak}; appended {len(add)} rows, cells:',
          sorted(set(map(tuple, add[['config', 'noise', 'n_voxels']].values))))


def main(a):
    if not a.append_only:
        from neural_priors.encoding_model.decoding_nvoxels_collect import per_subject
        d, logs = collect(a.bids_folder, a.tag, a.subjects)
        res = pd.concat([per_subject(d[d.config == c], t)
                         for c, t in (('responses', 'response'), ('groundtruth', 'n'))
                         if (d.config == c).any()], ignore_index=True)
        res.to_csv(a.out, sep='\t', index=False, float_format='%.5g')
        logs.to_csv(a.out.replace('_by_subject.tsv', '_noisefit.tsv'), sep='\t', index=False)
        print(res[res.range == 'both'].groupby(['config', 'noise', 'n_voxels'])[['r', 'mae']]
              .agg(['mean', 'count']).round(3).to_string())
    if a.append_to:
        append_missing(a.out, a.append_to)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-neuralpriors')
    p.add_argument('--tag', default='wholeroi')
    p.add_argument('--subjects', nargs='*', default=None)
    p.add_argument('--out', default='notes/data/decoding_nvoxels_wholeroi_by_subject.tsv')
    p.add_argument('--append_to', default=None)
    p.add_argument('--append_only', action='store_true')
    main(p.parse_args())
