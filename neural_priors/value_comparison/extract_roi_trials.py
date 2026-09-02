"""Dump one subject's GLMsingle single-trial estimates, masked to a few ROIs,
into a small .npz so the matched value-vs-numerosity analysis can run locally.

WHY.  The single-trial files (derivatives/glm_stim1.denoise/sub-XX/func/
*_desc-stim_pe.nii.gz) are ~300 MB whole-brain 4D NIfTIs and live only on the
cluster.  Everything the value-comparison analysis needs is a (480 trials x
n_voxels) matrix per ROI plus the paradigm, which is ~1 MB.  So we reduce here
(cluster) and plot there (laptop), per the usual split.

Besides the anatomical numerosity ROIs we also store a fixed random sample of
whole-brain voxels OUTSIDE the NPC mask.  That is the "what does this dataset
look like where there is no numerosity map" reference, and it plays the role
that a null / control ROI plays in the value dataset.

Run (cluster):
  python -m neural_priors.value_comparison.extract_roi_trials 01 \
      --bids_folder /shares/zne.uzh/gdehol/ds-neuralpriors \
      --out_dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/value_comparison
"""
import argparse
import os
import os.path as op

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker

from neural_priors.utils.data import Subject

DEFAULT_ROIS = ('NPCr', 'NPCl')
N_WHOLEBRAIN = 3000          # random voxels outside NPC, per subject


def get_paradigm(sub):
    """(480, ) numerosity + range + run/session index, exactly as fit_model.py
    builds it (presented numerosity, 'ground truth')."""
    behavior = sub.get_behavioral_data(session=None)
    par = behavior[['n', 'range']].copy()
    par['n'] = par['n'].astype(np.float32)
    return par


def main(subject, bids_folder, out_dir, rois=DEFAULT_ROIS, smoothed=False,
         n_wholebrain=N_WHOLEBRAIN, seed=None):
    sub = Subject(subject, bids_folder=bids_folder)
    par = get_paradigm(sub)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    assert data.shape[3] == len(par), (data.shape, len(par))

    out = {}
    for roi in rois:
        masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
        B = np.asarray(masker.fit_transform(data), dtype=np.float32)
        # GLMsingle leaves dead/edge voxels flat or NaN; drop them here so the
        # local analysis never has to think about it.
        ok = np.isfinite(B).all(0) & (B.std(0) > 1e-6)
        out[f'betas_{roi}'] = B[:, ok]
        print(f'  {roi}: {ok.sum()} / {B.shape[1]} voxels kept', flush=True)

    if n_wholebrain:
        brain = sub.get_brain_mask(epi_space=True, return_masker=False)
        npc = sub.get_volume_mask(roi='NPC', epi_space=True)
        npc = image.resample_to_img(npc, brain, interpolation='nearest',
                                    force_resample=False, copy_header=True)
        # the resampled NPC mask can come back with a trailing singleton axis
        brain_d = np.squeeze(np.asarray(brain.dataobj)) > 0
        npc_d = np.squeeze(np.asarray(npc.dataobj)) > 0
        outside = image.new_img_like(brain,
                                     (brain_d & ~npc_d).astype(np.int8))
        masker = NiftiMasker(mask_img=outside).fit()
        B = np.asarray(masker.transform(data), dtype=np.float32)
        ok = np.isfinite(B).all(0) & (B.std(0) > 1e-6)
        B = B[:, ok]
        rng = np.random.default_rng(int(subject) if seed is None else seed)
        take = rng.choice(B.shape[1], min(n_wholebrain, B.shape[1]),
                          replace=False)
        out['betas_wholebrain'] = B[:, np.sort(take)]
        print(f'  wholebrain(outside NPC): {B.shape[1]} usable, '
              f'{out["betas_wholebrain"].shape[1]} sampled', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    tag = '.smoothed' if smoothed else ''
    fn = op.join(out_dir, f'sub-{sub.subject_id}_trials{tag}.npz')
    np.savez_compressed(
        fn,
        n=par['n'].values.astype(np.float32),
        range_wide=(par['range'].values == 'wide').astype(np.int8),
        session=par.index.get_level_values('session').values.astype(np.int16),
        run=par.index.get_level_values('run').values.astype(np.int16),
        trial_nr=par.index.get_level_values('trial_nr').values.astype(np.int16),
        **out)
    print(f'saved {fn}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject')
    p.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--rois', default=','.join(DEFAULT_ROIS))
    p.add_argument('--n_wholebrain', type=int, default=N_WHOLEBRAIN)
    args = p.parse_args()
    out_dir = args.out_dir or op.join(args.bids_folder, 'derivatives',
                                      'value_comparison')
    main(args.subject, args.bids_folder, out_dir,
         rois=tuple(args.rois.split(',')), smoothed=args.smoothed,
         n_wholebrain=args.n_wholebrain)
