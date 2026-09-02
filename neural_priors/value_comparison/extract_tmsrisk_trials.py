"""Same as extract_roi_trials.py, but for the tms_risk dataset.

WHY tms_risk is here.  The numerosity-vs-value comparison has one big
confound: neural_priors has 480 trials per subject and value_prf has 128, so a
numerosity advantage could just be trial count.  Subsampling neural_priors down
to 128 is one control, but it is still the same subjects and the same
acquisition.  tms_risk is an INDEPENDENT test:

    * the stimulus is numerosity again (n1 of a risky-choice pair, 7-86),
    * the ROI is the same numerosity-tuned parietal cortex (NPC),
    * the betas are the same GLMsingle pipeline,
    * and session 1 (baseline, no TMS) has 6 runs x 20 trials = **120 trials**,
      essentially the value dataset's 128.

So if numerosity tuning is strong here too, 128 trials is enough and value is
genuinely weak; if it is weak here, trial count explains the difference.

Only session 1 is used: sessions 2 and 3 follow cTBS to parietal cortex or
vertex, which is exactly the manipulation that study is about.

Run (cluster):
  python -m neural_priors.value_comparison.extract_tmsrisk_trials 01 \
      --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk \
      --out_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/value_comparison
"""
import argparse
import os
import os.path as op

import numpy as np
from nilearn import image
from nilearn.maskers import NiftiMasker

from tms_risk.utils.data import Subject

DEFAULT_ROIS = ('NPCr', 'NPC12r', 'NPCl')
N_WHOLEBRAIN = 3000
SESSION = 1              # baseline only -- 2 and 3 are post-cTBS


def main(subject, bids_folder, out_dir, rois=DEFAULT_ROIS, smoothed=False,
         n_wholebrain=N_WHOLEBRAIN, session=SESSION):
    sub = Subject(subject, bids_folder=bids_folder)
    par = sub.get_paradigm(session=session)
    n = par['n1'].values.astype(np.float32)
    runs = par.index.get_level_values('run').values.astype(np.int16)
    trials = par.index.get_level_values('trial_nr').values.astype(np.int16)

    key = 'glm_stim1.denoise' + ('.smoothed' if smoothed else '')
    fn = op.join(bids_folder, 'derivatives', key, f'sub-{sub.subject}',
                 f'ses-{session}', 'func',
                 f'sub-{sub.subject}_ses-{session}_task-task_space-T1w'
                 f'_desc-stims1_pe.nii.gz')
    data = image.load_img(fn, dtype=np.float32)
    assert data.shape[3] == len(n), (data.shape, len(n))

    out = {}
    for roi in rois:
        mask = sub.get_volume_mask(roi=roi, session=session, epi_space=True)
        masker = NiftiMasker(mask_img=mask)
        B = np.asarray(masker.fit_transform(data), dtype=np.float32)
        ok = np.isfinite(B).all(0) & (B.std(0) > 1e-6)
        out[f'betas_{roi}'] = B[:, ok]
        print(f'  {roi}: {ok.sum()} / {B.shape[1]} voxels kept', flush=True)

    if n_wholebrain:
        brain = sub.get_volume_mask(roi=None, session=session, epi_space=True)
        npc = sub.get_volume_mask(roi='NPC', session=session, epi_space=True)
        npc = image.resample_to_img(npc, brain, interpolation='nearest',
                                    force_resample=True, copy_header=True)
        brain_d = np.squeeze(np.asarray(brain.dataobj)) > 0
        npc_d = np.squeeze(np.asarray(npc.dataobj)) > 0
        outside = image.new_img_like(brain, (brain_d & ~npc_d).astype(np.int8))
        masker = NiftiMasker(mask_img=outside).fit()
        B = np.asarray(masker.transform(data), dtype=np.float32)
        ok = np.isfinite(B).all(0) & (B.std(0) > 1e-6)
        B = B[:, ok]
        rng = np.random.default_rng(int(subject))
        take = np.sort(rng.choice(B.shape[1], min(n_wholebrain, B.shape[1]),
                                  replace=False))
        out['betas_wholebrain'] = B[:, take]
        print(f'  wholebrain(outside NPC): {B.shape[1]} usable, '
              f'{out["betas_wholebrain"].shape[1]} sampled', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    tag = '.smoothed' if smoothed else ''
    fnout = op.join(out_dir, f'sub-{sub.subject}_ses-{session}_trials{tag}.npz')
    np.savez_compressed(fnout, n=n, run=runs, trial_nr=trials,
                        session=np.full(len(n), session, np.int16),
                        range_wide=np.zeros(len(n), np.int8), **out)
    print(f'saved {fnout}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject')
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--rois', default=','.join(DEFAULT_ROIS))
    p.add_argument('--n_wholebrain', type=int, default=N_WHOLEBRAIN)
    p.add_argument('--session', type=int, default=SESSION)
    a = p.parse_args()
    main(a.subject, a.bids_folder,
         a.out_dir or op.join(a.bids_folder, 'derivatives', 'value_comparison'),
         rois=tuple(a.rois.split(',')), smoothed=a.smoothed,
         n_wholebrain=a.n_wholebrain, session=a.session)
