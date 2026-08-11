"""Estimate single-trial response amplitudes with GLMsingle.

Inputs: fMRIPrep-preprocessed BOLD in T1w space (with --smoothed, first
smoothed here with a 5 mm FWHM Gaussian kernel) and the BIDS events.tsv
onsets.

Design: one regressor per stimulus event AND per response event
(2 x 30 per run), with onsets snapped to the TR grid (TR 2.286 s, coded
as 2.3 s) as GLMsingle requires, and a stimulus duration of 0.6 s.
GLMsingle runs with its full pipeline: HRF library (wantlibrary),
GLMdenoise noise PCs (wantglmdenoise) and fractional ridge shrinkage
(wantfracridge).

The positional `session` argument selects one session; 0 means both
sessions concatenated (the production setting), with GLMsingle's
sessionindicator marking the session boundary.

Deliberately, no confound regressors beyond the GLMdenoise PCs are
included: motion/acquisition regressors correlate with the data-driven
PCs and can hurt downstream decoding (see Methods and
https://github.com/cvnlab/GLMsingle/pull/130).

Outputs, in derivatives/glm_stim1.denoise[.smoothed]: desc-stim_pe and
desc-response_pe 4D NIfTIs (one volume per trial) and a desc-R2 map.
"""
from glmsingle.glmsingle import GLM_single
import argparse
import os
import os.path as op
from nilearn import image
from neural_priors.utils.data import Subject
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def main(subject, session, bids_folder, confounds=False, smoothed=False):

    session = None if session == 0 else session

    derivatives = op.join(bids_folder, 'derivatives')

    sub = Subject(subject, bids_folder=bids_folder)

    runs = sub.get_runs(session)
    ims = sub.get_preprocessed_bold(session=session)

    base_dir = 'glm_stim1.denoise'

    if smoothed:
        base_dir += '.smoothed'
        ims = [image.smooth_img(im, fwhm=5.0) for im in ims]


    data = [image.load_img(im).get_fdata() for im in ims]

    # Every stimulus and response event gets its own regressor (single-trial design)
    onsets = sub.get_onsets(session)
    onsets['trial_type'] = onsets.apply(lambda row: f'stimulus_{row["n"]}' if row['trial_type'] == 'stimulus' else f'response_{row.response}', axis=1)
    onsets['duration'] = 0.0

    tr = 2.3
    n = 137
    frametimes = np.linspace(tr/2., (n - .5)*tr, n)
    # Snap onsets to the nearest TR: GLMsingle expects volume-aligned onset indicators
    onsets['onset'] = ((onsets['onset']+tr/2.) // 2.3) * 2.3

    if session is None:
        base_dir = op.join(derivatives, base_dir, f'sub-{subject}',
                        'func')
    else:
        base_dir = op.join(derivatives, base_dir, f'sub-{subject}',
                        f'ses-{session}', 'func')

        onsets = pd.concat([onsets], keys=[session], names=['session'])

    if not op.exists(base_dir):
        os.makedirs(base_dir)

    dm = onsets[['onset', 'trial_type', 'duration']].groupby(['session', 'run']).apply(lambda d: make_first_level_design_matrix(frametimes, d, hrf_model='fir', drift_model=None, drift_order=0).drop('constant', axis=1)).fillna(0.0)    # dm = [make_first_level_design_matrix(frametimes, on, hrf_model='fir', oversampling=100.,
    #                                      drift_order=0,
    #                                      drift_model=None).drop('constant', axis=1) for (session, run), on in onsets.groupby(['session', 'run'])]

    # dm = pd.concat(dm, keys=[(session, run) for (session, run), names=['run']).fillna(0)
    dm.columns = [c.replace('_delay_0', '') for c in dm.columns]
    # Binarize the FIR design to the 0/1 onset matrix GLMsingle expects
    dm /= dm.max()
    dm = np.round(dm)
    print(dm)
    print(dm.shape)

    X = [d.values for (session, run), d in dm.groupby(['session', 'run'])]
    print(X)

    for x in X:
        print(x.shape)

    # create a directory for saving GLMsingle outputs

    opt = dict()

    opt['sessionindicator'] = np.array([session for (session, run), d in dm.groupby(['session', 'run'])])[np.newaxis, :]
    # print(opt['sessionindicator'])

    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [0, 0, 0, 1]

    # Deliberately no extra confound regressors: they correlate with the
    # GLMdenoise PCs and can hurt decoding (see Methods).
    # See https://github.com/cvnlab/GLMsingle/pull/130
    # confounds = sub.get_confounds(session=session)
    # confounds = [d.values for run, d in sub.get_confounds().groupby('run')]
    # opt['extra_regressors'] = confounds

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    results_glmsingle = glmsingle_obj.fit(
        X,
        data,
        0.6,
        2.3,
        outputdir=base_dir)

    # Betas come back in chronological event order; stimulus and response events
    # alternate within a trial, so even indices are stimulus betas, odd are response
    betas = results_glmsingle['typed']['betasmd']
    betas = image.new_img_like(ims[0], betas)
    stim_betas = image.index_img(betas, slice(None, None, 2))
    resp_betas = image.index_img(betas, slice(1, None, 2))
    
    if session is None:
        fn_template = op.join(base_dir, 'sub-{subject}_task-task_space-T1w_desc-{par}_pe.nii.gz')
    else:
        fn_template = op.join(base_dir, 'sub-{subject}_ses-{session}_task-task_space-T1w_desc-{par}_pe.nii.gz')

    stim_betas.to_filename(fn_template.format(subject=subject, session=session, par='stim'))
    resp_betas.to_filename(fn_template.format(subject=subject, session=session, par='response'))

    r2 = results_glmsingle['typed']['R2']
    r2 = image.new_img_like(ims[0], r2)
    r2.to_filename(fn_template.format(subject=subject, session=session, par='R2'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder, smoothed=args.smoothed)
