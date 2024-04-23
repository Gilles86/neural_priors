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

    derivatives = op.join(bids_folder, 'derivatives')

    sub = Subject(subject, bids_folder=bids_folder)
    runs = sub.get_runs(session)

    ims = sub.get_preprocessed_bold(session=session)

    print(ims)

    base_dir = 'glm_stim1.denoise'

    if smoothed:
        base_dir += '.smoothed'
        ims = [image.smooth_img(im, fwhm=5.0) for im in ims]


    data = [image.load_img(im).get_fdata() for im in ims]

    base_dir = op.join(derivatives, base_dir, f'sub-{subject}',
                       f'ses-{session}', 'func')

    if not op.exists(base_dir):
        os.makedirs(base_dir)

    onsets = sub.get_onsets(session)
    onsets['trial_type'] = onsets.apply(lambda row: f'stimulus_{row["n"]}' if row['trial_type'] == 'stimulus' else f'response_{row.response}', axis=1)
    onsets['duration'] = 0.0

    tr = 2.3
    n = 137
    frametimes = np.linspace(tr/2., (n - .5)*tr, n)
    onsets['onset'] = ((onsets['onset']+tr/2.) // 2.3) * 2.3

    dm = [make_first_level_design_matrix(frametimes, onsets.loc[run], hrf_model='fir', oversampling=100.,
                                         drift_order=0,
                                         drift_model=None).drop('constant', axis=1) for run in runs]

    dm = pd.concat(dm, keys=runs, names=['run']).fillna(0)
    dm.columns = [c.replace('_delay_0', '') for c in dm.columns]
    dm /= dm.max()
    print(dm)

    X = [dm.loc[run].values for run in runs]

    # create a directory for saving GLMsingle outputs

    opt = dict()

    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [0, 0, 0, 1]

    # see https://github.com/cvnlab/GLMsingle/pull/130
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

    betas = results_glmsingle['typed']['betasmd']
    betas = image.new_img_like(ims[0], betas)
    stim_betas = image.index_img(betas, slice(None, None, 2))
    resp_betas = image.index_img(betas, slice(1, None, 2))
    stim_betas.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stim_pe.nii.gz'))
    resp_betas.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-response_pe.nii.gz'))

    r2 = results_glmsingle['typed']['R2']
    r2 = image.new_img_like(ims[0], r2)
    r2.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-R2.nii.gz'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder, smoothed=args.smoothed)
