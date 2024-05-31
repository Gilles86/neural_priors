import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
from braincoder.models import LogGaussianPRF, GaussianPRF
from braincoder.optimize import WeightFitter
import numpy as np
from braincoder.utils import get_rsq
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from nilearn import image

def main(subject, session, smoothed, bids_folder, range_n=None):

    if session == 0:
        session = None

    assert(range_n in [None, 'wide', 'narrow', 'wide2']), "range_n must be either None, 'wide', 'narrow', or 'wide2"

    key = 'encoding_model.cv.denoise.wprf'

    if smoothed:
        key += '.smoothed'

    if range_n is not None:
        key += f'.range_{range_n}'

    if session is None:
        target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')
        fn_template = op.join(target_dir, 'sub-{subject}_ses-{session}_run-{run}_desc-{par}.{optimizer}_space-T1w_pars.nii.gz')
    else:
        target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', f'ses-{session}', 'func')
        fn_template = op.join(target_dir, 'sub-{subject}_ses-{session}_run-{run}_desc-{par}.{optimizer}_space-T1w_pars.nii.gz')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    sub = Subject(subject, bids_folder=bids_folder)

    data = sub.get_single_trial_estimates(session, smoothed=smoothed)

    behavior = sub.get_behavioral_data(session=session)

    paradigm = behavior['n']

    if range_n is not None:
        if range_n == 'wide2':
            range_mask = (behavior['range'] == 'wide') & (behavior['n'] < 26.)
        else:
            range_mask = behavior['range'] == range_n

    if range_n is not None:
        data = image.index_img(data, range_mask)
        paradigm = paradigm.loc[range_mask]

    paradigm = paradigm.droplevel(['subject'])

    if session is not None:
        paradigm = pd.concat([paradigm], keys=[session], names=['session'])

    masker = sub.get_brain_mask(session=session, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)


    mus = np.linspace(10, 40, 10)
    sigmas = np.ones(10) * 5.
    amplitudes = np.ones(10) * 1.
    baselines = np.zeros(10)

    parameters = pd.DataFrame({'mu': mus, 'sd': sigmas, 'amplitude': amplitudes, 'baseline': baselines})

    model = GaussianPRF(parameters=parameters)    

    # runs = paradigm.index.unique(level='run')
    cv_r2s = []

    keys = []

    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run']):


        test_data, test_paradigm = data.loc[(test_session, test_run)].copy(
        ).astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        print(test_data, test_paradigm)

        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        print(train_data)

        optimizer = WeightFitter(model, parameters, train_data, train_paradigm)

        weights = optimizer.fit(1.0)
        pred = model.predict(paradigm, parameters, weights)
        r2 = get_rsq(data, pred)

        target_fn = fn_template.format(subject=subject, session=test_session, run=test_run, par='r2', optimizer='grid')
        masker.inverse_transform(r2).to_filename(target_fn)

        cv_r2 = get_rsq(test_data, model.predict(paradigm=test_paradigm, weights=weights, parameters=parameters))

        target_fn = fn_template.format(subject=subject, session=test_session, run=test_run, par='cvr2', optimizer='optim')
        masker.inverse_transform(cv_r2).to_filename(target_fn)

        print(f'{cv_r2.isnull().sum()} voxels have a NaN value in the cv_r2 map')

        cv_r2s.append(cv_r2)
        keys.append((test_session, test_run))

    cv_r2 = pd.concat(cv_r2s, keys=keys, names=['session', 'run']).groupby(level=2, axis=0).mean()

    if session is None:
        target_fn = fn_template.replace('_ses-{session}_run-{run}_', '_').format(subject=subject, session=session, par='cvr2', optimizer='optim')
    else:
        target_fn = fn_template.replace('_run-{run}_', '_').format(subject=subject, session=session, par='cvr2', optimizer='optim')

    masker.inverse_transform(cv_r2).to_filename(target_fn)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('session', type=int)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--range', default=None)
    args = parser.parse_args()

    main(args.subject, args.session, smoothed=args.smoothed, bids_folder=args.bids_folder,
         range_n=args.range)
