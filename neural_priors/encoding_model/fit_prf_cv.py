import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter
import numpy as np
from braincoder.utils import get_rsq
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from nilearn import image

def main(subject, session, smoothed, bids_folder, range_n=None):

    if session == 0:
        session = None

    assert(range_n in [None, 'wide', 'narrow']), "range_n must be either None, 'wide' or 'narrow'"

    key = 'encoding_model.cv.denoise'

    if smoothed:
        key += '.smoothed'

    if range_n is not None:
        key += f'.range_{range_n}'

    if session is None:
        target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')
        fn_template = op.join(target_dir, 'sub-{subject}_ses-{session}_run-{run}_desc-{par}.optim_space-T1w_pars.nii.gz')
    else:
        target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', f'ses-{session}', 'func')
        fn_template = op.join(target_dir, 'sub-{subject}_ses-{session}_run-{run}_desc-{par}.optim_space-T1w_pars.nii.gz')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    sub = Subject(subject, bids_folder=bids_folder)

    data = sub.get_single_trial_estimates(session, smoothed=smoothed)


    behavior = sub.get_behavioral_data(session=session, tasks=['estimation_task', ])

    paradigm = behavior['n']

    if range_n is not None:
        range_mask = behavior['range'] == range_n

    if range_n is not None:
        data = image.index_img(data, range_mask)
        paradigm = paradigm.loc[range_mask]

    paradigm = paradigm.droplevel(['subject', 'task', -1])

    if session is not None:
        paradigm = pd.concat([paradigm], keys=[session], names=['session'])

    masker = sub.get_brain_mask(session=session, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    print(data)
    print(paradigm)

    model = LogGaussianPRF()

    mus = np.linspace(5, 40, 50, dtype=np.float32)
    sds = np.linspace(3, 20, 50, dtype=np.float32)
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    # runs = paradigm.index.unique(level='run')
    cv_r2s = []

    keys = []

    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run']):


        test_data, test_paradigm = data.loc[(test_session, test_run)].copy(
        ).astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        print(test_data, test_paradigm)

        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        print(train_data)

        optimizer = ParameterFitter(model, train_data, train_paradigm)
        grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)

        grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=5)

        optimizer.fit(init_pars=grid_parameters, learning_rate=.001, store_intermediate_parameters=False, max_n_iterations=10000,
            r2_atol=0.0001, min_n_iterations=1000)

        # target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_run-{test_run}_desc-r2.optim_space-T1w_pars.nii.gz')
        target_fn = fn_template.format(subject=subject, session=session, run=test_run, par='r2')
        masker.inverse_transform(optimizer.r2).to_filename(target_fn)

        for par, values in optimizer.estimated_parameters.T.iterrows():
            print(values)
            target_fn = fn_template.format(subject=subject, session=session, run=test_run, par=par)
            masker.inverse_transform(values).to_filename(target_fn)

        cv_r2 = get_rsq(test_data, model.predict(paradigm=test_paradigm, parameters=optimizer.estimated_parameters))

        target_fn = fn_template.format(subject=subject, session=session, run=test_run, par='cvr2')
        masker.inverse_transform(cv_r2).to_filename(target_fn)

        print(f'{cv_r2.isnull().sum()} voxels have a NaN value in the cv_r2 map')

        cv_r2s.append(cv_r2)
        keys.append((test_session, test_run))

    cv_r2 = pd.concat(cv_r2s, keys=keys, names=['session', 'run']).groupby(level=2, axis=0).mean()

    if session is None:
        target_fn = fn_template.replace('_ses-{session}_run-{run}_', '_').format(subject=subject, session=session, par='cvr2')
    else:
        target_fn = fn_template.replace('_run-{run}_', '_').format(subject=subject, session=session, par='cvr2')

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