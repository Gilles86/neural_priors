import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
from braincoder.models import RegressionGaussianPRF
from braincoder.optimize import ParameterFitter
import numpy as np
from braincoder.utils import get_rsq
from nilearn import image
import pandas as pd
from models import get_paradigm, get_model, fit_model, get_conditionspecific_parameters

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', gaussian=True, debug=False):

    max_n_iterations = 10 if debug else 1000

    # Create target folder
    key = 'encoding_model'
    key += f'.model{model_label}'

    if gaussian:
        key += '.gaussian'
    else:
        key += '.logspace'

    if smoothed:
        key += '.smoothed'

    key += '.cv'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, model_label, gaussian=gaussian)

    paradigm = paradigm.set_index(pd.Index((paradigm.index.get_level_values('run') - 1) % 4 + 1, name='run2'), append=True)
    paradigm.index = paradigm.index.swaplevel('run', 'run2')
    paradigm = paradigm.astype(np.float32).droplevel(['run', 'trial_nr', 'subject'])

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_brain_mask(session=None, epi_space=True, return_masker=True, debug_mask=debug)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    all_cvr2 = []

    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run2']):

        print(f'Fitting using session {test_session} run {test_run} as test set')

        test_data, test_paradigm = data.loc[(test_session, test_run)].copy().astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        # Get model
        model = get_model(train_paradigm, model_label, gaussian=gaussian)
    
        # # Fit model
        pars = fit_model(model, train_paradigm, train_data, model_label, max_n_iterations=max_n_iterations, gaussian=gaussian)

        # In-set prediction
        pred = model.predict(parameters=pars, paradigm=train_paradigm)
        r2 = get_rsq(train_data, pred)

        target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run-{test_run}_desc-r2.optim_space-T1w_pars.nii.gz')
        masker.inverse_transform(r2).to_filename(target_fn)

        condition_specific_pars = get_conditionspecific_parameters(model_label, model, pars, gaussian=gaussian)

        for range_n, values in condition_specific_pars.groupby('range'):
            for par, value in values.T.iterrows():
                target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run-{test_run}_desc-{par}.{range_n}.optim_space-T1w_pars.nii.gz')
                masker.inverse_transform(value).to_filename(target_fn)

        model.set_paradigm(test_paradigm)
        pred = model.predict(parameters=pars, paradigm=test_paradigm)
        cvr2 = get_rsq(test_data, pred)

        target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run-{test_run}_desc-cvr2.optim_space-T1w_pars.nii.gz')
        masker.inverse_transform(cvr2).to_filename(target_fn)

        print(cvr2)

        all_cvr2.append(cvr2)

    all_cvr2 = pd.concat(all_cvr2, axis=1)
    mean_cvr2 = all_cvr2.mean(axis=1)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-cvr2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(mean_cvr2).to_filename(target_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--model_label', default=1, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_space', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug, gaussian=not args.log_space)
