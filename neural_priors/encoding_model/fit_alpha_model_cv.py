import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
from braincoder.models import RegressionGaussianPRF, RegressionAlphaGaussianPRF
from braincoder.optimize import ParameterFitter
import numpy as np
from braincoder.utils import get_rsq
from nilearn import image
import pandas as pd
from models import get_paradigm, get_conditionspecific_parameters
from fit_alpha_model import get_model, fit_model

def main(subject, smoothed, model_label=4, bids_folder='/data/ds-neuralpriors', debug=False, roi='NPCr', store_cv_parameters=False):
    max_n_iterations = 100 if debug else 1000

    # Create target folder
    key = 'encoding_model'
    key += f'.model{model_label}'
    key += '.alpha'

    if smoothed:
        key += '.smoothed'

    key += '.cv'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, model_label, gaussian=True)

    paradigm = paradigm.set_index(pd.Index((paradigm.index.get_level_values('run') - 1) % 4 + 1, name='run2'), append=True)
    paradigm.index = paradigm.index.swaplevel('run', 'run2')
    paradigm = paradigm.astype(np.float32).droplevel(['run', 'trial_nr', 'subject'])    



    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    all_cvr2 = []

    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run2']):

        print(f'Fitting using session {test_session} run {test_run} as test set')

        test_data, test_paradigm = data.loc[(test_session, test_run)].copy().astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        # # Get model
        # regressors = get_regressors(model_label)
        # print(regressors)
        # if model_label in [3]:
        #     model = RegressionAlphaGaussianPRF(train_paradigm, train_data, regressors=regressors, baseline_parameter_values={'mu':10})
        # else:
        #     model = RegressionAlphaGaussianPRF(train_paradigm, train_data, regressors=regressors)
    
        # # # Fit model

        # optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

        # grid = get_grids(model_label)
        # print(grid)

        # grid_pars = optimizer.fit_grid(*grid)

        # fixed_pars = list(model.parameter_labels)
        # fixed_mapping = {
        #     (1, 3, 4, 6, 8, 9, 10, 11): [('amplitude_unbounded', 'Intercept'), ('baseline_unbounded', 'Intercept')],
        #     (2, 5, 7): [
        #         ('amplitude_unbounded', 'C(range)[0.0]'),
        #         ('baseline_unbounded', 'C(range)[0.0]'),
        #         ('amplitude_unbounded', 'C(range)[1.0]'),
        #         ('baseline_unbounded', 'C(range)[1.0]'),
        #     ]
        # }

        # for keys, to_remove in fixed_mapping.items():
        #     if model_label in keys:
        #         for item in to_remove:
        #             fixed_pars.pop(fixed_pars.index(item))

        # # Fit one (only baseline/amplitude)
        # gd_pars = optimizer.fit(
        #     init_pars=grid_pars, learning_rate=.05, store_intermediate_parameters=False,
        #     max_n_iterations=max_n_iterations, fixed_pars=fixed_pars, r2_atol=0.001,
        #     shared_pars=[('alpha_unbounded', 'Intercept')]
        # )

        # print(gd_pars)

        # # Fit two
        # gd_pars = optimizer.fit(
        #     init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False,
        #     max_n_iterations=max_n_iterations, r2_atol=0.00001,
        #     shared_pars=[('alpha_unbounded', 'Intercept')]
        # )

        model = get_model(model_label, train_paradigm, train_data)

        gd_pars = fit_model(model_label, model, train_paradigm, train_data, max_n_iterations=max_n_iterations)

        pred = model.predict(parameters=gd_pars, paradigm=train_paradigm)
        r2 = get_rsq(train_data, pred)

        # Out-of-set predictions
        if model_label not in [12, 13]:
            model.set_paradigm(test_paradigm)

        pred = model.predict(parameters=gd_pars, paradigm=test_paradigm)
        cvr2 = get_rsq(test_data, pred)

        if store_cv_parameters:
            target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run-{test_run}_desc-r2.optim_space-T1w_pars.nii.gz')
            masker.inverse_transform(r2).to_filename(target_fn)

            condition_specific_pars = get_conditionspecific_parameters(model_label, model, gd_pars, gaussian=True)

            for range_n, values in condition_specific_pars.groupby('range'):
                for par, value in values.T.iterrows():
                    target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run-{test_run}_desc-{par}.{range_n}.optim_space-T1w_pars.nii.gz')
                    masker.inverse_transform(value).to_filename(target_fn)


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
    parser.add_argument('--store_cv_parameters', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug, store_cv_parameters=args.store_cv_parameters)
