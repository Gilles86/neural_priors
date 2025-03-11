import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
import numpy as np
from braincoder.utils import get_rsq
import pandas as pd
from models import get_regressors, get_paradigm, get_conditionspecific_parameters
from braincoder.models import RegressionAlphaGaussianPRF
from braincoder.optimize import ParameterFitter


def get_grids(model_label):

    modes_beta = np.arange(0, 16)
    modes1 = np.arange(10, 25)
    modes2 = np.arange(10, 40)
    sigmas = np.linspace(.5, 3, 10)
    alphas = [0.01]
    amplitudes = [1.0]
    baselines = [0.0]

    if model_label in [6]:
        return modes1, modes2, sigmas, alphas, amplitudes, baselines
    elif model_label in [3]:
        return modes_beta, sigmas, alphas, amplitudes, baselines
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")


def main(subject, smoothed, model_label=4, bids_folder='/data/ds-neuralpriors', debug=False, roi='NPCr'):

    max_n_iterations = 100 if debug else 1000

    # Create target folder
    key = 'encoding_model'
    key += f'.model{model_label}'
    key += '.alpha'

    if smoothed:
        key += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, model_label, gaussian=True)

    print(paradigm.describe())

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    print(data)

    regressors = get_regressors(model_label)
    print(regressors)

    if model_label in [3]:
        model = RegressionAlphaGaussianPRF(paradigm, data, regressors=regressors, baseline_parameter_values={'mu':10})
    else:
        model = RegressionAlphaGaussianPRF(paradigm, data, regressors=regressors)

    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    grid = get_grids(model_label)
    print(grid)

    grid_pars = optimizer.fit_grid(*grid)

    fixed_pars = list(model.parameter_labels)
    fixed_mapping = {
        (1, 3, 4, 6, 8, 9, 10, 11): [('amplitude_unbounded', 'Intercept'), ('baseline_unbounded', 'Intercept')],
        (2, 5, 7): [
            ('amplitude_unbounded', 'C(range)[0.0]'),
            ('baseline_unbounded', 'C(range)[0.0]'),
            ('amplitude_unbounded', 'C(range)[1.0]'),
            ('baseline_unbounded', 'C(range)[1.0]'),
        ]
    }

    for keys, to_remove in fixed_mapping.items():
        if model_label in keys:
            for item in to_remove:
                fixed_pars.pop(fixed_pars.index(item))

    # Fit one (only baseline/amplitude)
    gd_pars = optimizer.fit(
        init_pars=grid_pars, learning_rate=.05, store_intermediate_parameters=False,
        max_n_iterations=max_n_iterations, fixed_pars=fixed_pars, r2_atol=0.001,
        shared_pars=[('alpha_unbounded', 'Intercept')]
    )

    print(gd_pars)

    # Fit two
    gd_pars = optimizer.fit(
        init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False,
        max_n_iterations=max_n_iterations, r2_atol=0.00001,
        shared_pars=[('alpha_unbounded', 'Intercept')]
    )

    pred = model.predict(parameters=gd_pars, paradigm=paradigm)
    r2 = get_rsq(data, pred)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(r2).to_filename(target_fn)

    pars = get_conditionspecific_parameters(model_label, model, gd_pars, gaussian=True)

    print(pars.unstack('range'))

    for range_n, values in pars.groupby('range'):
        for par, value in values.T.iterrows():
            target_fn = op.join(target_dir, f'sub-{subject}_desc-{par}.{range_n}.optim_space-T1w_pars.nii.gz')
            masker.inverse_transform(value).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--model_label', default=4, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug)
