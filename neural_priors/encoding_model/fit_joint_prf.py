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

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', gaussian=True,):

    if model_label == 1:
        regressors = {'mu':' 0 + C(range)'}
    elif model_label == 2:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)'}
    elif model_label == 3:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)'}
    elif model_label == 4:
        regressors = {'mu':'0 + C(range)', 'amplitude':'0 + C(range)'}
    elif model_label == 5:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline': '0+C(range)'}

    key = 'encoding_model.joint'
    key += f'.model{model_label}'

    if gaussian:
        key += '.gaussian'

    if smoothed:
        key += '.smoothed'

    sub = Subject(subject, bids_folder=bids_folder)
    behavior = sub.get_behavioral_data(session=None)

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
    paradigm['range'] = (paradigm['range'] == 'wide')
    paradigm = paradigm.astype(np.float32)

    masker = sub.get_brain_mask(session=None, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    if gaussian:
<<<<<<< HEAD
        model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
    else:
        raise NotImplementedError("Only Gaussian PRF is implemented")

    modes = np.linspace(5, 45, 30, dtype=np.float32)
    sigmas = np.linspace(1, 60, 30, dtype=np.float32)
=======
        model = RegressionGaussianPRF(paradigm=paradigm, regressors={'mu':'range'})
    else:
        raise NotImplementedError("Only Gaussian PRF is implemented")

    modes = np.linspace(5, 45, 20, dtype=np.float32)
    delta_modes = np.array([0], dtype=np.float32)
    sigmas = np.linspace(1, 60, 20, dtype=np.float32)
>>>>>>> 629656d (Merge.)
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

<<<<<<< HEAD
    if model_label == 1:
        grid_parameters = optimizer.fit_grid(modes, modes, sigmas, amplitudes, baselines, use_correlation_cost=True)
    elif model_label == 2:
        grid_parameters = optimizer.fit_grid(modes[::2], modes[::2], sigmas[::2], sigmas[::2], amplitudes, baselines, use_correlation_cost=True)
    elif model_label == 3:
        grid_parameters = optimizer.fit_grid(modes[::2], modes[::2], sigmas[::2], sigmas[::2], amplitudes, amplitudes, baselines, use_correlation_cost=True)
    elif model_label == 4:
        grid_parameters = optimizer.fit_grid(modes, modes, sigmas, amplitudes, amplitudes, baselines, use_correlation_cost=True)
    elif model_label == 5:
        grid_parameters = optimizer.fit_grid(modes[::2], modes[::2], sigmas[::2], sigmas[::2], amplitudes, amplitudes, baselines, baselines, use_correlation_cost=True)
=======
    grid_parameters = optimizer.fit_grid(modes, delta_modes, sigmas, amplitudes, baselines, use_correlation_cost=True)
>>>>>>> 629656d (Merge.)

    pred = model.predict(paradigm, grid_parameters)
    r2 = get_rsq(data, pred)

    fixed_pars = list(model.parameter_labels)
<<<<<<< HEAD
    if model_label in [1, 2]:
        fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'Intercept')))
    elif model_label in [3, 4, 5]:
        fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[0.0]')))
        fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[1.0]')))
    
    if model_label in [1,2,3,4]:
        fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'Intercept')))
    elif model_label in [5]:
        fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[0.0]')))
        fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[1.0]')))

    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
            fixed_pars=fixed_pars,
        r2_atol=0.0001)

    optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False, max_n_iterations=10000,
                  r2_atol=0.00001)
=======
    fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'Intercept')))
    fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'Intercept')))

    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=1000,
            fixed_pars=fixed_pars,
        r2_atol=0.0001)

    optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False, max_n_iterations=1000,
                  fixed_pars=fixed_pars, r2_atol=0.00001)
>>>>>>> 629656d (Merge.)

    
    paradigm = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    pars = model.get_transformed_parameters(paradigm, optimizer.estimated_parameters)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    for range_n, values in pars.groupby('range'):

        for par, value in values.T.iterrows():
            target_fn = op.join(target_dir, f'sub-{subject}_desc-{range_n}.{par}.optim_space-T1w_pars.nii.gz')
            masker.inverse_transform(value).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
<<<<<<< HEAD
    parser.add_argument('--model', type=int, default=1)
=======
>>>>>>> 629656d (Merge.)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

<<<<<<< HEAD
    main(args.subject, smoothed=args.smoothed, bids_folder=args.bids_folder, model_label=args.model)
=======
    main(args.subject, smoothed=args.smoothed, bids_folder=args.bids_folder)
>>>>>>> 629656d (Merge.)
