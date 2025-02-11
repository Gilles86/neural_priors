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

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', gaussian=True, debug=False):


    max_n_iterations = 100 if debug else 1000

    if model_label == 1:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline':'0 + C(range)'}
    else:
        raise NotImplementedError("Only model 1 is implemented")

    key = 'encoding_model'
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

    masker = sub.get_brain_mask(session=None, epi_space=True, return_masker=True, debug_mask=debug)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    if gaussian:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
    else:
        raise NotImplementedError("Only Gaussian PRF is implemented")

    modes = np.linspace(5, 45, 5, dtype=np.float32)
    sigmas = np.linspace(1, 60, 5, dtype=np.float32)
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)
    
    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    print(model.parameter_labels)

    if model_label == 1:
        grid_pars = optimizer.fit_grid(modes, modes,
                                       sigmas, sigmas,
                                        amplitudes, amplitudes,
                                        baselines, baselines)


    fixed_pars = list(model.parameter_labels)
    fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[0.0]')))
    fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[0.0]')))
    fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[1.0]')))
    fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[1.0]')))

    gd_pars = optimizer.fit(init_pars=grid_pars, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=max_n_iterations,
            fixed_pars=fixed_pars,
        r2_atol=0.0001)

    gd_pars = optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False, max_n_iterations=max_n_iterations,
                  fixed_pars=fixed_pars, r2_atol=0.00001)

    conditions = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    pars = model.get_conditionspecific_parameters(conditions, optimizer.estimated_parameters)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    for range_n, values in pars.groupby('range'):
        for par, value in values.T.iterrows():
            target_fn = op.join(target_dir, f'sub-{subject}_desc-{par}.{range_n}.optim_space-T1w_pars.nii.gz')
            masker.inverse_transform(value).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--model_label', default=1, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug)
