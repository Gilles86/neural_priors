import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
import numpy as np
from braincoder.utils import get_rsq
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn import image
from braincoder.optimize import ParameterFitter
from braincoder.models import RegressionGaussianPRF
from models import AlphaDeltaModel, get_paradigm

# Model 0: null model
# Model 1: delta_wide is 2
# model 2: delta_wide is free but same across voxels
# model 3: delta_wide is free and different for each voxel
# model 4: delta_wide is 2 and identity below 10
# model 5: delta_wide is fitted, same across voxels, but identity below 10


def get_model(model_label):

    if model_label in [4, 5]:
        model = AlphaDeltaModel(identity_below_range=True)
    else:
        model = AlphaDeltaModel()

    return model

def get_grid(model_label):

    modes = np.linspace(5, 45, 41)
    sds = np.linspace(np.log(2), np.log(30), 30)
    alphas = np.array([1e-4], dtype=np.float32)
    intersection_point = [10.0]
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    if model_label in [0]:
        delta_wides = [1.0]
    elif model_label in [1, 4]:
        delta_wides = [2.0]
    elif model_label in [2, 3, 5]:
        delta_wides = [.5, 1.0, 1.5, 2.0, 2.5]

    return modes, sds, alphas, delta_wides, intersection_point,  amplitudes, baselines

def fit_model(model_label, model, data, paradigm, max_n_iterations=1000):

    # Fit model
    fitter = ParameterFitter(model, data, paradigm)
    grid = get_grid(model_label)

    print(grid)

    grid_pars = fitter.fit_grid(*grid, use_correlation_cost=True)
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

    fixed_pars = []
    shared_pars = []

    fixed_pars += ['alpha', 'lower_bound_range']

    if model_label in [0, 1, 4]:
        fixed_pars += ['delta_wide']
    elif model_label in [2, 5]:
        shared_pars += ['delta_wide']

    gd_pars = fitter.fit(max_n_iterations=max_n_iterations, init_pars=grid_pars,
                         shared_pars=shared_pars, fixed_pars=fixed_pars)

    return gd_pars


def get_conditionspecific_parameters(model_label, estimated_parameters):
    
    pars = pd.DataFrame()

    pars[('mu', 'narrow')] = estimated_parameters['mu_narrow']
    pars[('mu', 'wide')] = estimated_parameters['delta_wide'] * (estimated_parameters['mu_narrow'] - estimated_parameters['lower_bound_range']) + estimated_parameters['lower_bound_range']

    if model_label in [4, 5]:
        pars[('mu', 'wide')] = pars[('mu', 'wide')].where(pars[('mu', 'wide')] > 10, pars[('mu', 'narrow')])

    for p in ['sd', 'amplitude', 'baseline', 'alpha', 'delta_wide', 'lower_bound_range']:
        pars[(p, 'narrow')] = estimated_parameters[p]
        pars[(p, 'wide')] = estimated_parameters[p]

    pars.columns = pd.MultiIndex.from_tuples(pars.columns, names=['parameter', 'range'])
    
    return pars.stack('range').reorder_levels(['range', 'source'], axis=0).sort_index()

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', gaussian=True, debug=False, roi='NPCr'):

    max_n_iterations = 100 if debug else 1000

    # Create target folder
    key = f'model{model_label}'

    if smoothed:
        key += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', 'encoding_models2', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, model_label, gaussian=gaussian)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    # Get model
    model = get_model(model_label)
    
    gd_pars = fit_model(model_label, model, data, paradigm, max_n_iterations=max_n_iterations)

    pred = model.predict(parameters=gd_pars, paradigm=paradigm)
    r2 = get_rsq(data, pred)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(r2).to_filename(target_fn)

    pars = get_conditionspecific_parameters(model_label, gd_pars)

    print(pars.unstack('range'))

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
