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
from models import AlphaDeltaModel

# Model 0: null model
# Model 1: delta_wide is 2
# model 2: delta_wide is free but same across voxels
# model 3: delta_wide is free and different for each voxel
# model 4: delta_wide is 2 and identity below 10
# model 5: delta_wide is fitted, same across voxels, but identity below 10
# Model 6: Like model 3, gamma free
# Model 7: Like model 4, gamma free
# Model 8: Like model 5, gamma free


def get_model(model_label):

    if model_label in [4, 5, 7, 8]:
        model = AlphaDeltaModel(identity_below_range=True)
    else:
        model = AlphaDeltaModel()

    return model

def get_grid(model_label):

    modes = np.linspace(5, 45, 41)
    sds = np.linspace(np.log(2), np.log(30), 30)
    if model_label in range(6, 9):
        alphas = np.linspace(-1., 1., 5)
    else:
        alphas = np.array([1e-4], dtype=np.float32)

    intersection_point = [10.0]
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    if model_label in [0]:
        delta_wides = [1.0]
    elif model_label in [1, 4, 7]:
        delta_wides = [2.0]
    elif model_label in [2, 3, 5, 6, 8]:
        delta_wides = np.linspace(.3, 3., 10)

    return modes, sds, alphas, delta_wides, intersection_point,  amplitudes, baselines

def fit_model(model_label, model, data, paradigm, max_n_iterations=1000):

    # Fit model
    fitter = ParameterFitter(model, data, paradigm)
    grid = get_grid(model_label)

    grid_pars = fitter.fit_grid(*grid, use_correlation_cost=True)
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

    fixed_pars = []
    shared_pars = []

    if model_label not in [3, 6]:
        fixed_pars += ['lower_bound_range']

    if model_label in range(0, 6):
        fixed_pars += ['alpha']
    else:
        shared_pars += ['alpha']

    if model_label in [0, 1, 4, 7]:
        fixed_pars += ['delta_wide']
    elif model_label in [2, 5, 8]:
        shared_pars += ['delta_wide']

    gd_pars = fitter.fit(max_n_iterations=max_n_iterations, init_pars=grid_pars,
                         shared_pars=shared_pars, fixed_pars=fixed_pars)

    return gd_pars

def fit_model_cv(data, paradigm, model_label, max_n_iterations=2000):
    """
    Perform cross-validation by splitting the data based on 'run2'.

    Parameters:
    - data: pd.DataFrame, the data to be used for cross-validation.
    - paradigm: pd.DataFrame, the paradigm associated with the data.
    - model_label: int, the label of the model to be used.
    - max_n_iterations: int, maximum number of iterations for model fitting.

    Returns:
    - mean_cvr2: pd.Series, mean cross-validated R^2 for each voxel.
    """
    all_cvr2 = []

    # Cross-validation loop
    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run2']):
        test_data = data.loc[(test_session, test_run)].copy().astype(np.float32)
        train_data = data.drop((test_session, test_run)).copy()
        test_paradigm = paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        train_paradigm = paradigm.drop((test_session, test_run)).copy()

        # Get model
        model = get_model(model_label)

        # Fit model on training data
        gd_pars = fit_model(model_label, model, train_data, train_paradigm, max_n_iterations=max_n_iterations)

        # Predict on test data
        test_pred = model.predict(parameters=gd_pars, paradigm=test_paradigm)
        cv_r2 = get_rsq(test_data, test_pred)

        all_cvr2.append(cv_r2)

    # Aggregate results
    all_cvr2 = pd.concat(all_cvr2, axis=1)
    mean_cvr2 = all_cvr2.mean(axis=1)

    return mean_cvr2

def get_conditionspecific_parameters(model_label, estimated_parameters):
    
    pars = pd.DataFrame()

    pars[('mu', 'narrow')] = estimated_parameters['mu_narrow']
    pars[('mu', 'wide')] = estimated_parameters['delta_wide'] * (estimated_parameters['mu_narrow'] - estimated_parameters['lower_bound_range']) + estimated_parameters['lower_bound_range']

    if model_label in [4, 5, 7, 8]:
        pars[('mu', 'wide')] = pars[('mu', 'wide')].where(pars[('mu', 'wide')] > 10, pars[('mu', 'narrow')])

    for p in ['sd', 'amplitude', 'baseline', 'alpha', 'delta_wide', 'lower_bound_range']:
        pars[(p, 'narrow')] = estimated_parameters[p]
        pars[(p, 'wide')] = estimated_parameters[p]

    pars.columns = pd.MultiIndex.from_tuples(pars.columns, names=['parameter', 'range'])
    
    return pars.stack('range').reorder_levels(['range', 'source'], axis=0).sort_index()

def get_paradigm(sub, fit_responses=False):
    behavior = sub.get_behavioral_data(session=None)

    if fit_responses:
        paradigm = behavior[['response', 'range']].rename(columns={'response':'x'})
        paradigm['x'] = paradigm['x'].fillna(paradigm['x'].mean())
    else:
        paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})   

    paradigm['range'] = paradigm['range'].map({'narrow':False, 'wide':True})
    paradigm = paradigm[['x', 'range']]
    paradigm = paradigm.astype(np.float32)

    return paradigm

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', debug=False, roi='NPCr',
         fit_responses=False):

    max_n_iterations = 100 if debug else 2000

    # Create target folder
    key = f'model{model_label}'

    if smoothed:
        key += '.smoothed'

    if fit_responses:
        key += '.fit_responses'

    target_dir = op.join(bids_folder, 'derivatives', 'encoding_models2', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, fit_responses=fit_responses)

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
    parser.add_argument('--fit_responses', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug,
         fit_responses=args.fit_responses)
