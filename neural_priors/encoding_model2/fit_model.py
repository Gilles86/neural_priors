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
from neural_priors.encoding_model2.models import AlphaDeltaModel, LinearScalingModel

# Model 0: null model
# Model 1: delta_wide is 2
# model 2: delta_wide is free but same across voxels
# model 3: delta_wide is free and different for each voxel
# model 4: delta_wide is 2 and identity below 10
# model 5: delta_wide is fitted, same across voxels, but identity below 10
# Model 6: Like model 3, gamma free
# Model 7: Like model 4, gamma free
# Model 8: Like model 5, gamma free
# Model 9: Like model 3, baseline fixed at 0
# Model 10: Like model 3, baseline fixed at 0, separate amplitudes
# Model 11: Like model 10, but rescale baseline based on amplitude ratio
# Model 12: model 4 with free amplitude and free baseline”.
# Model 13: jmodel 4 with free amplitude, and r”.
# Model 14: model 4 with free widths.
# Model 15: model 4 with free widths (same slope across voxels)
# Model 16: model 4 with free amplitude (same intercept and slope across voxels)
# Model 17: model 4 with free widths and free amplitudes (same intercept and slope across voxels)

# model 21: model 4 with free amplitudes (no baseline rescaling)
# model 22: model 4 with free sd and free amplitudes (no baseline rescaling
# model 23: like 21, but with free delta_wide
# model 24: like 22, but with free delta_wide
# Model 25: Like model 4, but slope on widths is fixed at sqrt(2)
# Model 26: Like model 4, but sigma is defined in natural space and has same slope across voxels
# Model 27: Like model 4, but sigma is defined in natural space and has slope fixed at 2
# Model 28: Like model 4, but sigma is defined as FHWM natural space and has same slope across voxels
# Model 29: Like model 4, but sigma is defined as FWHM natural space and has slope fixed at 2
# Model 30: Like model 14, but using FWHM in natural space instead of sigma_log

def get_model(model_label):

    if model_label in [4, 5, 7, 8]:
        model = AlphaDeltaModel(identity_below_range=True)
    elif model_label in [10]:
        model = AlphaDeltaModel(separate_amplitudes=True)
    elif model_label in [11]:
        model = AlphaDeltaModel(separate_amplitudes=True, rescale_baseline=True)
    elif model_label in [12]:
        model = AlphaDeltaModel(separate_amplitudes=True, separate_baselines=True, identity_below_range=True)
    elif model_label in [13]:
        model = AlphaDeltaModel(separate_amplitudes=True, rescale_baseline=True, identity_below_range=True)
    elif model_label in [14]:
        model = AlphaDeltaModel(separate_sds=True, identity_below_range=True)
    elif model_label in [15, 18, 25]:
        model = LinearScalingModel(separate_amplitudes=False, identity_below_range=True, separate_sds=True)
    elif model_label in [16, 19]:
        model = LinearScalingModel(separate_amplitudes=True, identity_below_range=True, separate_sds=False, rescale_baseline=True)
    elif model_label in [17, 20]:
        model = LinearScalingModel(separate_amplitudes=True, identity_below_range=True, separate_sds=True, rescale_baseline=True)
    elif model_label in [21, 23]:
        model = LinearScalingModel(separate_amplitudes=True, identity_below_range=True, separate_sds=False, rescale_baseline=False)
    elif model_label in [22, 24]:
        model = LinearScalingModel(separate_amplitudes=True, identity_below_range=True, separate_sds=True, rescale_baseline=False)
    elif model_label in [26, 27]:
        model = LinearScalingModel(separate_amplitudes=False, identity_below_range=True, separate_sds=True, sd_natural=True)
    elif model_label in [28, 29, 30]:
        model = LinearScalingModel(separate_amplitudes=False, identity_below_range=True, separate_sds=True, sigma_fwhm=True)
    else:
        model = AlphaDeltaModel()

    return model

def get_grid(model_label):

    modes = np.linspace(5, 45, 41)
    
    if model_label in [26, 27]:
        sds = np.linspace(2, 30, 30)
    elif model_label in [28, 29, 30]:
        sds = np.linspace(4, 60, 30) # Empirically, FWHM is roughly twice the sigma in natural space
    else:
        sds = np.linspace(np.log(2), np.log(30), 30)

    if model_label in [25]:
        sd_scales = [np.sqrt(2)]

    elif model_label in [27, 29]:
        sd_scales = [2.0]

    else:
        # sd_scales = [1.]
        sd_scales = [.6, .8, 1., 1.2, 1.4, 1.6]

    if model_label in range(6, 9):
        alphas = np.linspace(-1., 1., 5)
    else:
        alphas = np.array([1e-4], dtype=np.float32)

    intersection_point = [10.0]
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    if model_label in [0]:
        delta_wides = [1.0]
    elif model_label in [1, 4, 7, 12, 13, 14, 15, 16, 17, 21, 22, 25, 26, 27, 28, 29, 30]:
        delta_wides = [2.0]
    elif model_label in [2, 3, 5, 6, 8, 9, 10, 11, 18, 19, 20, 23, 24]:
        delta_wides = np.linspace(.3, 3., 10)

    baseline_ratios = [.25, .4, .6, 0.8]

    amplitudes_alpha = [0.0]
    amplitudes_beta = [1.0]

    if model_label < 10:
        return modes, alphas, delta_wides, intersection_point, sds, amplitudes, baselines
    else:
        if model_label in [11]:
            return modes, alphas, delta_wides, intersection_point, sds,  amplitudes, amplitudes, baselines, baseline_ratios
        elif model_label in [12]:
            return modes, alphas, delta_wides, intersection_point, sds,  amplitudes, amplitudes, baselines, baselines
        elif model_label in [13]:
            return modes, alphas, delta_wides, intersection_point, sds,  amplitudes, amplitudes, baselines, baseline_ratios
        elif model_label in [14]:
            return modes, alphas, delta_wides, intersection_point, sds,  sds, amplitudes, baselines
        elif model_label in [15, 18]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude']
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes
        elif model_label in [16, 19]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta', 'baseline_ratio']
            return modes, delta_wides, intersection_point, baselines, sds, amplitudes, amplitudes_alpha, amplitudes_beta, baseline_ratios
        elif model_label in [17, 20]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta', 'baseline_ratio']
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes, amplitudes_alpha, amplitudes_beta, baseline_ratios
        elif model_label in [21, 23]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta', 'baseline_ratio']
            return modes, delta_wides, intersection_point, baselines, sds, amplitudes, amplitudes_alpha, amplitudes_beta
        elif model_label in [22, 24]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta', 'baseline_ratio']
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes, amplitudes_alpha, amplitudes_beta
        elif model_label in [25, 26, 27, 28, 29, 30]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta', 'baseline_ratio']
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes

def fit_model(model_label, model, data, paradigm, max_n_iterations=1000, whole_brain=False):

    # Fit model
    fitter = ParameterFitter(model, data, paradigm)
    grid = get_grid(model_label)

    print(len(grid))
    print(model.parameter_labels)
    print(len(model.parameter_labels))
    print(grid)

    grid_pars = fitter.fit_grid(*grid, use_correlation_cost=True)
    

    if model_label < 9 or model_label == 14:
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

    fixed_pars = []
    shared_pars = []

    if model_label not in [3, 6, 9, 10]:
        fixed_pars += ['lower_bound_range']

    if ((model_label in range(0, 6)) or (model_label > 8)) and ( model_label < 15):
        fixed_pars += ['alpha']
    else:
        shared_pars += ['alpha']

    if model_label in [0, 1, 4, 7, 12, 13, 14]:
        fixed_pars += ['delta_wide']
    elif model_label in [2, 5, 8]:
        shared_pars += ['delta_wide']
    
    if model_label in [9, 10]:
        fixed_pars += ['baseline']
    
    if model_label in [13]:
        shared_pars += ['baseline_ratio']

    # MODELS ABOVE 14 (LinearScalingModel)
    if model_label in [15, 16, 17, 21, 22, 25, 26, 27, 28, 29, 30]:
        fixed_pars += ['delta_wide']

    if model_label in [25, 27, 29]:
        fixed_pars += ['sd_wide_scale']

    if model_label in [15, 17, 18, 20, 22, 24, 26, 28]:
        shared_pars += ['sd_wide_scale']

    if model_label in [16, 17, 19, 20]:
        shared_pars += ['amplitude_alpha', 'amplitude_beta']
        shared_pars += ['baseline_ratio']
    
    if model_label in [21, 22, 23, 24]:
        shared_pars += ['amplitude_alpha', 'amplitude_beta']

    if model_label in [18, 19, 20, 23, 24]:
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

    if (model_label in [4, 5, 7, 8]) or (model_label > 11):
        pars[('mu', 'wide')] = pars[('mu', 'wide')].where(pars[('mu', 'wide')] > 10, pars[('mu', 'narrow')])

    if (model_label > 9) & (model_label < 15):

        par_labels = ['alpha', 'delta_wide', 'lower_bound_range']

        if model_label in [14]:
            par_labels += ['amplitude']
        else:
            pars[('amplitude', 'narrow')] = estimated_parameters['amplitude_narrow']
            pars[('amplitude', 'wide')] = estimated_parameters['amplitude_wide']

        if model_label in [11, 13]:
            par_labels += ['baseline_ratio']

        if model_label in [12]:
            pars[('baseline', 'narrow')] = estimated_parameters['baseline_narrow']
            pars[('baseline', 'wide')] = estimated_parameters['baseline_wide']
        else:
            par_labels += ['baseline']

        if model_label in [14]:
            pars[('sd', 'narrow')] = estimated_parameters['sd_narrow']
            pars[('sd', 'wide')] = estimated_parameters['sd_wide']
        else:
            par_labels += ['sd']

        for p in par_labels:
            pars[(p, 'narrow')] = estimated_parameters[p]
            pars[(p, 'wide')] = estimated_parameters[p]

    elif model_label > 14:

        par_labels = ['delta_wide']

        if model_label in [16, 17, 19, 20]:
            pars[('amplitude', 'narrow')] = estimated_parameters['amplitude_narrow']
            pars[('amplitude', 'wide')] = estimated_parameters['amplitude_alpha'] + estimated_parameters['amplitude_beta'] * estimated_parameters['amplitude_narrow']

            pars[('baseline', 'narrow')] = estimated_parameters['baseline'] - pars[('amplitude', 'narrow')] * estimated_parameters['baseline_ratio']
            pars[('baseline', 'wide')] = estimated_parameters['baseline'] -  pars[('amplitude', 'wide')] * estimated_parameters['baseline_ratio']

            par_labels += ['baseline_ratio']

        elif model_label in [21, 22, 23, 24]:
            pars[('amplitude', 'narrow')] = estimated_parameters['amplitude_narrow']
            pars[('amplitude', 'wide')] = estimated_parameters['amplitude_alpha'] + estimated_parameters['amplitude_beta'] * estimated_parameters['amplitude_narrow']

            par_labels += ['baseline']

        else:
            pars[('amplitude', 'narrow')] = estimated_parameters['amplitude']
            pars[('amplitude', 'wide')] = estimated_parameters['amplitude']

            pars[('baseline', 'narrow')] = estimated_parameters['baseline']
            pars[('baseline', 'wide')] = estimated_parameters['baseline']

        if model_label in [15, 17, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30]:
            pars[('sd', 'narrow')] = estimated_parameters['sd_narrow']
            pars[('sd', 'wide')] = estimated_parameters['sd_wide_scale'] * estimated_parameters['sd_narrow']
        else:
            pars[('sd', 'narrow')] = estimated_parameters['sd']
            pars[('sd', 'wide')] = estimated_parameters['sd'] 

        for p in par_labels:
            pars[(p, 'narrow')] = estimated_parameters[p]
            pars[(p, 'wide')] = estimated_parameters[p]

    else:
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
         fit_responses=False, whole_brain=False):

    max_n_iterations = 100 if debug else 5000

    # Create target folder
    key = f'model{model_label}'

    if whole_brain:
        key += '.whole_brain'

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
    
    if whole_brain:
        masker = sub.get_brain_mask(epi_space=True, return_masker=True)
    else:
        masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    # Get model
    model = get_model(model_label)

    print(model, model.parameter_labels)
    
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
    parser.add_argument('--fit_responses', action='store_true')
    parser.add_argument('--whole_brain', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug,
         fit_responses=args.fit_responses, whole_brain=args.whole_brain)
