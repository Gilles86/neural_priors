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
from neural_priors.encoding_model.models import AlphaDeltaModel, LinearScalingModel

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
# Model 18: delta_wide is fitted, identity below 10, width is scaled with r_sigma. # mu, baseline, sd_narrow, amplitude + 2 shared (delta_wide, sd_wide_scale)

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
# Model 31: like model 25, but slope is fixed at 1.287794, whcih is the mean of the slopes across subjects according to model 15
# Model 32: like model 31, but amplitude is allowed to vary, for every voxel individually
# Model 33: Like model 31, but amplitude is allowed to vary, with subject-specific slope
# Model 34: all parameters fixed except for amplitude
# Model 35: all parameters fixed except for amplitude, same ratio across voxels
# Model 36: like model 35 but only a shared scaling factor (amplitude_alpha fixed at 0, no intercept)


# Model 115 - model 145, these are all versions of model 31, buth the slope is changed between 1.15 and 1.45

# Number of free parameters per voxel (per-voxel free + shared counted as 1 + sigma for noise).
# Used for AIC/BIC computation: AIC = 2k - 2*LL, BIC = ln(n)*k - 2*LL
N_PARAMETERS = {
    -1: 2,   # mean + sigma
    0:  5,   # mu, sd, amp, baseline + sigma  (lb, alpha, delta_wide fixed)
    1:  5,   # same as 0 (delta_wide=2 fixed)
    2:  6,   # + shared delta_wide
    3:  7,   # mu, delta_wide, lb, sd, amp, baseline + sigma
    4:  5,   # same as 1 with identity_below_range
    5:  6,   # + shared delta_wide
    14: 6,   # mu, sd_narrow, sd_wide, amp, baseline + sigma
    15: 6,   # mu, baseline, sd_narrow, amp + shared sd_wide_scale + sigma
    18: 7,   # mu, baseline, sd_narrow, amp + shared delta_wide, sd_wide_scale + sigma
    31: 5,   # mu, baseline, sd_narrow, amp + sigma  (delta_wide, sd_wide_scale fixed)
    32: 6,   # mu, baseline, sd_narrow, amp_narrow, amp_beta + sigma
    33: 6,   # mu, baseline, sd_narrow, amp_narrow + shared amp_beta + sigma
    34: 7,   # mu, lb, baseline, sd, amp_narrow, amp_beta + sigma
    35: 8,   # mu, lb, baseline, sd, amp_narrow + shared amp_alpha, amp_beta + sigma
    36: 7,   # mu, lb, baseline, sd, amp_narrow + shared amp_beta + sigma
}

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
    elif (model_label in [15, 18, 25, 31]) or (model_label > 100):
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
    elif model_label in [32, 33]:
        model = LinearScalingModel(separate_amplitudes=True, identity_below_range=True, separate_sds=True)
    elif model_label in [34, 35, 36]:
        model = LinearScalingModel(separate_amplitudes=True, identity_below_range=True, separate_sds=False, rescale_baseline=False)
    else:
        model = AlphaDeltaModel()

    return model

def get_grid(model_label):

    modes = np.linspace(5, 45, 41)
    
    if model_label in [26, 27]: # sigma in natural space
        sds = np.linspace(2, 30, 30)
    elif model_label in [28, 29, 30]: # FWHM in natural space
        sds = np.linspace(4, 60, 30) # Empirically, FWHM is roughly twice the sigma in natural space
    else:                       # sigma in log space
        sds = np.linspace(np.log(2), np.log(30), 30)

    if model_label in [25]:
        sd_scales = [np.sqrt(2)]

    elif model_label in [31, 32, 33]:
        sd_scales = [1.287794]
    elif model_label > 100:
        sd_scales = [float(model_label)/100.]


    elif model_label in [27, 29]:
        sd_scales = [2.0]

    elif model_label in [28, 30]:
        sd_scales = [.6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
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

    if model_label in [0, 34, 35]:
        delta_wides = [1.0]
    elif (model_label in [1, 4, 7, 12, 13, 14, 15, 16, 17, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33]) or (model_label > 100):
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
        elif model_label in [22, 24]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta']
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes, amplitudes_alpha, amplitudes_beta
        elif model_label in [25, 26, 27, 28, 29, 30, 31]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta']
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes
        elif model_label > 100:
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes
        elif model_label in [32, 33]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'sd_wide_scale', 'amplitude_narrow', 'amplitude_alpha', 'amplitude_beta', 'baseline_ratio']
            return modes, delta_wides, intersection_point, baselines, sds, sd_scales, amplitudes, amplitudes_alpha, amplitudes_beta
        elif model_label in [34, 35, 36]: # ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline', 'sd_narrow', 'amplitude']
            return modes, delta_wides, intersection_point, baselines, sds, amplitudes, amplitudes_alpha, amplitudes_beta

        

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
        if model_label in [6, 7, 8]:
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
    if model_label in [15, 16, 17, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33] or (model_label > 100):
        fixed_pars += ['delta_wide']

    if model_label in ([25, 27, 29, 31, 32, 33]) or (model_label > 100):
        fixed_pars += ['sd_wide_scale']

    if model_label in [15, 17, 18, 20, 22, 24, 26, 28]:
        shared_pars += ['sd_wide_scale']

    if model_label in [16, 17, 19, 20]:
        shared_pars += ['amplitude_alpha', 'amplitude_beta']
        shared_pars += ['baseline_ratio']
    
    if model_label in [21, 22, 23, 24]:
        shared_pars += ['amplitude_alpha', 'amplitude_beta']

    if model_label in [32, 33]:
        fixed_pars += ['amplitude_alpha']

    if model_label in [33]:
        shared_pars += ['amplitude_beta']

    if model_label in [18, 19, 20, 23, 24]:
        shared_pars += ['delta_wide']

    if model_label in [34]:
        fixed_pars = ['delta_wide', 'amplitude_alpha']

    if model_label in [35]:
        shared_pars = ['amplitude_alpha', 'amplitude_beta']
        fixed_pars = ['delta_wide']

    if model_label in [36]:
        shared_pars = ['amplitude_beta']
        fixed_pars = ['delta_wide', 'amplitude_alpha']

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

def conditionspecific_to_raw_pars(model_label, pars_cs):
    """
    Inverse of get_conditionspecific_parameters.

    Reconstructs raw optimizer parameter DataFrame (columns = parameter_labels)
    from the condition-specific (narrow/wide) DataFrame loaded by
    get_prf_parameters_volume (MultiIndex columns: (parameter, range)).

    For parameters that are shared across voxels (amplitude_alpha/beta,
    baseline_ratio for certain models) the values are recovered via linear
    regression or are read from the file where they were stored as a
    constant per-voxel column.

    Parameters
    ----------
    model_label : int
    pars_cs : pd.DataFrame
        MultiIndex columns (parameter, range).

    Returns
    -------
    pd.DataFrame  –  columns matching the model's parameter_labels.
    """
    from scipy.stats import linregress

    raw = pd.DataFrame(index=pars_cs.index)
    raw['mu_narrow'] = pars_cs[('mu', 'narrow')]

    # ------------------------------------------------------------------
    # AlphaDeltaModel  (models 0 – 14)
    # All of alpha, delta_wide, lower_bound_range are stored in files.
    # ------------------------------------------------------------------
    if model_label <= 14:
        raw['alpha']             = pars_cs[('alpha', 'narrow')]
        raw['delta_wide']        = pars_cs[('delta_wide', 'narrow')]
        raw['lower_bound_range'] = pars_cs[('lower_bound_range', 'narrow')]

        # sd
        if model_label == 14:          # separate_sds
            raw['sd_narrow'] = pars_cs[('sd', 'narrow')]
            raw['sd_wide']   = pars_cs[('sd', 'wide')]
        else:
            raw['sd'] = pars_cs[('sd', 'narrow')]

        # amplitude
        if model_label in [10, 11, 12, 13]:   # separate_amplitudes
            raw['amplitude_narrow'] = pars_cs[('amplitude', 'narrow')]
            raw['amplitude_wide']   = pars_cs[('amplitude', 'wide')]
        else:
            raw['amplitude'] = pars_cs[('amplitude', 'narrow')]

        # baseline
        if model_label == 12:          # separate_baselines
            raw['baseline_narrow'] = pars_cs[('baseline', 'narrow')]
            raw['baseline_wide']   = pars_cs[('baseline', 'wide')]
        else:
            raw['baseline'] = pars_cs[('baseline', 'narrow')]

        if model_label in [11, 13]:    # rescale_baseline – stored as shared column
            raw['baseline_ratio'] = pars_cs[('baseline_ratio', 'narrow')]

    # ------------------------------------------------------------------
    # LinearScalingModel  (models 15+)
    # lower_bound_range is never written to files → always 10.
    # ------------------------------------------------------------------
    else:
        raw['delta_wide']        = pars_cs[('delta_wide', 'narrow')]
        raw['lower_bound_range'] = 10.

        # sd
        if model_label in ([15, 17, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]) or (model_label > 100):
            raw['sd_narrow']     = pars_cs[('sd', 'narrow')]
            raw['sd_wide_scale'] = pars_cs[('sd', 'wide')] / pars_cs[('sd', 'narrow')]
        else:
            raw['sd'] = pars_cs[('sd', 'narrow')]

        # amplitude + baseline
        if model_label in [16, 17, 19, 20]:
            # rescale_baseline: baseline_narrow = baseline - amp_narrow * baseline_ratio
            # → baseline = baseline_narrow + amp_narrow * baseline_ratio
            raw['baseline_ratio'] = pars_cs[('baseline_ratio', 'narrow')]
            raw['baseline'] = (pars_cs[('baseline', 'narrow')]
                               + pars_cs[('amplitude', 'narrow')] * pars_cs[('baseline_ratio', 'narrow')])
            raw['amplitude_narrow'] = pars_cs[('amplitude', 'narrow')]
            # amplitude_alpha/beta are shared → recover via linregress
            slope, intercept, *_ = linregress(
                pars_cs[('amplitude', 'narrow')].values,
                pars_cs[('amplitude', 'wide')].values)
            raw['amplitude_alpha'] = intercept
            raw['amplitude_beta']  = slope

        elif model_label in [32, 34, 36]:
            # amplitude_alpha = 0 fixed; amplitude_beta = wide / narrow (per voxel for 32/34, shared for 36)
            raw['baseline']         = pars_cs[('baseline', 'narrow')]
            raw['amplitude_narrow'] = pars_cs[('amplitude', 'narrow')]
            raw['amplitude_alpha']  = 0.
            raw['amplitude_beta']   = pars_cs[('amplitude', 'wide')] / pars_cs[('amplitude', 'narrow')]

        elif model_label in [21, 22, 23, 24, 33, 35]:
            # amplitude_alpha/beta shared across voxels → linregress
            raw['baseline']         = pars_cs[('baseline', 'narrow')]
            raw['amplitude_narrow'] = pars_cs[('amplitude', 'narrow')]
            slope, intercept, *_ = linregress(
                pars_cs[('amplitude', 'narrow')].values,
                pars_cs[('amplitude', 'wide')].values)
            raw['amplitude_alpha'] = intercept
            raw['amplitude_beta']  = slope

        else:
            # single amplitude: models 15, 18, 25–31, >100
            raw['baseline']   = pars_cs[('baseline', 'narrow')]
            raw['amplitude']  = pars_cs[('amplitude', 'narrow')]

    return raw


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

        elif model_label in [21, 22, 23, 24, 32, 33, 34, 35, 36]:
            pars[('amplitude', 'narrow')] = estimated_parameters['amplitude_narrow']
            pars[('amplitude', 'wide')] = estimated_parameters['amplitude_alpha'] + estimated_parameters['amplitude_beta'] * estimated_parameters['amplitude_narrow']

            par_labels += ['baseline']

        else:
            pars[('amplitude', 'narrow')] = estimated_parameters['amplitude']
            pars[('amplitude', 'wide')] = estimated_parameters['amplitude']

            pars[('baseline', 'narrow')] = estimated_parameters['baseline']
            pars[('baseline', 'wide')] = estimated_parameters['baseline']

        if model_label in ([15, 17, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]) or (model_label > 100):
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
         fit_responses=False, whole_brain=False, censored=False):

    max_n_iterations = 100 if debug else 5000

    # Create target folder
    key = f'model{model_label}'

    if whole_brain:
        key += '.whole_brain'

    if censored:
        key += '.censored'

    if smoothed:
        key += '.smoothed'

    if fit_responses:
        key += '.fit_responses'

    target_dir = op.join(bids_folder, 'derivatives', 'encoding_models', key, f'sub-{subject}', 'func')

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

    if censored:
        data = data[paradigm['x'] < 26]
        paradigm = paradigm[paradigm['x'] < 26]

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
    parser.add_argument('--censored', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug,
         fit_responses=args.fit_responses, whole_brain=args.whole_brain, censored=args.censored)