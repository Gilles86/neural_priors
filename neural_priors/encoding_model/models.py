from neural_priors.utils.data import Subject
from braincoder.models import RegressionGaussianPRF, GaussianPRF
from braincoder.optimize import ParameterFitter
import os.path as op
import numpy as np
import pandas as pd
from braincoder.utils.math import norm
from braincoder.models import AlphaGaussianPRF
import tensorflow as tf
import tensorflow_probability as tfp
import logging

'''
1. 4-parameter model (everything same)
2. 8-parameter model (everything different)
3. Model A, 4-parameters (mu_wide = 10 + 2* (mu_narrow  - 10))
4. Model B, 4-parameters (mu_wide = 10 + 2* (mu_narrow  - 10), sd_wide = sd_narrow * 2)
5. Model C, 7-parameters (mu_wide = 10 + 2* (mu_narrow  - 10), everything else free)
6. Model D, 5-parameters (mu free, everything else fixed)
7. Model E, 7-parameters (mu is the same across two conditions)
8. 6-parameters (mu and sd free, everything else fixed)
9. 5-parameters (sd free, everything else fixed)
10. 4-parameters, only SD follows range adaptation
11. Like model 8, BUT starting points from 11 to 24 and 11 to 39
'''

range_increase_natural_space = (40 - 10) / (25 - 10) # (2)
range_increase_log_space = (np.log(40) - np.log(10)) / (np.log(25) - np.log(10)) # (~1.5)

def get_paradigm(sub, model_label, gaussian=True):
    behavior = sub.get_behavioral_data(session=None)

    range_increase = range_increase_log_space if not gaussian else range_increase_natural_space

    if model_label in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
        paradigm['range'] = (paradigm['range'] == 'wide')

        if not gaussian:
            paradigm['x'] = np.log(paradigm['x'])
    elif model_label in range(12, 18):
        paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
        paradigm['range'] = paradigm['range'].map({'narrow':False, 'wide':True})
        paradigm = paradigm[['x', 'range']]
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    if model_label in[3, 4, 10]:

        paradigm['beta'] = paradigm['range'].map({False:1, True:range_increase})

        paradigm.drop('range', axis=1, inplace=True)

    elif model_label in [5]:
        paradigm['beta'] = paradigm['range'].map({False:1, True:range_increase})

    paradigm = paradigm.astype(np.float32)

    return paradigm


def get_regressors(model_label):
    if model_label == 1:
        regressors = {}
    elif model_label == 2:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline':'0 + C(range)'}
    elif model_label == 3:
        regressors = {'mu':'0 + beta'}
    elif model_label == 4:
        regressors = {'mu':'0 + beta', 'sd':'0 + beta'}
    elif model_label == 5:
        regressors = {'mu':'0 + beta', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline':'0 + C(range)'}
    elif model_label == 6:
        regressors = {'mu':'0 + C(range)'}
    elif model_label == 7:
        regressors = {'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline':'0 + C(range)'}
    elif model_label == 8:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)'}
    elif model_label == 9:
        regressors = {'sd':'0 + C(range)'}
    elif model_label == 10:
        regressors = {'sd':'0 + beta'}
    elif model_label == 11:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)'}
    elif model_label in range(12, 18):
        return None
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    return regressors

def get_model(paradigm, model_label, gaussian=True):

    regressors = get_regressors(model_label)

    if model_label in [1, 2, 6, 7, 8, 9, 10, 11]:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
    elif model_label in [3, 4, 5]:
        if gaussian:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors, baseline_parameter_values={'mu':10})
        else:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors, baseline_parameter_values={'mu':np.log(10)})

    elif model_label in [12, 13]:
        model = GaussianDeltaModel(paradigm=paradigm)

    return model

def get_parameter_grids(model_label, gaussian=True):
    """Returns modes, sigmas, amplitudes, and baselines based on model_label and gaussian flag."""
    
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    def make_grid(min_val, max_val, steps, log_space):
        """Helper function to create either linear or log-spaced grids."""
        if log_space:
            return np.linspace(np.log(min_val), np.log(max_val), steps, dtype=np.float32)
        return np.linspace(min_val, max_val, steps, dtype=np.float32)

    # Special case for model_label 11 (two separate mode grids)
    if model_label == 11:
        modes1 = make_grid(11, 24, int(14/2.), not gaussian)
        modes2 = make_grid(11, 39, int(29/2.), not gaussian)
        # sigmas = make_grid(3, 30, int(15/2.), not gaussian)
        sigmas = np.linspace(2.5, 15, 15) if gaussian else np.linspace(.1, 2., 10)

    # Standard mode/sigma definitions for other models
    model_params = {
        (1, 7,): (5, 45, 41, 3, 30, 30),
        (2, 8, 9): (5, 45, 13, 3, 30, 13),
        (5,): (0, 15, 16, 3, 30, 15),  # Special case for log-space
        (3, 4,):   (0, 15, 41, 3, 15, 30),  # Special case for log-space
        (6,10):   (5, 45, 16, 3, 15, 15),
        (12, 13):   (10, 25, 15, 3, 15, 15),
    }

    for labels, (mode_min, mode_max, mode_steps, sigma_min, sigma_max, sigma_steps) in model_params.items():
        if model_label in labels:
            modes = make_grid(mode_min, mode_max, mode_steps, not gaussian)
            sigmas = make_grid(sigma_min, sigma_max, sigma_steps, not gaussian)
            return modes, sigmas, amplitudes, baselines

    raise ValueError(f"Unknown model_label: {model_label}")

def fit_model(model, paradigm, data, model_label, max_n_iterations=1000, gaussian=True):

    print(paradigm)
    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    # Get parameter grids
    if model_label == 11:
        modes1, modes2, sigmas, amplitudes, baselines = get_parameter_grids(model_label, gaussian)
    else:
        modes, sigmas, amplitudes, baselines = get_parameter_grids(model_label, gaussian)

    # Define grid fitting based on model_label
    model_grid_configs = {
        6: lambda: optimizer.fit_grid(modes, modes, sigmas, amplitudes, baselines),
        8: lambda: optimizer.fit_grid(modes, modes, sigmas, sigmas, amplitudes, baselines),
        9: lambda: optimizer.fit_grid(modes, sigmas, sigmas, amplitudes, baselines),
        11: lambda: optimizer.fit_grid(modes1, modes2, sigmas, sigmas, amplitudes, baselines),
        2: lambda: optimizer.fit_grid(modes, modes, sigmas, sigmas, amplitudes, amplitudes, baselines, baselines),
        5: lambda: optimizer.fit_grid(modes, sigmas, sigmas, amplitudes, amplitudes, baselines, baselines),
        7: lambda: optimizer.fit_grid(modes, sigmas, sigmas, amplitudes, amplitudes, baselines, baselines),
    }

    print("FITTING GRID")
    # Default grid fitting
    grid_pars = model_grid_configs.get(model_label, lambda: optimizer.fit_grid(modes, sigmas, amplitudes, baselines))()

    print(grid_pars.describe())

    # Define fixed parameters
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
        max_n_iterations=max_n_iterations, fixed_pars=fixed_pars, r2_atol=0.001
    )

    # Fit two
    gd_pars = optimizer.fit(
        init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False,
        max_n_iterations=max_n_iterations, r2_atol=0.00001
    )

    print(gd_pars.describe())

    return gd_pars

def get_conditionspecific_parameters(model_label, model, estimated_parameters, gaussian=True):

    range_increase = range_increase_log_space if not gaussian else range_increase_natural_space

    print("Getting parameters with range_increase", range_increase)


    if model_label in range(12,18):

        pars = pd.DataFrame()

        pars[('mu', 'narrow')] = estimated_parameters['mu_narrow']
        pars[('mu', 'wide')] = estimated_parameters['delta_wide'] * (estimated_parameters['mu_narrow'] - estimated_parameters['lower_bound_range']) + estimated_parameters['lower_bound_range']

        for p in ['sd', 'amplitude', 'baseline', 'alpha', 'delta_wide', 'lower_bound_range']:
            pars[(p, 'narrow')] = estimated_parameters[p]
            pars[(p, 'wide')] = estimated_parameters[p]

        pars.columns = pd.MultiIndex.from_tuples(pars.columns, names=['parameter', 'range'])
        
        return pars.stack('range').reorder_levels(['range', 'source'], axis=0).sort_index()


    
    if model_label in [1,2, 6, 7, 8, 9, 11]:
        conditions = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    elif model_label in [3, 4, 10]:
        conditions = pd.DataFrame({'beta':[1,range_increase]}, index=pd.Index(['narrow', 'wide'], name='range'))
    elif model_label in [5]:
        conditions = pd.DataFrame({'beta':[1,range_increase], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")
        
    pars = model.get_conditionspecific_parameters(conditions, estimated_parameters)

    return pars



class AlphaDeltaModel(AlphaGaussianPRF):

    parameter_labels = ['mu_narrow', 'sd', 'alpha', 'delta_wide', 'lower_bound_range', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 **kwargs):

        if allow_neg_amplitudes:
            raise NotImplementedError("Negative amplitudes are not allowed for AlphaDeltaModel")

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)
    @tf.function
    def _transform_parameters_forward2(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 5][:, tf.newaxis]), # Amplitude
                          parameters[:, 6][:, tf.newaxis]], axis=1)
    
    @tf.function
    def _transform_parameters_backward2(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]),
                          parameters[:, 6][:, tf.newaxis]], axis=1)

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):

        def alpha_transform(x, alpha, eps=1e-6):
            """ Computes a numerically stable alpha transformation. """
            return tf.where(
                tf.abs(alpha) < eps,
                tf.math.log(x),  # Directly use log(x) when alpha ≈ 0
                (tf.pow(x, alpha) - 1) / alpha
            )

        def f_x(x, mu_x, sigma_mu, alpha):
            """ Computes p_x(x | mu_x, sigma_x) using the given formula. """
            mu_alpha_x = alpha_transform(x, alpha)  # Using your transformation
            mu_alpha_mu = alpha_transform(mu_x, alpha)  # Using your transformation
            exponent = -tf.square(mu_alpha_x - mu_alpha_mu) / (2 * tf.square(sigma_mu))
            return tf.exp(exponent)

        # Extract stimulus feature values
        x = paradigm[..., tf.newaxis, 0]
        wide_condition = tf.cast(paradigm[..., 1], tf.bool)  # Ensure this is a tensor

        delta_wide = parameters[..., 3]
        lower_bound_range = parameters[..., 4]

        mu_narrow = parameters[..., 0]
        mu_wide = tf.clip_by_value(((mu_narrow - lower_bound_range) * delta_wide) + lower_bound_range, 1e-6, float('inf'))

        mu = tf.where(tf.transpose(wide_condition), mu_wide, mu_narrow)[tf.newaxis, ...]

        return f_x(x,
                   mu,
                    parameters[:, tf.newaxis, :, 1],
                    parameters[:, tf.newaxis, :, 2]) * \
            parameters[:, tf.newaxis, :, 5] + parameters[:, tf.newaxis, :, 6]

class GaussianDeltaModel(GaussianPRF):

    parameter_labels = ['mu_narrow', 'sd', 'delta_wide', 'lower_bound_range', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 **kwargs):

        if allow_neg_amplitudes:
            raise NotImplementedError("Negative amplitudes are not allowed for AlphaDeltaModel")

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)
    @tf.function
    def _transform_parameters_forward2(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 4][:, tf.newaxis]), # Amplitude
                          parameters[:, 5][:, tf.newaxis]], axis=1)
    
    @tf.function
    def _transform_parameters_backward2(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          tfp.math.softplus_inverse(parameters[:, 4][:, tf.newaxis]),
                          parameters[:, 5][:, tf.newaxis]], axis=1)

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):

        # Extract stimulus feature values
        x = paradigm[..., tf.newaxis, 0]
        wide_condition = tf.cast(paradigm[..., 1], tf.bool)  # Ensure this is a tensor

        delta_wide = parameters[..., 2]
        lower_bound_range = parameters[..., 3]

        mu_narrow = parameters[..., 0]
        mu_wide = tf.clip_by_value(((mu_narrow - lower_bound_range) * delta_wide) + lower_bound_range, 1e-6, float('inf'))

        mu = tf.where(tf.transpose(wide_condition), mu_wide, mu_narrow)[tf.newaxis, ...]

        return norm(x,
                    mu,
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 4] + parameters[:, tf.newaxis, :, 5]