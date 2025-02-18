from neural_priors.utils.data import Subject
from braincoder.models import RegressionGaussianPRF
from braincoder.optimize import ParameterFitter
import os.path as op
import numpy as np
import pandas as pd

'''
1. 4-parameter model (everything same)
2. 8-parameter model (everything different)
3. Model A, 4-parameters (mu_wide = 10 + 2* (mu_narrow  - 10))
4. Model B, 4-parameters (mu_wide = 10 + 2* (mu_narrow  - 10), sd_wide = sd_narrow * 2)
5. Model C, 7-parameters (mu_wide = 10 + 2* (mu_narrow  - 10), everything else free)
6. Model D, 5-parameters (mu free, everything else fixed)
7. Model E, 7-parameters (mu is the same across two conditions)
8. Model F, 6-parameters (mu and sd free, everything else fixed)
9. Model F, 4-parameters (sd free, everything else fixed)
'''

range_increase_natural_space = (40 - 10) / (25 - 10) # (2)
range_increase_log_space = (np.log(40) - np.log(10)) / (np.log(25) - np.log(10)) # (2)

def get_paradigm(sub, model_label, gaussian=True):
    behavior = sub.get_behavioral_data(session=None)

    range_increase = range_increase_log_space if not gaussian else range_increase_natural_space

    if model_label in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
        paradigm['range'] = (paradigm['range'] == 'wide')

        if not gaussian:
            paradigm['x'] = np.log(paradigm['x'])
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    if model_label in[3, 4, 10]:

        paradigm['beta'] = paradigm['range'].map({False:1, True:range_increase})

        paradigm.drop('range', axis=1, inplace=True)

    elif model_label in [5]:
        paradigm['beta'] = paradigm['range'].map({False:1, True:range_increase})

    paradigm = paradigm.astype(np.float32)

    return paradigm

def get_model(paradigm, model_label, gaussian=True):

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
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    if model_label in [1, 2, 6, 7, 8, 9, 10]:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
    elif model_label in [3, 4, 5]:
        if gaussian:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors, baseline_parameter_values={'mu':10})
        else:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors, baseline_parameter_values={'mu':np.log(10)})

    return model

def get_parameter_grids(model_label, gaussian=True):
    """Returns modes, sigmas, amplitudes, and baselines based on model_label and gaussian flag."""
    
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    # Define mode and sigma ranges
    model_params = {
        (1, 7,): (5, 45, 41, 3, 30, 30),
        (2, 8, 9): (5, 45, 13, 3, 30, 13),
        (5,): (0, 15, 16, 3, 30, 15),  # Special case for log-space
        (3, 4,):   (0, 15, 41, 3, 15, 30),  # Special case for log-space
        (6,10):   (5, 45, 16, 3, 15, 15),
    }

    for labels, (mode_min, mode_max, mode_steps, sigma_min, sigma_max, sigma_steps) in model_params.items():
        if model_label in labels:
            if gaussian:
                modes = np.linspace(mode_min, mode_max, mode_steps, dtype=np.float32)
            else:
                # Special log-space case for models 3, 4, 5
                if model_label in [3, 4, 5]:
                    modes = np.linspace(0, np.log(25) - np.log(10), mode_steps, dtype=np.float32)
                else:
                    modes = np.linspace(np.log(mode_min), np.log(mode_max), mode_steps, dtype=np.float32)

            sigmas = np.linspace(np.log(sigma_min) if not gaussian else sigma_min, 
                                 np.log(sigma_max) if not gaussian else sigma_max, 
                                 sigma_steps, dtype=np.float32)
            
            return modes, sigmas, amplitudes, baselines

    raise ValueError(f"Unknown model_label: {model_label}")

def fit_model(model, paradigm, data, model_label, max_n_iterations=1000, gaussian=True):

    modes, sigmas, amplitudes, baselines = get_parameter_grids(model_label, gaussian)
    
    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    if model_label in [1, 3, 4, 6, 8, 9, 10]:
        
        if model_label == 6:
            grid_pars = optimizer.fit_grid(modes, modes,
                                        sigmas, 
                                            amplitudes, 
                                            baselines)
        elif model_label == 8:
            grid_pars = optimizer.fit_grid(modes, modes,
                                        sigmas, sigmas,
                                            amplitudes, 
                                            baselines)
        elif model_label == 9:
            grid_pars = optimizer.fit_grid(modes, 
                                           sigmas,
                                            sigmas, 
                                            amplitudes, 
                                            baselines)
        else:
            grid_pars = optimizer.fit_grid(modes, 
                                        sigmas, 
                                            amplitudes, 
                                            baselines)

        fixed_pars = list(model.parameter_labels)
        fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'Intercept')))
        fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'Intercept')))

    elif model_label in [2, 5, 7]:
        
        if model_label == 2:
            grid_pars = optimizer.fit_grid(modes, modes,
                                        sigmas, sigmas,
                                            amplitudes, amplitudes,
                                            baselines, baselines)
        elif model_label == 5:
            grid_pars = optimizer.fit_grid(modes, 
                                        sigmas, sigmas,
                                        amplitudes, amplitudes,
                                        baselines, baselines)

        elif model_label == 7:
            grid_pars = optimizer.fit_grid(modes,
                                        sigmas, sigmas,
                                        amplitudes, amplitudes,
                                        baselines, baselines)

        fixed_pars = list(model.parameter_labels)
        fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[0.0]')))
        fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[0.0]')))
        fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[1.0]')))
        fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[1.0]')))

    # Fit one (only baseline/amplitude)
    gd_pars = optimizer.fit(init_pars=grid_pars, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=max_n_iterations,
            fixed_pars=fixed_pars,
        r2_atol=0.001)

    # Fit two
    gd_pars = optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False, max_n_iterations=max_n_iterations,
                r2_atol=0.00001)

    return gd_pars

def get_conditionspecific_parameters(model_label, model, estimated_parameters, gaussian=True):

    range_increase = range_increase_log_space if not gaussian else range_increase_natural_space

    print("Getting parameters with range_increase", range_increase)
    
    if model_label in [1,2, 6, 7, 8, 9]:
        conditions = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    elif model_label in [3, 4, 10]:
        conditions = pd.DataFrame({'beta':[1,range_increase]}, index=pd.Index(['narrow', 'wide'], name='range'))
    elif model_label in [5]:
        conditions = pd.DataFrame({'beta':[1,range_increase], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")
        
    pars = model.get_conditionspecific_parameters(conditions, estimated_parameters)

    return pars
