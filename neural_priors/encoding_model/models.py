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

def get_paradigm(sub, model_label):
    behavior = sub.get_behavioral_data(session=None)

    if model_label in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
        paradigm['range'] = (paradigm['range'] == 'wide')
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    if model_label in[3, 4]:
        paradigm['beta'] = paradigm['range'].map({False:1, True:2})
        paradigm.drop('range', axis=1, inplace=True)
    elif model_label in [5]:
        paradigm['beta'] = paradigm['range'].map({False:1, True:2})

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
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    if gaussian:
        if model_label in [1, 2, 6, 7, 8, 9]:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
        elif model_label in [3, 4, 5]:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors, baseline_parameter_values={'mu':10})
    else:
        raise NotImplementedError("Only Gaussian PRF is implemented")


    return model


def fit_model(model, paradigm, data, model_label, max_n_iterations=1000):

    if model_label in [1, 7]:
        modes = np.linspace(5, 45, 41, dtype=np.float32)
        sigmas = np.linspace(3, 30, 30, dtype=np.float32)
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)
    elif model_label in [2, 8]:
        modes = np.linspace(5, 45, 13, dtype=np.float32)
        sigmas = np.linspace(3, 30, 13, dtype=np.float32)
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)
    elif model_label in [3, 5]:
        # From (10 + ) 0 to 15 (so from 10 to 25)
        modes = np.linspace(0, 15, 41, dtype=np.float32)
        sigmas = np.linspace(3, 30, 30, dtype=np.float32)
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)
    elif model_label in [4]:
        # From (10 + ) 0 to 15 (so from 10 to 25)
        modes = np.linspace(0, 15, 41, dtype=np.float32)
        sigmas = np.linspace(3, 30, 30, dtype=np.float32)
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)
    elif model_label in [6, 9]:
        modes = np.linspace(5, 45, 16, dtype=np.float32)
        sigmas = np.linspace(3, 15, 15, dtype=np.float32)
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)
    
    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    if model_label in [1, 3, 4, 6, 8, 9]:
        
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

def get_conditionspecific_parameters(model_label, model, estimated_parameters):
    
    if model_label in [1,2, 6, 7, 8, 9]:
        conditions = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    elif model_label in [3, 4]:
        conditions = pd.DataFrame({'beta':[1,2]}, index=pd.Index(['narrow', 'wide'], name='range'))
    elif model_label in [5]:
        conditions = pd.DataFrame({'beta':[1,2], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")
        
    pars = model.get_conditionspecific_parameters(conditions, estimated_parameters)

    return pars