from neural_priors.utils.data import Subject
from braincoder.models import RegressionGaussianPRF
from braincoder.optimize import ParameterFitter
import os.path as op
import numpy as np
import pandas as pd


def get_paradigm(sub, model_label):
    behavior = sub.get_behavioral_data(session=None)

    if model_label in [1, 2]:
        paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
        paradigm['range'] = (paradigm['range'] == 'wide')
        paradigm = paradigm.astype(np.float32)
    else:
        raise NotImplementedError("Only model 1 is implemented")

    return paradigm

def get_model(paradigm, model_label, gaussian=True):

    if model_label == 1:
        regressors = {}
    elif model_label == 2:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline':'0 + C(range)'}
    else:
        raise NotImplementedError("Only model 1 is implemented")

    if gaussian:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
    else:
        raise NotImplementedError("Only Gaussian PRF is implemented")

    return model


def fit_model(model, paradigm, data, model_label, max_n_iterations=1000):

    if model_label in [1]:
        modes = np.linspace(5, 45, 41, dtype=np.float32)
        sigmas = np.linspace(1, 30, 30, dtype=np.float32)
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)
    elif model_label in [1, 2]:
        modes = np.linspace(5, 45, 15, dtype=np.float32)
        sigmas = np.linspace(1, 30, 15, dtype=np.float32)
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)
    
    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    if model_label == 1:
        grid_pars = optimizer.fit_grid(modes, 
                                       sigmas, 
                                        amplitudes, 
                                        baselines)
        fixed_pars = list(model.parameter_labels)
        fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'Intercept')))
        fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'Intercept')))

    elif model_label == 2:
        grid_pars = optimizer.fit_grid(modes, modes,
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
        r2_atol=0.0001)

    # Fit two
    gd_pars = optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False, max_n_iterations=max_n_iterations,
                r2_atol=0.00001)

    return gd_pars

def get_conditionspecific_parameters(model, estimated_parameters):
    conditions = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    pars = model.get_conditionspecific_parameters(conditions, estimated_parameters)

    return pars