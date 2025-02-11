import argparse
from neural_priors.utils.data import Subject
import pandas as pd
from braincoder.utils import get_rsq
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter
import numpy as np
from braincoder.optimize import ResidualFitter
import os 
import os.path as op


def main(subject, smoothed, bids_folder, gaussian=True):

    key = 'fisher_information'

    if gaussian:
        key += '.gaussian'

    if smoothed:
        key+= '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')
    os.makedirs(target_dir, exist_ok=True)

    sub = Subject(subject)

    prf_pars_wide = sub.get_prf_parameters_volume(session=None, smoothed=smoothed, gaussian=gaussian, roi='NPCr',
                                            cross_validated=False, range_n='wide')
    prf_pars_narrow = sub.get_prf_parameters_volume(session=None, smoothed=smoothed, gaussian=gaussian, roi='NPCr',
                                                cross_validated=False, range_n='narrow')

    prf_pars = pd.concat((prf_pars_wide, prf_pars_narrow), axis=0,
                        keys=['wide', 'narrow'], names=['range_n'])

    mask = (prf_pars.unstack(-1)['cvr2'].T > 0.0).any(axis=1)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed, roi='NPCr',)
    behavior = sub.get_behavioral_data(session=None).set_index('range', append=True)

    data = pd.DataFrame(data, index=behavior.index)
    data = data.loc[:, mask]

    modes = np.linspace(1, 50, 100, dtype=np.float32)
    sigmas = np.linspace(1, 60, 100, dtype=np.float32) / 2.35
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)
 
    for condition in ['wide', 'narrow']:
        paradigm = behavior.xs(condition, level='range').astype(np.float32)[['n']]

        model = GaussianPRF()

        optimizer = ParameterFitter(model, data.xs(condition, level='range').astype(np.float32), paradigm.astype(np.float32))

        grid_parameters = optimizer.fit_grid(modes, sigmas, amplitudes, baselines, use_correlation_cost=True)

        pred = model.predict(paradigm, grid_parameters)
        r2 = get_rsq(data.xs(condition, level='range').astype(np.float32), pred)


        optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                fixed_pars=['mu', 'sd'],
            r2_atol=0.00001)

        optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.005, store_intermediate_parameters=False, max_n_iterations=10000,
            min_n_iterations=1000,
            r2_atol=0.00001)

        stimulus_range = np.arange(10, 40.5, 1.0, dtype=np.float32)
        model.init_pseudoWWT(stimulus_range, optimizer.estimated_parameters)
        resid_fitter = ResidualFitter(model, data.xs(condition, level='range').astype(np.float32), paradigm.astype(np.float32), parameters=optimizer.estimated_parameters)

        omega, dof = resid_fitter.fit()

        fisher_information = model.get_fisher_information(stimulus_range, omega, dof)

        fisher_information.to_csv(op.join(target_dir, f'sub-{subject}_range-{condition}_fisher_information.tsv'), sep='\t')

        for key in resid_fitter.fitted_omega_parameters.keys():

            if key == 'tau':
                p = pd.Series(resid_fitter.fitted_omega_parameters[key][0], name=key)
            else:
                p =pd.Series([resid_fitter.fitted_omega_parameters[key]], name=key)

            p.to_csv(op.join(target_dir, f'sub-{subject}_range-{condition}_{key}.tsv'), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    args = parser.parse_args()

    main(args.subject, smoothed=args.smoothed, bids_folder=args.bids_folder,
         gaussian=args.gaussian)
