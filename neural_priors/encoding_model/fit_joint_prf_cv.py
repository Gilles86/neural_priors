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

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', gaussian=True,):


    if model_label == 1:
        regressors = {'mu':' 0 + C(range)'}
    elif model_label == 2:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)'}
    elif model_label == 3:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)'}
    elif model_label == 4:
        regressors = {'mu':'0 + C(range)', 'amplitude':'0 + C(range)'}
    elif model_label == 5:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline': '0+C(range)'}

    else:
        raise ValueError('Only model 1/2/3 supported')

    key = 'encoding_model.joint.cv'
    key += f'.model{model_label}'

    if gaussian:
        key += '.gaussian'

    if smoothed:
        key += '.smoothed'

    sub = Subject(subject, bids_folder=bids_folder)
    behavior = sub.get_behavioral_data(session=None)

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    fn_template = op.join(target_dir, 'sub-{subject}_session-{session}_run2-{run}_desc-{par}.{optimizer}_space-T1w_pars.nii.gz')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    paradigm = behavior[['n', 'range']].rename(columns={'n':'x'}).droplevel('subject')
    paradigm['range'] = (paradigm['range'] == 'wide')
    paradigm = paradigm.set_index(pd.Index((paradigm.index.get_level_values('run') - 1) % 4 + 1, name='run2'), append=True)
    paradigm.index = paradigm.index.swaplevel('run', 'run2')
    paradigm = paradigm.astype(np.float32).droplevel(['run', 'trial_nr'])
    

    masker = sub.get_brain_mask(session=None, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    if gaussian:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
    else:
        raise NotImplementedError("Only Gaussian PRF is implemented")


    modes = np.linspace(5, 45, 30, dtype=np.float32)
    sigmas = np.linspace(1, 60, 30, dtype=np.float32)
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    cv_r2s = []
    keys = []
    
    print(data)
    print(paradigm)

    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run2']):

        test_data, test_paradigm = data.loc[(test_session, test_run)].copy().astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        model.set_paradigm(train_paradigm, regressors=regressors)

        optimizer = ParameterFitter(model, train_data.astype(np.float32), train_paradigm.astype(np.float32))

        if model_label == 1:
            grid_parameters = optimizer.fit_grid(modes, modes, sigmas, amplitudes, baselines, use_correlation_cost=True)
        elif model_label == 2:
            grid_parameters = optimizer.fit_grid(modes[::2], modes[::2], sigmas[::2], sigmas[::2], amplitudes, baselines, use_correlation_cost=True)
        elif model_label == 3:
            grid_parameters = optimizer.fit_grid(modes[::2], modes[::2], sigmas[::2], sigmas[::2], amplitudes, amplitudes, baselines, use_correlation_cost=True)
        elif model_label == 4:
            grid_parameters = optimizer.fit_grid(modes, modes, sigmas, amplitudes, amplitudes, baselines, use_correlation_cost=True)
        elif model_label == 5:
            grid_parameters = optimizer.fit_grid(modes[::2], modes[::2], sigmas[::2], sigmas[::2], amplitudes, amplitudes, baselines, baselines, use_correlation_cost=True)

        fixed_pars = list(model.parameter_labels)
        if model_label in [1, 2]:
            fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'Intercept')))
        elif model_label in [3, 4, 5]:
            fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[0.0]')))
            fixed_pars.pop(fixed_pars.index(('amplitude_unbounded', 'C(range)[1.0]')))
        
        if model_label in [1,2,3,4]:
            fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'Intercept')))
        elif model_label in [5]:
            fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[0.0]')))
            fixed_pars.pop(fixed_pars.index(('baseline_unbounded', 'C(range)[1.0]')))

        optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                fixed_pars=fixed_pars,
            r2_atol=0.0001)

        optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False, max_n_iterations=10000,
                    r2_atol=0.00001)

        
        dummy_paradigm = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
        pars = model.get_transformed_parameters(dummy_paradigm, optimizer.estimated_parameters)

        target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run2-{test_run}_desc-r2.optim_space-T1w_pars.nii.gz')
        masker.inverse_transform(optimizer.r2).to_filename(target_fn)

        for range_n, values in pars.groupby('range'):

            for par, value in values.T.iterrows():
                target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run2-{test_run}_desc-{range_n}.{par}.optim_space-T1w_pars.nii.gz')
                masker.inverse_transform(value).to_filename(target_fn)

        model.set_paradigm(test_paradigm, regressors=regressors)
        cv_r2 = get_rsq(test_data, model.predict(paradigm=test_paradigm, parameters=optimizer.estimated_parameters))

        target_fn = fn_template.format(subject=subject, session=test_session, run=test_run, par='cvr2', optimizer='optim')
        masker.inverse_transform(cv_r2).to_filename(target_fn)

        print(f'{cv_r2.isnull().sum()} voxels have a NaN value in the cv_r2 map')

        cv_r2s.append(cv_r2)
        keys.append((test_session, test_run))

    cv_r2 = pd.concat(cv_r2s, keys=keys, names=['session', 'run2']).groupby(level=2, axis=0).mean()
    target_fn = op.join(target_dir, f'sub-{subject}_desc-cvr2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(cv_r2).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--model', type=int, default=1)
    args = parser.parse_args()

    main(args.subject, smoothed=args.smoothed, bids_folder=args.bids_folder, model_label=args.model)
