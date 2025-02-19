import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
import numpy as np
from braincoder.utils import get_rsq
import pandas as pd
from models import get_paradigm, get_model, fit_model, get_conditionspecific_parameters
from nilearn import image
from braincoder.optimize import ParameterFitter

def main(subject, smoothed, source_model=4, bids_folder='/data/ds-neuralpriors', gaussian=True, debug=False):

    if source_model != 4:
        raise NotImplementedError("Only source_model=4 is implemented")


    target_model = 8

    max_n_iterations = 100 if debug else 1000

    # Create target folder
    key = 'encoding_model'
    key += f'.refined_model{source_model}'

    if gaussian:
        key += '.gaussian'
    else:
        key += '.logspace'

    if smoothed:
        key += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, target_model, gaussian=gaussian)
    print(paradigm)

    data_image = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_brain_mask(session=None, epi_space=True, return_masker=True, debug_mask=debug)
    data = pd.DataFrame(masker.fit_transform(data_image), index=paradigm.index).astype(np.float32)
    # print(data)

    init_pars = sub.get_prf_parameters_volume(smoothed=smoothed,
                                              model_label=source_model,
                                              gaussian=gaussian,
                                              include_cvr2=False,
                                              return_image=True)

    init_pars = masker.fit_transform(init_pars)

    init_pars = pd.DataFrame(init_pars).T.iloc[:, :8]
    # print(init_pars)
    init_pars.columns = pd.MultiIndex.from_product([['mu', 'sd', 'amplitude', 'baseline'], ['narrow', 'wide']])

    # Get model
    model = get_model(paradigm, target_model, gaussian=gaussian)

    if (source_model == 4) & (target_model == 8):
        target_pars = [('mu', 'narrow'), ('mu', 'wide'), ('sd', 'narrow'), ('sd', 'wide'),
                       ('amplitude', 'narrow'), ('baseline', 'narrow'), ]

        init_pars = init_pars[target_pars]
        init_pars.columns = model.parameter_labels

    else:
        raise NotImplementedError(f"Source model {source_model} to target model {target_model} is not implemented")

    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    # print(model. parameter_labels)
    
    # print(init_pars.describe())


    gd_pars = optimizer.fit(init_pars=init_pars, learning_rate=.01, store_intermediate_parameters=False,
                            max_n_iterations=max_n_iterations,
                r2_atol=0.00001)
    # Fit model
    # pars = fit_model(model, paradigm, data, target_model, max_n_iterations=max_n_iterations, gaussian=gaussian)

    # print(pars)
    pred = model.predict(parameters=gd_pars, paradigm=paradigm)
    r2 = get_rsq(data, pred)

    # print(r2)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(r2).to_filename(target_fn)

    conditions = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    pars = model.get_conditionspecific_parameters(conditions, gd_pars)

    for range_n, values in pars.groupby('range'):
        for par, value in values.T.iterrows():
            target_fn = op.join(target_dir, f'sub-{subject}_desc-{par}.{range_n}.optim_space-T1w_pars.nii.gz')
            masker.inverse_transform(value).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--source_model', default=4, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--log_space', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, source_model=args.source_model, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug, gaussian=not args.log_space)
