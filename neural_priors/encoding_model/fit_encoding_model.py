import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
import numpy as np
from braincoder.utils import get_rsq
import pandas as pd
from models import get_paradigm, get_model, fit_model, get_conditionspecific_parameters

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', gaussian=True, debug=False):

    max_n_iterations = 100 if debug else 1000

    # Create target folder
    key = 'encoding_model'
    key += f'.model{model_label}'

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
    paradigm = get_paradigm(sub, model_label, gaussian=gaussian)

    print(paradigm)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_brain_mask(session=None, epi_space=True, return_masker=True, debug_mask=debug)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    # Get model
    model = get_model(paradigm, model_label, gaussian=gaussian)
    
    # Fit model
    pars = fit_model(model, paradigm, data, model_label, max_n_iterations=max_n_iterations, gaussian=gaussian)

    print(pars)
    pred = model.predict(parameters=pars, paradigm=paradigm)
    r2 = get_rsq(data, pred)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(r2).to_filename(target_fn)

    pars = get_conditionspecific_parameters(model_label, model, pars, gaussian=gaussian)

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
    parser.add_argument('--log_space', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug, gaussian=not args.log_space)
