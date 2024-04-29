import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter
import numpy as np
from braincoder.utils import get_rsq
from nilearn import image
import pandas as pd

def main(subject, session, smoothed, bids_folder, on_response=False, range_n=None):

    if session == 0:
        session = None

    key = 'encoding_model.denoise'

    if smoothed:
        key += '.smoothed'

    if on_response:
        key += '.on_response'

    if range_n is not None:
        key += f'.range_{range_n}'

    sub = Subject(subject, bids_folder=bids_folder)
    behavior = sub.get_behavioral_data(session=session, tasks=['estimation_task', ])

    assert(range_n in [None, 'wide', 'narrow']), "range_n must be either None, 'wide' or 'narrow'"

    if range_n is not None:
        range_mask = behavior['range'] == range_n

    if session is None:
        target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')
    else:
        target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', f'ses-{session}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)


    st_type = 'response' if on_response else 'stim'

    data = sub.get_single_trial_estimates(session, smoothed=smoothed, type=st_type)
    paradigm = behavior['n']

    if range_n is not None:
        data = image.index_img(data, range_mask)
        paradigm = paradigm.loc[range_mask]

    masker = sub.get_brain_mask(session=session, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)

    model = LogGaussianPRF()

    mus = np.linspace(5, 40, 30, dtype=np.float32)
    sds = np.linspace(5, 15, 30, dtype=np.float32)
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)
    grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=5)

    pred = model.predict(paradigm, grid_parameters)
    r2 = get_rsq(data, pred)

    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
        r2_atol=0.00001)

    if session is None:
        target_fn = op.join(target_dir, f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz')
    else:
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-T1w_pars.nii.gz')

    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    for par, values in optimizer.estimated_parameters.T.iterrows():
        print(values)

        if session is None:
            target_fn = op.join(target_dir, f'sub-{subject}_desc-{par}.optim_space-T1w_pars.nii.gz')
        else:
            target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')

        masker.inverse_transform(values).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('session', type=int)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--on_response', action='store_true')
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--range', default=None)
    args = parser.parse_args()

    main(args.subject, args.session, smoothed=args.smoothed, bids_folder=args.bids_folder, on_response=args.on_response,
         range_n=args.range)