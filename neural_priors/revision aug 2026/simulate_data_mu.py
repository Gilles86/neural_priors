"""Parameter-recovery simulation for the shift parameter (delta_wide) — SFig 8.

Clean reimplementation of `revision feb 2026/simulate_data.py` (which had a
bug where the generative `delta_wide` was overwritten by the recovered
estimate after the first iteration), extended with the same options as the
width-recovery simulation (`simulate_data_sd.py`):

  --sample_subject   one random subject per iteration, all of their
                     supra-threshold voxels (instead of 250 pooled voxels)
  --noise            Gaussian noise SD (0.5 = original SFig 8 setting;
                     0.8 matches the empirical median single-trial R^2)

Generative model and fit: AlphaDeltaModel with identity below the range
(model 5: per-voxel mu/sd/amplitude/baseline, one shared freely fitted
delta_wide, alpha and lower_bound_range fixed), exactly as in SFig 8.
Ground-truth (mu, sd) pairs are resampled from the empirical model-3 fits
in NPCr (voxels with cvr2 > 0).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from neural_priors.encoding_model.fit_model import get_model, fit_model


def _wire_transforms(model):
    # AlphaDeltaModel stores its 7-parameter transforms as _forward2/_backward2;
    # point the standard names ParameterFitter uses at them (see simulate_data_sd.py).
    if hasattr(model, '_transform_parameters_forward2'):
        model._transform_parameters_forward = model._transform_parameters_forward2
        model._transform_parameters_backward = model._transform_parameters_backward2
    return model


def main(design='full', start_iteration=0, n_iterations=10, delta_wide=2.0,
         noise=0.5, n_voxels=250, sample_subject=False, bids_folder='/data/ds-neuralpriors'):

    assert design in ['full', 'censored'], "Design must be either 'full' or 'censored'"

    bids_folder = Path(bids_folder)

    design_dir = f'design_{design}_subjectwise' if sample_subject else f'design_{design}'
    target_dir = bids_folder / 'simulated_recovery_mu' / f'delta_wide_{delta_wide}' / f'noise_{noise}' / design_dir
    target_dir.mkdir(exist_ok=True, parents=True)

    pars = pd.read_csv(bids_folder / 'derivatives' / 'encoding_models' / 'group_roi-NPCr_desc-groundtruth_parameters.tsv',
                       sep='\t', index_col=[0, 1, 2, 3], header=[0, 1])

    # Realistic (mu, sd) combinations: empirical model-3 fits (as in SFig 8), well-fitting voxels only
    pars = pars.xs(3, level='model_label')
    pars = pars[pars[('cvr2', 'nan')] > 0.0]

    iterations = list(range(start_iteration, start_iteration + n_iterations))

    print(f'Running {n_iterations} iterations starting from iteration {start_iteration} '
          f'with delta_wide={delta_wide}, noise={noise}, design={design}, sample_subject={sample_subject}')

    x_narrow = np.repeat(np.arange(10, 25), 10)

    if design == 'full':
        x_wide = np.repeat(np.arange(10, 40), 5)
    elif design == 'censored':
        x_wide = np.repeat(np.arange(10, 25), 10)

    paradigm = pd.DataFrame({'x': x_narrow.tolist() + x_wide.tolist(),
                             'range': [0] * len(x_narrow) + [1] * len(x_wide)})

    for iteration in tqdm(iterations):

        model = _wire_transforms(get_model(5))

        if sample_subject:
            rng = np.random.RandomState(iteration)
            subject = rng.choice(pars.index.unique('subject_id'))
            p = pars.xs(subject, level='subject_id', drop_level=False)
        else:
            subject = None
            p = pars.sample(n=n_voxels, random_state=iteration)

        prf_pars = pd.DataFrame({'mu_narrow': p[('mu', 'narrow')].values,
                                 'alpha': 1.0,
                                 'delta_wide': delta_wide,
                                 'lower_bound_range': 10.0,
                                 'sd': p[('sd', 'narrow')].values,
                                 'amplitude': 1.0,
                                 'baseline': 0.0})

        data = model.simulate(paradigm=paradigm, parameters=prf_pars, noise=noise)

        fitted_pars = fit_model(5, model, data, paradigm)

        estimated_delta_wide = fitted_pars.iloc[0]['delta_wide']

        print(f'Estimated delta_wide in iteration {iteration}: {estimated_delta_wide}')

        results = pd.DataFrame({'iteration': iteration, 'delta_wide': estimated_delta_wide,
                                'subject': subject, 'n_voxels': len(p)}, index=[0])
        results.to_csv(target_dir / f'iteration-{iteration}_results.csv', index=False)

        pervoxel = pd.DataFrame({'iteration': iteration,
                                 'gen_mu_narrow': prf_pars['mu_narrow'].values,
                                 'gen_sd': prf_pars['sd'].values,
                                 'est_mu_narrow': fitted_pars['mu_narrow'].values,
                                 'est_sd': fitted_pars['sd'].values,
                                 'est_amplitude': fitted_pars['amplitude'].values,
                                 'est_baseline': fitted_pars['baseline'].values})
        pervoxel.to_csv(target_dir / f'iteration-{iteration}_pervoxel.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate data and fit model 5 to recover the shift parameter')
    parser.add_argument('--design', type=str, default='full', help="Design of the experiment: 'full' or 'censored'")
    parser.add_argument('--start_iteration', type=int, default=0, help='Starting iteration number')
    parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations to run')
    parser.add_argument('--delta_wide', type=float, default=2.0, help='Generative shift parameter')
    parser.add_argument('--noise', type=float, default=0.5, help='Standard deviation of the Gaussian noise added to simulated responses')
    parser.add_argument('--n_voxels', type=int, default=250, help='Number of simulated voxels per iteration (ignored with --sample_subject)')
    parser.add_argument('--sample_subject', action='store_true',
                        help='Sample one random subject per iteration and use all their supra-threshold voxels, instead of pooling voxels across subjects')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-neuralpriors', help='Path to the BIDS folder')

    args = parser.parse_args()

    main(design=args.design, start_iteration=args.start_iteration, n_iterations=args.n_iterations,
         delta_wide=args.delta_wide, noise=args.noise, n_voxels=args.n_voxels,
         sample_subject=args.sample_subject, bids_folder=args.bids_folder)
