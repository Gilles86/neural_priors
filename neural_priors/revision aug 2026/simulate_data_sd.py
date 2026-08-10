"""Parameter-recovery simulation for the tuning-width scaling parameter (sd_wide_scale).

Analogous to simulate_data.py (which does this for the shift parameter delta_wide,
Fig. S8), but for the width of the numerosity tuning curves:

  1. Sample realistic per-voxel ground-truth parameters (mu_narrow, sd_narrow)
     from the empirical model-15 fits in NPCr (voxels with cvr2 > 0).
  2. Simulate responses with the LinearScalingModel using a *known* width
     scaling factor sd_wide_scale (e.g. 1.0 = no widening, 1.29 = empirical
     group value) and full shift (delta_wide = 2).
  3. Refit model 15 (per-voxel mu/sd/amplitude/baseline, shared free
     sd_wide_scale, fixed delta_wide = 2) and record the recovered
     sd_wide_scale, as well as the per-voxel recovered mu/sd.

Noise model: model.simulate() adds i.i.d. Gaussian noise with standard
deviation `--noise` (default 0.5) to every simulated response; tuning-curve
amplitude is 1 and baseline 0, so noise=0.5 corresponds to a peak
signal-to-noise ratio of 2.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from neural_priors.encoding_model.fit_model import get_model, fit_model


def _wire_lsm_transforms(model):
    # LinearScalingModel stores its 7-parameter transforms as _forward2/_backward2;
    # point the standard names ParameterFitter uses at them (see archive/gp_prior/fit_gp_prior.py).
    if hasattr(model, '_transform_parameters_forward2'):
        model._transform_parameters_forward = model._transform_parameters_forward2
        model._transform_parameters_backward = model._transform_parameters_backward2
    return model


def main(design='full', start_iteration=0, n_iterations=10, sd_wide_scale=1.287794,
         noise=0.5, n_voxels=250, sample_subject=False, bids_folder='/data/ds-neuralpriors'):

    assert design in ['full', 'censored'], "Design must be either 'full' or 'censored'"

    bids_folder = Path(bids_folder)

    design_dir = f'design_{design}_subjectwise' if sample_subject else f'design_{design}'
    target_dir = bids_folder / 'simulated_recovery_sd' / f'sd_scale_{sd_wide_scale}' / f'noise_{noise}' / design_dir
    target_dir.mkdir(exist_ok=True, parents=True)

    pars = pd.read_csv(bids_folder / 'derivatives' / 'encoding_models' / 'group_roi-NPCr_desc-groundtruth_parameters.tsv',
                       sep='\t', index_col=[0, 1, 2, 3], header=[0, 1])

    # Realistic (mu, sd) combinations: empirical model-15 fits, well-fitting voxels only
    pars = pars.xs(15, level='model_label')
    pars = pars[pars[('cvr2', 'nan')] > 0.0]

    iterations = list(range(start_iteration, start_iteration + n_iterations))

    print(f'Running {n_iterations} iterations starting from iteration {start_iteration} '
          f'with sd_wide_scale={sd_wide_scale}, noise={noise} and design={design}')

    x_narrow = np.repeat(np.arange(10, 25), 10)

    if design == 'full':
        x_wide = np.repeat(np.arange(10, 40), 5)
    elif design == 'censored':
        x_wide = np.repeat(np.arange(10, 25), 10)

    paradigm = pd.DataFrame({'x': x_narrow.tolist() + x_wide.tolist(),
                             'range': [0] * len(x_narrow) + [1] * len(x_wide)})

    for iteration in tqdm(iterations):

        model = _wire_lsm_transforms(get_model(15))

        if sample_subject:
            # Mirror the real per-subject model-15 fit: one random subject,
            # all of their supra-threshold voxels (29-535 across subjects)
            rng = np.random.RandomState(iteration)
            subject = rng.choice(pars.index.unique('subject_id'))
            p = pars.xs(subject, level='subject_id', drop_level=False)
        else:
            subject = None
            p = pars.sample(n=n_voxels, random_state=iteration)

        prf_pars = pd.DataFrame({'mu_narrow': p[('mu', 'narrow')].values,
                                 'delta_wide': 2.0,
                                 'lower_bound_range': 10.0,
                                 'baseline': 0.0,
                                 'sd_narrow': p[('sd', 'narrow')].values,
                                 'sd_wide_scale': sd_wide_scale,
                                 'amplitude': 1.0})

        data = model.simulate(paradigm=paradigm, parameters=prf_pars, noise=noise)

        fitted_pars = fit_model(15, model, data, paradigm)

        recovered_sd_wide_scale = fitted_pars.iloc[0]['sd_wide_scale']

        print(f'Estimated sd_wide_scale in iteration {iteration}: {recovered_sd_wide_scale}')

        results = pd.DataFrame({'iteration': iteration, 'sd_wide_scale': recovered_sd_wide_scale,
                                'subject': subject, 'n_voxels': len(p)}, index=[0])
        results.to_csv(target_dir / f'iteration-{iteration}_results.csv', index=False)

        # Per-voxel ground truth vs recovered parameters (for mu/sd trade-off checks)
        pervoxel = pd.DataFrame({'iteration': iteration,
                                 'gen_mu_narrow': prf_pars['mu_narrow'].values,
                                 'gen_sd_narrow': prf_pars['sd_narrow'].values,
                                 'est_mu_narrow': fitted_pars['mu_narrow'].values,
                                 'est_sd_narrow': fitted_pars['sd_narrow'].values,
                                 'est_amplitude': fitted_pars['amplitude'].values,
                                 'est_baseline': fitted_pars['baseline'].values})
        pervoxel.to_csv(target_dir / f'iteration-{iteration}_pervoxel.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate data and fit model 15 to recover the width scaling parameter')
    parser.add_argument('--design', type=str, default='full', help="Design of the experiment: 'full' or 'censored'")
    parser.add_argument('--start_iteration', type=int, default=0, help='Starting iteration number')
    parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations to run')
    parser.add_argument('--sd_wide_scale', type=float, default=1.287794, help='Generative width scaling factor')
    parser.add_argument('--noise', type=float, default=0.5, help='Standard deviation of the Gaussian noise added to simulated responses')
    parser.add_argument('--n_voxels', type=int, default=250, help='Number of simulated voxels per iteration (ignored with --sample_subject)')
    parser.add_argument('--sample_subject', action='store_true',
                        help='Sample one random subject per iteration and use all their supra-threshold voxels, instead of pooling voxels across subjects')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-neuralpriors', help='Path to the BIDS folder')

    args = parser.parse_args()

    main(design=args.design, start_iteration=args.start_iteration, n_iterations=args.n_iterations,
         sd_wide_scale=args.sd_wide_scale, noise=args.noise, n_voxels=args.n_voxels,
         sample_subject=args.sample_subject, bids_folder=args.bids_folder)
