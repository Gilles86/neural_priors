"""Recovery of the width-scaling parameter (sd_wide_scale / r_sigma) across a
DENSE sweep of generative values, rather than just the two conditions (1.0
null, 1.287794 empirical) tested in simulate_data_sd.py / Supplementary Fig.
S9. Addresses the follow-up question of whether recovery is unbiased across
the whole plausible range of r_sigma, not only at those two specific points.

Identical simulate-then-refit procedure to simulate_data_sd.py (same model
15, same empirical (mu_narrow, sd_narrow) resampling, same noise=0.8
empirically-matched calibration), just looped over more generative values
with fewer iterations each (this is a diagnostic sweep, not the full
100-iterations-per-condition production analysis).

Deliberately writes to a SEPARATE top-level directory
(simulated_recovery_sd_sweep/, not simulated_recovery_sd/) so it cannot be
picked up by the `simulated_recovery_sd/*/noise_0.8/...` glob used by
plot_recovery_figures.py / plot_mu_below_range_figure.py /
plot_sd_below_range_figure.py -- those figures' populations must stay
exactly as already reported and must not silently grow because of this
sweep.

Writes per-iteration recovered sd_wide_scale to
<bids>/derivatives/simulated_recovery_sd_sweep/sd_scale_{v}/noise_{n}/design_{d}/iteration-{i}_results.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from neural_priors.encoding_model.fit_model import get_model, fit_model


def _wire_lsm_transforms(model):
    if hasattr(model, '_transform_parameters_forward2'):
        model._transform_parameters_forward = model._transform_parameters_forward2
        model._transform_parameters_backward = model._transform_parameters_backward2
    return model


def run_one(sd_wide_scale, design, noise, n_voxels, iteration, pars, bids_folder):
    target_dir = (bids_folder / 'simulated_recovery_sd_sweep' / f'sd_scale_{sd_wide_scale}'
                 / f'noise_{noise}' / f'design_{design}')
    target_dir.mkdir(exist_ok=True, parents=True)

    out_path = target_dir / f'iteration-{iteration}_results.csv'
    if out_path.exists():
        return

    x_narrow = np.repeat(np.arange(10, 25), 10)
    if design == 'full':
        x_wide = np.repeat(np.arange(10, 40), 5)
    else:
        x_wide = np.repeat(np.arange(10, 25), 10)
    paradigm = pd.DataFrame({'x': x_narrow.tolist() + x_wide.tolist(),
                             'range': [0] * len(x_narrow) + [1] * len(x_wide)})

    p = pars.sample(n=n_voxels, random_state=iteration)
    prf_pars = pd.DataFrame({'mu_narrow': p[('mu', 'narrow')].values,
                             'delta_wide': 2.0,
                             'lower_bound_range': 10.0,
                             'baseline': 0.0,
                             'sd_narrow': p[('sd', 'narrow')].values,
                             'sd_wide_scale': sd_wide_scale,
                             'amplitude': 1.0})

    model = _wire_lsm_transforms(get_model(15))
    data = model.simulate(paradigm=paradigm, parameters=prf_pars, noise=noise)

    fitted_pars = fit_model(15, model, data, paradigm)
    recovered = fitted_pars.iloc[0]['sd_wide_scale']

    pd.DataFrame({'iteration': iteration, 'gen_sd_wide_scale': sd_wide_scale,
                 'sd_wide_scale': recovered, 'n_voxels': len(p)}, index=[0]).to_csv(out_path, index=False)
    print(f'sd_wide_scale={sd_wide_scale}, iteration={iteration}: recovered={recovered:.4f}')


def main(sweep, design='full', noise=0.8, n_voxels=250, n_iterations=10, bids_folder='/data/ds-neuralpriors'):
    bids_folder = Path(bids_folder)

    pars = pd.read_csv(bids_folder / 'derivatives' / 'encoding_models' / 'group_roi-NPCr_desc-groundtruth_parameters.tsv',
                       sep='\t', index_col=[0, 1, 2, 3], header=[0, 1])
    pars = pars.xs(15, level='model_label')
    pars = pars[pars[('cvr2', 'nan')] > 0.0]

    jobs = [(v, i) for v in sweep for i in range(n_iterations)]
    for sd_wide_scale, iteration in tqdm(jobs):
        run_one(sd_wide_scale, design, noise, n_voxels, iteration, pars, bids_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep sd_wide_scale recovery across a dense range of generative values')
    parser.add_argument('--sweep', type=str, default='0.6,0.8,1.0,1.2,1.4,1.6,1.8',
                        help='Comma-separated list of generative sd_wide_scale values')
    parser.add_argument('--design', type=str, default='full')
    parser.add_argument('--noise', type=float, default=0.8)
    parser.add_argument('--n_voxels', type=int, default=250)
    parser.add_argument('--n_iterations', type=int, default=10)
    parser.add_argument('--bids_folder', type=str, default='/data/ds-neuralpriors')
    args = parser.parse_args()

    sweep = [float(v) for v in args.sweep.split(',')]
    main(sweep, design=args.design, noise=args.noise, n_voxels=args.n_voxels,
        n_iterations=args.n_iterations, bids_folder=args.bids_folder)
