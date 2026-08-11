import braincoder
from neural_priors.encoding_model.fit_model import LinearScalingModel, get_paradigm, AlphaDeltaModel, fit_model
import argparse
from neural_priors.utils.data import Subject
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np


def main(design='full', start_iteration=0, n_iterations=100, delta_wide=2.0, bids_folder='/data/ds-neuralpriors'):

    assert design in ['full', 'censored'], "Design must be either 'full' or 'censored'"

    print(f'N ITERATIONS: {n_iterations}')

    target_dir = Path(bids_folder) / 'simulated_recovery' / f'delta_wide_{delta_wide}' / f'design_{design}'

    target_dir.mkdir(exist_ok=True, parents=True)

    bids_folder = Path(bids_folder)
    # NB: these monolithic all-model TSVs are a legacy snapshot that no current
    # script regenerates (extract_pars.py now writes per-model files to
    # derivatives/extracted_pars/); they only exist on the machine where the
    # recovery simulations were originally run.
    pars_ground_truth = pd.read_csv(bids_folder / 'derivatives' / 'encoding_models' / 'group_roi-NPCr_desc-groundtruth_parameters.tsv', sep='\t', index_col=[0, 1, 2, 3], header=[0, 1, ])
    pars_estimates = pd.read_csv(bids_folder / 'derivatives' / 'encoding_models' / 'group_roi-NPCr_desc-responses_parameters.tsv', sep='\t', index_col=[0, 1, 2, 3], header=[0, 1, ])

    pars = pd.concat([pars_ground_truth, pars_estimates], keys=['ground_truth', 'estimates'], names=['type'], axis=0)

    # rename index level subject_id to subject
    pars.index.set_names('subject', level='subject_id', inplace=True)

    pars.index.unique(level='smoothed')
    pars = pars.xs(3, level='model_label').xs('ground_truth', level='type')
    pars = pars[pars[('cvr2', 'nan')] > 0.0]

    iterations = list(range(start_iteration, start_iteration + n_iterations))

    print(f'Running {n_iterations} iterations starting from iteration {start_iteration} with delta_wide={delta_wide} and design={design}')

    for iteration in tqdm(iterations):
        
        model = AlphaDeltaModel(identity_below_range=True)

        p = pars.sample(n=250)

        mu_narrow = p[('mu', 'narrow')]
        lower_bound_range = 10
        baseline = 0.0
        sd = p[('sd', 'narrow')]

        prf_pars = pd.DataFrame({'mu_narrow': mu_narrow, 'delta_wide': delta_wide, 'lower_bound_range': lower_bound_range, 'baseline': baseline, 'sd': sd, 'amplitude':1.0, 'alpha':1.0})

        x_narrow = np.repeat(np.arange(10, 25), 10)
        
        if design == 'full':
            x_wide = np.repeat(np.arange(10, 40), 5)
        elif design == 'censored':
            x_wide = np.repeat(np.arange(10, 25), 10)

        paradigm = pd.DataFrame({'x': np.repeat(x_narrow.tolist() + x_wide.tolist(), 1), 'range': [0] * len(x_narrow) + [1] * len(x_wide)})

        data = model.simulate(paradigm=paradigm, parameters=prf_pars, noise=.5)

        fitted_pars = fit_model(5, model, data, paradigm)

        # NOTE: do not rebind `delta_wide` here — it is the *generating* value
        # for every iteration in this run. An earlier version overwrote it with
        # the recovered estimate, so iterations after the first simulated from
        # the previous iteration's estimate instead of the requested value.
        estimated_delta_wide = fitted_pars.iloc[0]['delta_wide']

        print(f'Estimated delta_wide in iteration {iteration}: {estimated_delta_wide}')

        # Write results to file
        results = pd.DataFrame({'iteration': iteration, 'delta_wide': estimated_delta_wide}, index=[0])
        results.to_csv(target_dir / f'iteration-{iteration}_results.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate data and fit model to recover parameters')
    parser.add_argument('--design', type=str, default='full', help="Design of the experiment: 'full' or 'censored'")
    parser.add_argument('--start_iteration', type=int, default=0, help='Starting iteration number')
    parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations to run')
    parser.add_argument('--delta_wide', type=float, default=2.0, help='Value of delta_wide to use in simulations')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-neuralpriors', help='Path to the BIDS folder containing the data')

    args = parser.parse_args()

    main(design=args.design, start_iteration=args.start_iteration, n_iterations=args.n_iterations, delta_wide=args.delta_wide, bids_folder=args.bids_folder)