# Standard argparse stuff
import argparse
import pandas as pd
from neural_priors.utils.data import Subject
import numpy as np
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.models import LogGaussianPRF, GaussianPRF
from braincoder.utils.stats import get_rsq
import pingouin as pg
import os
import os.path as op


def main(subject, range_n, smoothed, bids_folder, gaussian, n_voxels=100, roi='NPCr'):


    sub = Subject(subject, bids_folder)

    betas = pd.DataFrame(sub.get_single_trial_estimates(None, roi=roi, smoothed=smoothed))

    behavior = sub.get_behavioral_data()

    betas.index = behavior.index

    key = 'decoding_across'

    if gaussian:
        key += '.gaussian'

    if smoothed:
        key += '.smoothed'

    key += f'.test_range-{range_n}'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    modes = np.linspace(5, 45, 60, dtype=np.float32)
    widths = np.linspace(4, 30, 60, dtype=np.float32)
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    stimulus_range = np.linspace(10, 40, 100, dtype=np.float32)

    pdfs = []

    if range_n not in ['narrow', 'wide', 'both']:
        raise ValueError('range must be one of "narrow", "wide", or "both"')


    if gaussian:
        model = GaussianPRF()
        fixed_pars = ['mu', 'sigm']
        widths = widths / 2.35
    else:
        model = LogGaussianPRF(parameterisation='mode_fwhm_natural')
        fixed_pars = ['mode', 'fwhm']

    keys = []
    pdfs = []


    test_paradigm = behavior[behavior.range == range_n]['n']
    test_data = betas[behavior.range == range_n]

    train_paradigm = behavior[behavior.range != range_n]['n']
    train_data = betas[behavior.range != range_n]

    optimizer = ParameterFitter(model, train_data, train_paradigm)
    grid_parameters = optimizer.fit_grid(modes, widths, amplitudes, baselines, use_correlation_cost=True)

    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
        fixed_pars=fixed_pars,
        r2_atol=0.0001)

    optimizer.fit(init_pars=optimizer.estimated_parameters, learning_rate=.005, store_intermediate_parameters=False, max_n_iterations=10000,
        r2_atol=0.0001, min_n_iterations=1000)

    pars = optimizer.estimated_parameters

    pred = model.predict(train_paradigm, pars)

    r2 = get_rsq(train_data, pred)
    print(r2.describe())

    r2 = r2[r2 < 1.0]
    r2_mask = r2.sort_values(ascending=False).index[:n_voxels]

    train_data = train_data[r2_mask]
    test_data = test_data[r2_mask]

    # model.apply_mask(r2_mask)

    pars = optimizer.estimated_parameters.loc[r2_mask]

    model.init_pseudoWWT(stimulus_range, pars)   

    residfit = ResidualFitter(model, train_data, train_paradigm.astype(np.float32),
                            parameters=pars)

    omega, dof = residfit.fit(init_sigma2=10.0,
            init_dof=10.0,
            method='t',
            learning_rate=0.05,
            max_n_iterations=20000)

    print('DOF', dof)

    bins = stimulus_range.astype(np.float32)

    pdf = model.get_stimulus_pdf(test_data, bins,
            model.parameters,
            omega=omega,
            dof=dof)


    pdf /= np.trapz(pdf, bins, axis=1)[:, np.newaxis]

    E = np.trapz(pdf.columns * pdf, pdf.columns, axis=1)

    print(pg.corr(E, test_paradigm))

    pdf.to_csv(op.join(target_dir, f'sub-{subject}_desc-decoding_across.pdfs.tsv'), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('range', default=None, type=str)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--gaussian', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.range, smoothed=args.smoothed, bids_folder=args.bids_folder,
         gaussian=args.gaussian)