import argparse
from neural_priors.utils.data import Subject
from fit_model import get_paradigm, get_model, fit_model, fit_model_cv
from pathlib import Path
import numpy as np
from braincoder.utils import get_rsq
import pandas as pd
import os
import os.path as op
from braincoder.optimize import ParameterFitter
from braincoder.models import AlphaGaussianPRF
from neural_priors.encoding_model.models import AlphaDeltaModel
from braincoder.optimize import ResidualFitter
import pingouin as pg

def get_decoding_paradigm(sub, fit_responses=False, drop_levels=True):
        # Get paradigm/data/model
    paradigm = get_paradigm(sub, fit_responses=fit_responses)
    paradigm = paradigm.set_index(pd.Index((paradigm.index.get_level_values('run') - 1) % 4 + 1, name='run2'), append=True)
    paradigm.index = paradigm.index.swaplevel('run', 'run2')
    paradigm = paradigm.astype(np.float32)
    
    if drop_levels:
        paradigm = paradigm.droplevel(['run', 'trial_nr', 'subject'])
    
    return paradigm

def main(subject, model_label=3, roi='NPCr', bids_folder='/data/ds-neural_priors', smoothed=True, debug=False, fit_responses=False,
         n_voxels=100, spherical_noise=False, separate_sigmas=False):

    assert model_label in [15, 18, 31], 'Only model 15 and 18 are supported for decoding'


    sub = Subject(subject_id=subject, bids_folder=bids_folder)
    bids_folder = Path(bids_folder)

    max_n_iterations = 100 if debug else 2000

    # Create target folder
    key = f'model{model_label}'

    if smoothed:
        key += '.smoothed'

    if spherical_noise:
        key += '.spherical_noise'

    if fit_responses:
        key += '.fit_responses'

    if separate_sigmas:
        key += '.separate_sigmas'

    target_dir = bids_folder / 'derivatives' / 'decoding2' / key / f'sub-{subject}' / 'func'

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    paradigm = get_decoding_paradigm(sub, fit_responses=fit_responses)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    # all_cvr2 = []

    stimulus_range = np.sort(paradigm['x'].unique())
    stimulus_range = np.stack([np.repeat(stimulus_range, 2), np.stack(np.tile([0, 1], len(stimulus_range)), axis=0)], axis=1)
    stimulus_mi = pd.MultiIndex.from_arrays(stimulus_range.T, names=['n', 'range'])

    pdfs = []

    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run2']):

        print(f'Fitting using session {test_session} run {test_run} as test set')

        test_data, test_paradigm = data.loc[(test_session, test_run)].copy().astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        print(test_paradigm)

        # Get model
        model = get_model(model_label)

        # Cross-validate to get number of (and which) voxels
        if n_voxels == 0:

            print('Cross-validating to get number of voxels')

            cvr2 = fit_model_cv(train_data, train_paradigm, model_label, max_n_iterations=max_n_iterations)
            print(cvr2)
            r2_mask = cvr2 > 0.0

            target_fn = op.join(target_dir, f'sub-{subject}_ses-{test_session}_run2-{test_run}_mask-{roi}_desc-cvr2_pars.tsv')
            cvr2.to_csv(target_fn, sep='\t')

            print(f'Selecting {np.sum(r2_mask)} voxels with cvr2 > 0.0')

            print(train_data)
            train_data = train_data.loc[:, r2_mask]
            print(train_data)
            test_data = test_data.loc[:, r2_mask]
            gd_pars = fit_model(model_label, model, train_data.loc[:, r2_mask], train_paradigm, max_n_iterations=max_n_iterations)        

        else:

            gd_pars = fit_model(model_label, model, train_data, train_paradigm, max_n_iterations=max_n_iterations)        

            pred = model.predict(paradigm=train_paradigm, parameters=gd_pars)

            r2 = get_rsq(train_data, pred)
            print(r2.describe())

            r2 = r2[r2 < 1.0]
            r2_mask = r2.sort_values(ascending=False).index[:n_voxels]

            gd_pars = gd_pars.loc[r2_mask]
            model.apply_mask(r2_mask)

            train_data = train_data[r2_mask]
            test_data = test_data[r2_mask]


        if separate_sigmas:

            train_narrow_ix = train_paradigm['range'] == 0.0
            train_wide_ix = train_paradigm['range'] == 1.0

            train_data_narrow = train_data[train_narrow_ix]
            train_data_wide = train_data[train_wide_ix]
            train_paradigm_narrow = train_paradigm[train_narrow_ix]
            train_paradigm_wide = train_paradigm[train_wide_ix]

            
            test_narrow_ix = test_paradigm['range'] == 0.0
            test_wide_ix = test_paradigm['range'] == 1.0
            test_data_narrow = test_data[test_narrow_ix]
            test_data_wide = test_data[test_wide_ix]


            stimulus_range_narow = stimulus_range[:len(stimulus_range) // 2]
            stimulus_range_wide = stimulus_range[len(stimulus_range) // 2:]

            # Get pdf for narrow condition
            model.init_pseudoWWT(stimulus_range_narow, gd_pars)

            residfit_narrow = ResidualFitter(model, train_data_narrow,
                                            train_paradigm_narrow, parameters=gd_pars,)
            omega_narrow, dof_narrow = residfit_narrow.fit(init_sigma2=0.1,
                    init_dof=10.0,
                    method='t',
                    learning_rate=0.05,
                    spherical=spherical_noise,
                    max_n_iterations=20000 if not debug else 100)

            print('DOF narrow', dof_narrow)
            pdf_narrow = model.get_stimulus_pdf(test_data_narrow, stimulus_range,
                    gd_pars,
                    omega=omega_narrow,
                    dof=dof_narrow,
                    normalize=False)
            

            # Get pdf for wide condition
            model.init_pseudoWWT(stimulus_range_wide, gd_pars)
            residfit_wide = ResidualFitter(model, train_data_wide,
                                            train_paradigm_wide, parameters=gd_pars,)
            omega_wide, dof_wide = residfit_wide.fit(init_sigma2=0.1,
                    init_dof=10.0,
                    method='t',
                    learning_rate=0.05,
                    spherical=spherical_noise,
                    max_n_iterations=20000 if not debug else 100)

            print('DOF wide', dof_wide)
            pdf_wide = model.get_stimulus_pdf(test_data_wide, stimulus_range,
                    gd_pars,
                    omega=omega_wide,
                    dof=dof_wide,
                    normalize=False)

            pdf = np.zeros((len(test_data), len(stimulus_range)),  dtype=np.float32)

            pdf[test_narrow_ix, :] = pdf_narrow
            pdf[test_wide_ix, :] = pdf_wide

            pdf = pd.DataFrame(pdf, index=test_data.index, columns=stimulus_mi)
            pdfs.append(pdf)

        else:
            model.init_pseudoWWT(stimulus_range, gd_pars)

            residfit = ResidualFitter(model, train_data,
                                        train_paradigm, parameters=gd_pars,)

            omega, dof = residfit.fit(init_sigma2=0.1,
                    init_dof=10.0,
                    method='t',
                    learning_rate=0.05,
                    spherical=spherical_noise,
                    max_n_iterations=20000 if not debug else 100)

            print('DOF', dof)

            pdf = model.get_stimulus_pdf(test_data, stimulus_range,
                    gd_pars,
                    omega=omega,
                    dof=dof,
                    normalize=False)

            pdf.columns = stimulus_mi
            print(pdf)
            pdfs.append(pdf)

    pdfs = pd.concat(pdfs)        

    target_fn = op.join(target_dir, f'sub-{subject}_mask-{roi}_nvoxels-{n_voxels}_pars.tsv')
    pdfs.to_csv(target_fn, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--model_label', default=3, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--n_voxels', type=int)
    parser.add_argument('--fit_responses', action='store_true')
    parser.add_argument('--separate_sigmas', action='store_true',
                        help='Whether to fit separate sigmas for narrow and wide conditions')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--spherical_noise', action='store_true')
    args = parser.parse_args()

    main(subject=args.subject, model_label=args.model_label, bids_folder=args.bids_folder, smoothed=args.smoothed, debug=args.debug, fit_responses=args.fit_responses, n_voxels=args.n_voxels, spherical_noise=args.spherical_noise, separate_sigmas=args.separate_sigmas)
