"""Fisher information of the fitted encoding model per numerosity and condition.

Fits a residual noise model to the single-trial data under the fitted nPRF
parameters, then computes the encoding Fisher information I(n) over the
numerosity grid (10-40) for the narrow and wide conditions (Fig. 5a left;
its inverse square root is the plotted imprecision). Writes per-subject TSVs
to derivatives/fisher_information2/.

The production configuration for the paper used --spherical_noise (no
correlations between voxels; see Methods, "fMRI-derived Fisher information").
"""
import argparse
from pathlib import Path
from neural_priors.utils.data import Subject
from neural_priors.encoding_model.fit_model import get_model, get_paradigm
import pandas as pd
import numpy as np
from braincoder.utils import get_rsq
from braincoder.optimize import ResidualFitter

def main(subject, model_label, smoothed, fit_responses, bids_folder, spherical_noise=False, roi='NPCr', separate_sigmas=False):

    sub = Subject(subject, bids_folder=bids_folder)

    pars = sub.get_prf_parameters_volume(model_label, smoothed=smoothed, roi=roi, response_fit=fit_responses, raw=True)
    cvr2 = sub.get_prf_parameters_volume(model_label, smoothed=smoothed, roi=roi, response_fit=fit_responses)['cvr2'].squeeze()
    # Only voxels with positive cross-validated R2 ("signal voxels") enter the
    # noise model and the Fisher-information computation (see Methods).
    mask = cvr2 > 0.0
    pars = pars.loc[mask, :]

    # Create target folder
    key = f'model{model_label}'

    if smoothed:
        key += '.smoothed'

    if spherical_noise:
        key += '.spherical_noise'

    if separate_sigmas:
        key += '.separate_sigmas'

    if fit_responses:
        key += '.fit_responses'

    target_dir = Path(bids_folder) / 'derivatives' / 'fisher_information2' / key / f'sub-{subject:02d}' / 'func'
    target_dir.mkdir(parents=True, exist_ok=True)


    # Get paradigm/data/model
    paradigm = get_paradigm(sub, fit_responses=fit_responses)
    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    data = data.loc[:, mask]


    # pred = model.predict(paradigm, pars)

    # r2 = get_rsq(data, pred)

    # print(r2.sort_values())

    narrow_stimuli = np.concatenate((np.arange(10, 41, 1)[:, np.newaxis], np.zeros(31)[:, np.newaxis]), axis=1)
    wide_stimuli = np.concatenate((np.arange(10, 41, 1)[:, np.newaxis], np.ones(31)[:, np.newaxis]), axis=1)
    stimuli = np.concatenate((narrow_stimuli, wide_stimuli), axis=0).astype(np.float32)

    
    if separate_sigmas:
        narrow_ix = paradigm['range'] == 0.0
        wide_ix = paradigm['range'] == 1.0

        narrow_model = get_model(model_label)
        wide_model = get_model(model_label)

        narrow_model.init_pseudoWWT(narrow_stimuli, pars)
        wide_model.init_pseudoWWT(wide_stimuli, pars)

        resid_fitter_narrow = ResidualFitter(narrow_model, data[narrow_ix], paradigm[narrow_ix], pars)
        resid_fitter_wide = ResidualFitter(wide_model, data[wide_ix], paradigm[wide_ix], pars)

        omega_narrow, dof_narrow = resid_fitter_narrow.fit(spherical=spherical_noise)
        omega_wide, dof_wide = resid_fitter_wide.fit(spherical=spherical_noise)

        pd.DataFrame(omega_narrow).to_csv(target_dir / f'sub-{subject:02d}_desc-omega_narrow.tsv', sep='\t')
        pd.DataFrame(omega_wide).to_csv(target_dir / f'sub-{subject:02d}_desc-omega_wide.tsv', sep='\t')

        pd.Series(resid_fitter_narrow.fitted_omega_parameters).to_csv(target_dir / f'sub-{subject:02d}_desc-omega_parameters_narrow.tsv', sep='\t')
        pd.Series(resid_fitter_wide.fitted_omega_parameters).to_csv(target_dir / f'sub-{subject:02d}_desc-omega_parameters_wide.tsv', sep='\t')

        fisher_information_narrow = narrow_model.get_fisher_information(stimuli, omega_narrow)
        fisher_information_wide = wide_model.get_fisher_information(stimuli, omega_wide)

        fisher_information = pd.concat([fisher_information_narrow, fisher_information_wide], axis=0)

        # fisher_information.rename(columns={'Fisher information': 'fisher_information'}, inplace=True)
        # fisher_information = fisher_information.to_frame('fisher_information')

        fisher_information.name = 'fisher_information'

    else:

        model = get_model(model_label)
        resid_fitter = ResidualFitter(model, data, paradigm, pars)

        model.init_pseudoWWT(stimuli, pars)

        omega, dof = resid_fitter.fit(spherical=spherical_noise)

        pd.DataFrame(omega).to_csv(target_dir / f'sub-{subject:02d}_desc-omega.tsv', sep='\t')
        fisher_information = model.get_fisher_information(stimuli, omega)

        fisher_information = fisher_information.to_frame('fisher_information')

    fisher_information.index.names = ['n', 'wide']
    fisher_information = fisher_information.reset_index()

    fisher_information['condition'] = fisher_information['wide'].map({0: 'narrow', 1: 'wide'})
    fisher_information.drop('wide', axis=1).to_csv(target_dir / f'sub-{subject:02d}_desc-fisher_information.tsv', sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Fisher information for encoding model.")
    parser.add_argument("subject", type=int, help="Subject ID")
    parser.add_argument("--model_label", default=15, type=int, help="Model label")
    parser.add_argument("--smoothed", action='store_true', help="Whether the data is smoothed")
    parser.add_argument("--spherical_noise", action='store_true', help="Whether noise is spherical")
    parser.add_argument("--fit_responses", action='store_true', help="Whether to fit responses")
    parser.add_argument("--bids_folder", default='/data/ds-neuralpriors', type=Path, help="BIDS folder path")
    parser.add_argument("--separate_sigmas", action='store_true', help="Whether to fit separate sigmas for narrow and wide conditions")
    parser.add_argument("--roi", default='NPCr', help="ROI mask to use")

    args = parser.parse_args()
    main(args.subject, args.model_label, args.smoothed, args.fit_responses, args.bids_folder, spherical_noise=args.spherical_noise,
         roi=args.roi, separate_sigmas=args.separate_sigmas)