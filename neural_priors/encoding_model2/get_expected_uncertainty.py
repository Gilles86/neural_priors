import argparse
from pathlib import Path
from neural_priors.utils.data import Subject
from neural_priors.encoding_model2.fit_model import get_model, get_paradigm
import pandas as pd
import numpy as np
from braincoder.utils import get_rsq
from braincoder.utils.math import get_expected_value, get_sd_posterior
from braincoder.optimize import ResidualFitter

def main(subject, model_label, smoothed, fit_responses, bids_folder, spherical_noise=False, roi='NPCr'):

    sub = Subject(subject, bids_folder=bids_folder)

    pars = sub.get_prf_parameters_volume2(model_label, smoothed=smoothed, roi='NPCr', raw=True)
    cvr2 = sub.get_prf_parameters_volume2(model_label, smoothed=smoothed, roi='NPCr', raw=False, par_keys=[])['cvr2'].squeeze()
    print(cvr2)
    mask = cvr2 > 0.0
    print(mask)
    pars = pars.loc[mask, :]

    # Create target folder
    key = f'model{model_label}'

    if smoothed:
        key += '.smoothed'

    if spherical_noise:
        key += '.spherical_noise'

    if fit_responses:
        key += '.fit_responses'

    target_dir = Path(bids_folder) / 'derivatives' / 'expected_uncertainty' / key / f'sub-{subject:02d}' / 'func'
    target_dir.mkdir(parents=True, exist_ok=True)


    # Get paradigm/data/model
    paradigm = get_paradigm(sub, fit_responses=fit_responses)
    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    data = data.loc[:, mask]


    narrow_stimuli = pd.DataFrame(np.concatenate((np.arange(10, 26, 1)[:, np.newaxis], np.zeros(16)[:, np.newaxis]), axis=1), columns=['n', 'range'])
    wide_stimuli = pd.DataFrame(np.concatenate((np.arange(10, 26, 1)[:, np.newaxis], np.ones(16)[:, np.newaxis]), axis=1), columns=['n', 'range'])
    narrow_stimuli.index.name = 'stimulus'
    wide_stimuli.index.name = 'stimulus'

    narrow_ix = paradigm['range'] == 0.0
    wide_ix = paradigm['range'] == 1.0

    model_narrow = get_model(model_label)
    resid_fitter_narrrow = ResidualFitter(model_narrow, data.loc[narrow_ix], paradigm.loc[narrow_ix], pars)

    model_wide = get_model(model_label)
    resid_fitter_wide = ResidualFitter(model_wide, data.loc[wide_ix], paradigm.loc[wide_ix], pars)

    model_narrow.init_pseudoWWT(narrow_stimuli, pars)
    model_wide.init_pseudoWWT(wide_stimuli, pars)

    omega_narrow, dof_narrow = resid_fitter_narrrow.fit(max_n_iterations=5000, spherical=spherical_noise, method='t', init_dof=10.0,)
    omega_wide, dof_wide = resid_fitter_wide.fit(max_n_iterations=5000, spherical=spherical_noise, method='t', init_dof=10.0,)

    # omega, dof = resid_fitter.fit()

    print(omega_narrow, dof_narrow)
    print(omega_wide, dof_wide)

    simulated_data_narrow = model_narrow.simulate(narrow_stimuli, pars, noise=omega_narrow, dof=dof_narrow, n_repeats=1000)
    simulated_data_wide = model_narrow.simulate(wide_stimuli, pars, noise=omega_wide, dof=dof_wide, n_repeats=1000)

    p_stim_narrow =  model_narrow.get_stimulus_pdf(simulated_data_narrow, parameters=pars, omega=omega_narrow, dof=dof_narrow, stimulus_range=narrow_stimuli, normalize=False).droplevel(1, 1)
    p_stim_wide =  model_wide.get_stimulus_pdf(simulated_data_wide, parameters=pars, omega=omega_wide, dof=dof_wide, stimulus_range=wide_stimuli, normalize=False).droplevel(1, 1)

    E_narrow = get_expected_value(p_stim_narrow, normalize=True).to_frame().join(narrow_stimuli)
    E_wide = get_expected_value(p_stim_wide, normalize=True).to_frame().join(wide_stimuli)

    sd_narrow = get_sd_posterior(p_stim_narrow, normalize=True).to_frame().join(narrow_stimuli)
    sd_wide = get_sd_posterior(p_stim_wide, normalize=True).to_frame().join(wide_stimuli)


    E = pd.concat((E_narrow, E_wide), axis=0, keys=['narrow', 'wide'], names=['range'])
    E['error'] = E['E'] - E['n']
    E['error'] = E['error'].abs()

    sd = pd.concat((sd_narrow, sd_wide), axis=0, keys=['narrow', 'wide'], names=['range'])
    E.to_csv(target_dir / f'sub-{subject:02d}_desc-expected_value.tsv', sep='\t')
    sd.to_csv(target_dir / f'sub-{subject:02d}_desc-sd.tsv', sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Fisher information for encoding model.")
    parser.add_argument("subject", type=int, help="Subject ID")
    parser.add_argument("--model_label", default=15, type=int, help="Model label")
    parser.add_argument("--smoothed", action='store_true', help="Whether the data is smoothed")
    parser.add_argument("--fit_responses", action='store_true', help="Whether to fit responses")
    parser.add_argument("--spherical_noise", action='store_true', help="Spherical noise?")
    parser.add_argument("--bids_folder", default='/data/ds-neuralpriors', type=Path, help="BIDS folder path")

    args = parser.parse_args()
    main(args.subject, args.model_label, args.smoothed, args.fit_responses, args.bids_folder, spherical_noise=args.spherical_noise)