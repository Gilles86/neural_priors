import argparse
from pathlib import Path
from neural_priors.utils.data import Subject
from neural_priors.encoding_model.fit_model import get_model, get_paradigm
import pandas as pd
import numpy as np
from braincoder.utils import get_rsq
from braincoder.utils.math import get_expected_value, get_sd_posterior
from braincoder.optimize import ResidualFitter

def main(subject, model_label, smoothed, fit_responses, bids_folder, spherical_noise=False, wide=False, roi='NPCr'):

    sub = Subject(subject, bids_folder=bids_folder)

    pars = sub.get_prf_parameters_volume(model_label, smoothed=smoothed, roi='NPCr', raw=True)
    cvr2 = sub.get_prf_parameters_volume(model_label, smoothed=smoothed, roi='NPCr', raw=False, par_keys=[])['cvr2'].squeeze()
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

    if wide:
        key += '.wide'

    target_dir = Path(bids_folder) / 'derivatives' / 'expected_uncertainty' / key / f'sub-{subject:02d}' / 'func'
    target_dir.mkdir(parents=True, exist_ok=True)


    # Get paradigm/data/model
    paradigm = get_paradigm(sub, fit_responses=fit_responses)
    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    data = data.loc[:, mask]


    if wide:
        narrow_stimuli = pd.DataFrame(np.concatenate((np.arange(10, 41, 1)[:, np.newaxis], np.zeros(31)[:, np.newaxis]), axis=1), columns=['n', 'range'])
        wide_stimuli = pd.DataFrame(np.concatenate((np.arange(10, 41, 1)[:, np.newaxis], np.ones(31)[:, np.newaxis]), axis=1), columns=['n', 'range'])
    else:
        narrow_stimuli = pd.DataFrame(np.concatenate((np.arange(10, 26, 1)[:, np.newaxis], np.zeros(16)[:, np.newaxis]), axis=1), columns=['n', 'range'])
        wide_stimuli = pd.DataFrame(np.concatenate((np.arange(10, 26, 1)[:, np.newaxis], np.ones(16)[:, np.newaxis]), axis=1), columns=['n', 'range'])

    narrow_stimuli.index.name = 'stimulus'
    wide_stimuli.index.name = 'stimulus'

    model = get_model(model_label)
    model.init_pseudoWWT(narrow_stimuli, pars)

    # resid_fitter = ResidualFitter(model, data, paradigm, stimuli, masker=masker, fit_responses=fit_responses)
    resid_fitter = ResidualFitter(model, data, paradigm, pars)

    omega, dof = resid_fitter.fit(max_n_iterations=5000, spherical=spherical_noise, method='t', init_dof=10.0,)


    simulated_data_narrow = model.simulate(narrow_stimuli, pars, noise=omega, dof=dof, n_repeats=20000)
    simulated_data_wide = model.simulate(wide_stimuli, pars, noise=omega, dof=dof, n_repeats=20000)


    p_stim_narrow =  model.get_stimulus_pdf(simulated_data_narrow, parameters=pars, omega=omega, dof=dof, stimulus_range=narrow_stimuli, normalize=False).droplevel(1, 1)
    p_stim_wide =  model.get_stimulus_pdf(simulated_data_wide, parameters=pars, omega=omega, dof=dof, stimulus_range=wide_stimuli, normalize=False).droplevel(1, 1)

    E_narrow = get_expected_value(p_stim_narrow, normalize=True).to_frame().join(narrow_stimuli)
    E_wide = get_expected_value(p_stim_wide, normalize=True).to_frame().join(wide_stimuli)


    E = pd.concat((E_narrow, E_wide), axis=0, keys=['narrow', 'wide'], names=['range']).drop(columns=['range'])
    E['error'] = E['E'] - E['n']
    E['abs(error)'] = E['error'].abs()

    mean_E = E.groupby(['range', 'n'])['E'].mean().to_frame("mean_E")
    mean_error = E.groupby(['range', 'n'])['error'].mean().to_frame("mean_error")
    var_E = E.groupby(['range', 'n'])['E'].var().to_frame("var_E")
    abs_error = E.groupby(['range', 'n'])['abs(error)'].mean().to_frame("mean_abs_error")

    decode_pars = mean_E.join(mean_error).join(var_E).join(abs_error)

    decode_pars.to_csv(target_dir / f'sub-{subject:02d}_desc-expected_error.tsv', sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Fisher information for encoding model.")
    parser.add_argument("subject", type=int, help="Subject ID")
    parser.add_argument("--model_label", default=15, type=int, help="Model label")
    parser.add_argument("--smoothed", action='store_true', help="Whether the data is smoothed")
    parser.add_argument("--fit_responses", action='store_true', help="Whether to fit responses")
    parser.add_argument("--spherical_noise", action='store_true', help="Spherical noise?")
    parser.add_argument("--wide", action='store_true', help="Use wide prior for both wide and narrow stimuli")
    parser.add_argument("--bids_folder", default='/data/ds-neuralpriors', type=Path, help="BIDS folder path")

    args = parser.parse_args()
    main(args.subject, args.model_label, args.smoothed, args.fit_responses, args.bids_folder, spherical_noise=args.spherical_noise, wide=args.wide)