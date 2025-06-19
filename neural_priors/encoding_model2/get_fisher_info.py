import argparse
from pathlib import Path
from neural_priors.utils.data import Subject
from neural_priors.encoding_model2.fit_model import get_model, get_paradigm
import pandas as pd
import numpy as np
from braincoder.utils import get_rsq
from braincoder.optimize import ResidualFitter

def main(subject, model_label, smoothed, fit_responses, bids_folder, roi='NPCr'):

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

    model = get_model(model_label)

    pred = model.predict(paradigm, pars)

    r2 = get_rsq(data, pred)

    print(r2.sort_values())

    narrow_stimuli = np.concatenate((np.arange(10, 26, 1)[:, np.newaxis], np.zeros(16)[:, np.newaxis]), axis=1)
    wide_stimuli = np.concatenate((np.arange(10, 41, 1)[:, np.newaxis], np.ones(31)[:, np.newaxis]), axis=1)
    stimuli = np.concatenate((narrow_stimuli, wide_stimuli), axis=0).astype(np.float32)

    resid_fitter = ResidualFitter(model, data, paradigm, pars)

    model.init_pseudoWWT(stimuli, pars)

    omega, dof = resid_fitter.fit()

    print(omega)
    print(dof)

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
    parser.add_argument("--fit_responses", action='store_true', help="Whether to fit responses")
    parser.add_argument("--bids_folder", default='/data/ds-neuralpriors', type=Path, help="BIDS folder path")

    args = parser.parse_args()
    main(args.subject, args.model_label, args.smoothed, args.fit_responses, args.bids_folder)