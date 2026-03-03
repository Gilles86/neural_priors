
import argparse
import numpy as np
from pathlib import Path
from neural_priors.encoding_model.fit_model import get_model, get_paradigm
from neural_priors.utils.data import Subject
from scipy.stats import norm


def log_likelihood(data, pred, sigma):
    return norm.logpdf(data-pred, scale=sigma).sum(0)


def main(subject, model_label, roi, bids_folder, smoothed, fit_responses, censored=False):
    sub = Subject(subject, bids_folder=bids_folder)
    model = get_model(model_label)
    data = sub.get_single_trial_estimates(session=None, roi=roi, smoothed=smoothed)

    if model_label == -1:
        # For the null model, we just use the mean of the data as prediction (for every voxel separately!)
        pred = data.mean(0)
        pred = np.tile(pred, (data.shape[0], 1))
        
    else:

        pars = sub.get_prf_parameters_volume(smoothed=smoothed, model_label=model_label, roi=roi, response_fit=fit_responses, raw=True, censored=censored)
        paradigm = get_paradigm(sub, fit_responses=fit_responses)
        pred = model.predict(paradigm=paradigm, parameters=pars)

    residuals = data - pred
    sigma = np.std(residuals, axis=0)
    ll = log_likelihood(data, pred, sigma)

    # Build key and output path as in fit_model.py
    key = f"model{model_label}"
    if censored:
        key += ".censored"
    if smoothed:
        key += ".smoothed"
    if fit_responses:
        key += ".fit_responses"

    subj_str = f"{int(subject):02d}"
    outdir = Path(bids_folder) / "derivatives" / "encoding_models" / key / f"sub-{subj_str}" / "func"
    outdir.mkdir(parents=True, exist_ok=True)

    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    outfile = outdir / f"sub-{subj_str}_desc-loglikelihood_roi-{roi}_space-T1w_pars.nii.gz"
    masker.inverse_transform(ll).to_filename(str(outfile))
    print(f"Wrote likelihoods to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=int)
    parser.add_argument("model_label", type=int)
    parser.add_argument("--roi", type=str, default="NPCr")
    parser.add_argument("--bids_folder", type=str, default="/data/ds-neuralpriors")
    parser.add_argument("--smoothed", action="store_true")
    parser.add_argument("--fit_responses", action="store_true")
    parser.add_argument("--censored", action="store_true")
    args = parser.parse_args()
    main(args.subject, args.model_label, args.roi, args.bids_folder, args.smoothed, args.fit_responses, args.censored)
