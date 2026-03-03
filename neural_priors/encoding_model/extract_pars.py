"""
Extract pRF parameters for all subjects and concatenate into a single group TSV file.

Output is written to:
  <bids_folder>/derivatives/extracted_pars/
    group_roi-<roi>_model-<model_label>_desc-<desc>_parameters.tsv

where <desc> is one of:
  groundtruth          default
  responses            when --fit_responses is set
  groundtruth.censored when --censored is set
  responses.censored   when both --fit_responses and --censored are set
"""

import argparse
from pathlib import Path
from neural_priors.utils.data import Subject, get_all_subject_ids
from tqdm import tqdm
import pandas as pd


def main(model_label, roi='NPCr', bids_folder='/data/ds-neuralpriors', smoothed=True, fit_responses=False, censored=False):
    """Extract and save pRF parameters for all subjects for a given model."""


    if not smoothed:
        raise NotImplementedError("Only smoothed parameters are currently supported, as the unsmoothed ones are very noisy and not useful for analysis.")

    target_dir = Path(bids_folder) / 'derivatives' / 'extracted_pars'
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f'Writing to {target_dir}')

    subject_ids = get_all_subject_ids()
    subjects = [Subject(subject_id=subject_id, bids_folder=bids_folder) for subject_id in subject_ids]

    # Key for likelihood NIfTI files (same logic as calc_likelihood.py)
    ll_key = f'model{model_label}'
    if censored:
        ll_key += '.censored'
    if smoothed:
        ll_key += '.smoothed'
    if fit_responses:
        ll_key += '.fit_responses'

    pars = []
    keys = []

    for sub in tqdm(subjects, desc=f'Model {model_label}'):
        try:
            ll_fn = (Path(bids_folder) / 'derivatives' / 'encoding_models' / ll_key
                     / f'sub-{sub.subject_id}' / 'func'
                     / f'sub-{sub.subject_id}_desc-loglikelihood_roi-{roi}_space-T1w_pars.nii.gz')

            if model_label == -1:
                # Null model has no pRF parameters; only load loglikelihood
                if not ll_fn.exists():
                    print(f"No loglikelihood file for {sub.subject_id}, skipping")
                    continue
                masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
                ll = pd.Series(masker.transform(str(ll_fn)).squeeze(), name=('loglikelihood', None))
                sub_pars = ll.to_frame()
            else:
                sub_pars = sub.get_prf_parameters_volume(smoothed=smoothed, model_label=model_label, roi=roi, response_fit=fit_responses, censored=censored, use_nifti=True)
                if ll_fn.exists():
                    try:
                        masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
                        ll = pd.Series(masker.transform(str(ll_fn)).squeeze(), name=('loglikelihood', None))
                        sub_pars = pd.concat([sub_pars, ll], axis=1)
                    except Exception as e:
                        print(f"Warning: could not load loglikelihood for {sub.subject_id}: {e}")

            pars.append(sub_pars)
            keys.append((sub.subject_id, model_label, smoothed))
        except Exception as e:
            print(f"Failed for {sub.subject_id} model {model_label}: {e}")

    pars = pd.concat(pars, keys=keys, names=['subject_id', 'model_label', 'smoothed'], axis=0)
    pars.columns.names = ['parameter', 'range']

    desc = 'responses' if fit_responses else 'groundtruth'
    if censored:
        desc += '.censored'
    fn = target_dir / f'group_roi-{roi}_model-{model_label}_desc-{desc}_parameters.tsv'
    pars.to_csv(fn, sep='\t')
    print(f'Wrote {fn}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argparser.add_argument('model_label', type=int,
                           help='Integer model label (e.g. 1, 31, 101)')
    argparser.add_argument('--roi', default='NPCr', type=str,
                           help='ROI label used in filename and parameter lookup (default: NPCr)')
    argparser.add_argument('--bids_folder', default='/data/ds-neuralpriors',
                           help='Root of the BIDS dataset; output goes to '
                                '<bids_folder>/derivatives/extracted_pars/ (default: /data/ds-neuralpriors)')
    argparser.add_argument('--smoothed', action='store_true',
                           help='Use spatially smoothed parameter maps')
    argparser.add_argument('--fit_responses', action='store_true',
                           help='Use response-fitted parameters instead of ground-truth parameters')
    argparser.add_argument('--censored', action='store_true',
                           help='Use censored parameter estimates')
    args = argparser.parse_args()
    main(args.model_label, roi=args.roi, bids_folder=args.bids_folder, smoothed=args.smoothed, fit_responses=args.fit_responses, censored=args.censored)