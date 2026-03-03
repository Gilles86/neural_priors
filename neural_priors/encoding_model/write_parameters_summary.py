"""
Collect all pRF parameters for the specified encoding models across subjects
and write a single long-format summary TSV.

Output:
  <bids_folder>/derivatives/summary_tsvs/main_models_roi-<roi>_desc-<desc>_parameters.tsv

One row per voxel per subject per model per response_fit condition.
MultiIndex columns from get_prf_parameters_volume are flattened:
  ('mu', 'narrow') -> 'mu_narrow',  ('cvr2', nan) -> 'cvr2', etc.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from neural_priors.utils.data import Subject, get_all_subject_ids


MODELS = {
    -1: 'No tuning (null model)',
    0:  'No shift (μ_wide = μ_narrow)',
    1:  'Fixed shift (δ=2, no range constraint)',
    2:  'Fitted shift ratio, shared across voxels',
    3:  'Fitted shift ratio, free per voxel',
    4:  'Efficient coding: fixed shift (δ=2)',
    5:  'Efficient coding: shared shift ratio',
    14: 'Free width ratio, per voxel',
    15: 'Fitted width scaling, shared across voxels',
    31: 'Fixed width scaling (δ_σ=1.29)',
    32: 'Fixed width scaling + free amplitude ratio',
    33: 'Fixed width scaling + shared amplitude ratio',
    34: 'Fixed tuning, free amplitude per voxel',
    35: 'Fixed tuning, shared amplitude ratio',
}


def main(models=None, roi='NPCr', bids_folder='/data/ds-neuralpriors', smoothed=True, censored=False):
    if models is None:
        models = list(MODELS.keys())

    subject_ids = get_all_subject_ids()

    target_dir = Path(bids_folder) / 'derivatives' / 'summary_tsvs'
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for subject_id in tqdm(subject_ids, desc='Subjects'):
        sub = Subject(subject_id, bids_folder=bids_folder)

        for model_label in models:
            model_name = MODELS.get(model_label, str(model_label))

            if model_label == -1:
                # Null model: no pRF parameters, load loglikelihood directly
                ll_key = 'model-1'
                if censored:
                    ll_key += '.censored'
                if smoothed:
                    ll_key += '.smoothed'
                ll_fn = (Path(bids_folder) / 'derivatives' / 'encoding_models' / ll_key
                         / f'sub-{subject_id}' / 'func'
                         / f'sub-{subject_id}_desc-loglikelihood_roi-{roi}_space-T1w_pars.nii.gz')
                if ll_fn.exists():
                    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
                    ll = masker.transform(str(ll_fn)).squeeze()
                    pars = pd.DataFrame({'loglikelihood': ll})
                    pars['subject'] = subject_id
                    pars['model_label'] = model_label
                    pars['model'] = model_name
                    pars['response_fit'] = False
                    pars['voxel'] = pars.index
                    rows.append(pars)
                else:
                    print(f'Warning: subject {subject_id} model -1: no loglikelihood file')
                continue

            for response_fit in [False, True]:
                try:
                    pars = sub.get_prf_parameters_volume(
                        model_label=model_label, roi=roi, smoothed=smoothed,
                        censored=censored, use_nifti=False, response_fit=response_fit)

                    # Flatten MultiIndex columns: ('mu', 'narrow') -> 'mu_narrow', ('cvr2', nan) -> 'cvr2'
                    pars.columns = [
                        '_'.join(c for c in col if c and c != 'nan')
                        for col in pars.columns
                    ]

                    pars['subject'] = subject_id
                    pars['model_label'] = model_label
                    pars['model'] = model_name
                    pars['response_fit'] = response_fit
                    pars['voxel'] = pars.index
                    rows.append(pars)

                except Exception as e:
                    print(f'Warning: subject {subject_id} model {model_label} response_fit={response_fit}: {e}')

    out = pd.concat(rows).reset_index(drop=True)

    meta_cols = ['subject', 'model_label', 'model', 'response_fit', 'voxel']
    data_cols = [c for c in out.columns if c not in meta_cols]
    out = out[meta_cols + data_cols]

    desc = 'groundtruth'
    if censored:
        desc += '.censored'

    fn = target_dir / f'main_models_roi-{roi}_desc-{desc}_parameters.tsv'
    out.to_csv(fn, sep='\t', index=False)
    print(f'Wrote {fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--models', default=None,
                        help='Comma-separated model labels to include (default: all). E.g. --models 3,5')
    parser.add_argument('--roi', default='NPCr',
                        help='ROI label (default: NPCr)')
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors',
                        help='Root of the BIDS dataset (default: /data/ds-neuralpriors)')
    parser.add_argument('--smoothed', action='store_true',
                        help='Use spatially smoothed parameter maps')
    parser.add_argument('--censored', action='store_true',
                        help='Use censored parameter estimates')
    args = parser.parse_args()

    models = [int(m) for m in args.models.split(',')] if args.models else None

    main(models=models, roi=args.roi, bids_folder=args.bids_folder,
         smoothed=args.smoothed, censored=args.censored)
