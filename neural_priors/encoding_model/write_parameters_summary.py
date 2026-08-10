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
import numpy as np
import pandas as pd
from tqdm import tqdm
from neural_priors.utils.data import Subject, get_all_subject_ids
from neural_priors.encoding_model.fit_model import get_paradigm


def _load_ll_from_extracted_pars(bids_folder, roi, model_label, smoothed, fit_responses, censored):
    """Load loglikelihood (and cvr2 for model -1) from the extract_pars TSV.

    Returns dict mapping subject_id -> dict of {column_name -> np.array per voxel}.
    """
    extracted_dir = Path(bids_folder) / 'derivatives' / 'extracted_pars'
    desc = 'responses' if fit_responses else 'groundtruth'
    if censored:
        desc += '.censored'
    fn = extracted_dir / f'group_roi-{roi}_model-{model_label}_desc-{desc}_parameters.tsv'
    if not fn.exists():
        return {}
    try:
        df = pd.read_csv(fn, sep='\t', header=[0, 1], index_col=[0, 1, 2])
        # Flatten MultiIndex columns, dropping empty/nan levels
        df.columns = ['_'.join(str(c) for c in col if str(c) not in ('', 'nan'))
                      for col in df.columns]
        wanted = [c for c in df.columns if c in ('loglikelihood', 'cvr2')]
        if not wanted:
            return {}
        result = {}
        for subj in df.index.get_level_values(0).unique():
            subj_str = str(subj).zfill(2)
            subj_df = df.loc[subj][wanted]
            result[subj_str] = {col: subj_df[col].values for col in wanted}
        return result
    except Exception as e:
        print(f'Warning: could not load extract_pars TSV for model {model_label}: {e}')
        return {}


# Number of free parameters per voxel for each model (per-voxel free + shared counted as 1 + sigma).
# Shared parameters are counted conservatively as one per voxel.
N_PARAMETERS = {
    -1: 2,   # mean + sigma
    0:  5,   # mu, sd, amp, baseline + sigma  (lb, alpha, delta_wide fixed)
    1:  5,   # same as 0 (delta_wide=2 fixed)
    2:  6,   # + shared delta_wide
    3:  7,   # mu, delta_wide, lb, sd, amp, baseline + sigma
    4:  5,   # same as 1 with identity_below_range
    5:  6,   # + shared delta_wide
    14: 6,   # mu, sd_narrow, sd_wide, amp, baseline + sigma
    15: 6,   # mu, baseline, sd_narrow, amp + shared sd_wide_scale + sigma
    18: 7,   # mu, baseline, sd_narrow, amp + shared delta_wide, sd_wide_scale + sigma
    31: 5,   # mu, baseline, sd_narrow, amp + sigma  (delta_wide, sd_wide_scale fixed)
    32: 6,   # mu, baseline, sd_narrow, amp_narrow, amp_beta + sigma
    33: 6,   # mu, baseline, sd_narrow, amp_narrow + shared amp_beta + sigma
    34: 7,   # mu, lb, baseline, sd, amp_narrow, amp_beta + sigma
    35: 8,   # mu, lb, baseline, sd, amp_narrow + shared amp_alpha, amp_beta + sigma
}

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
    18: 'Fitted shift + width scaling, shared across voxels',
    31: 'Fixed width scaling (δ_σ=1.29)',
    32: 'Fixed width scaling + free amplitude ratio',
    33: 'Fixed width scaling + shared amplitude ratio',
    34: 'Fixed tuning, free amplitude per voxel',
    35: 'Fixed tuning, shared amplitude ratio + fitted shift',
    36: 'Fixed tuning, shared amplitude ratio',
}


def main(models=None, roi='NPCr', bids_folder='/data/ds-neuralpriors', smoothed=True, censored=False):
    if models is None:
        models = list(MODELS.keys())

    subject_ids = get_all_subject_ids()

    target_dir = Path(bids_folder) / 'derivatives' / 'summary_tsvs'
    target_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load loglikelihood (and cvr2 for model -1) from extract_pars TSVs
    ll_lookup = {}  # (model_label, response_fit) -> {subject_id -> {col -> array}}
    for model_label in models:
        for response_fit in ([False] if model_label == -1 else [False, True]):
            ll_lookup[(model_label, response_fit)] = _load_ll_from_extracted_pars(
                bids_folder, roi, model_label, smoothed, response_fit, censored)

    rows = []

    for subject_id in tqdm(subject_ids, desc='Subjects'):
        sub = Subject(subject_id, bids_folder=bids_folder)

        try:
            paradigm = get_paradigm(sub, fit_responses=False)
            n_obs = int((paradigm['x'] < 26).sum()) if censored else len(paradigm)
        except Exception as e:
            print(f'Warning: could not get n_obs for {subject_id}: {e}')
            n_obs = None

        for model_label in models:
            model_name = MODELS.get(model_label, str(model_label))

            if model_label == -1:
                ll_data = ll_lookup.get((-1, False), {}).get(subject_id, {})
                if 'loglikelihood' not in ll_data:
                    print(f'Warning: subject {subject_id} model -1: no loglikelihood in TSV')
                    continue
                pars = pd.DataFrame(ll_data)
                k = N_PARAMETERS[-1]
                pars['aic'] = 2 * k - 2 * pars['loglikelihood']
                if n_obs is not None:
                    pars['bic'] = np.log(n_obs) * k - 2 * pars['loglikelihood']
                pars['subject'] = subject_id
                pars['model_label'] = model_label
                pars['model'] = model_name
                pars['response_fit'] = False
                pars['voxel'] = pars.index
                rows.append(pars)
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

                    # Add loglikelihood from extract_pars TSV and compute AIC/BIC
                    ll_data = ll_lookup.get((model_label, response_fit), {}).get(subject_id, {})
                    if 'loglikelihood' in ll_data:
                        pars['loglikelihood'] = ll_data['loglikelihood']
                        if model_label in N_PARAMETERS:
                            k = N_PARAMETERS[model_label]
                            pars['aic'] = 2 * k - 2 * pars['loglikelihood']
                            if n_obs is not None:
                                pars['bic'] = np.log(n_obs) * k - 2 * pars['loglikelihood']

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

    fn = target_dir / f'main_models_roi-{roi}_desc-{desc}_parameters.tsv.gz'
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
