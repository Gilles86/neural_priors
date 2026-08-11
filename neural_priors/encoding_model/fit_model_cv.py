"""
Cross-validated version of fit_model.py: 8-fold leave-one-run-out fit of an
nPRF model for one subject. Runs are re-indexed as run2 = (run - 1) % 4 + 1,
pairing runs k and k+4 of the same session; each of the 8 folds holds out one
(session, run2) pair, so train and test sets both contain
narrow- and wide-condition trials. Refits the full model on each training set
and evaluates out-of-sample R2 on the held-out runs.

Writes the fold-mean cvR2 per voxel as a NIfTI to
derivatives/encoding_models/model{N}.cv[.flags]/sub-{id}/func/.
Used for voxel selection (analyses keep voxels with mean cvR2 > 0) and model
comparison (proportion of voxels with cvR2 > 0). model_label -1 is a
tuning-free null model predicting each voxel's training-set mean.
"""

import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject
import numpy as np
from braincoder.utils import get_rsq
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn import image
from braincoder.optimize import ParameterFitter
from braincoder.models import RegressionGaussianPRF
from fit_model import get_paradigm, get_model, fit_model

def main(subject, smoothed, model_label=1, bids_folder='/data/ds-neuralpriors', gaussian=True, debug=False, roi='NPCr',
         fit_responses=False, whole_brain=False, censored=False):

    max_n_iterations = 100 if debug else 5000

    # Create target folder
    key = f'model{model_label}.cv'

    if whole_brain:
        key += '.whole_brain'    

    if censored:
        key += '.censored'

    if smoothed:
        key += '.smoothed'

    if fit_responses:
        key += '.fit_responses'

    target_dir = op.join(bids_folder, 'derivatives', 'encoding_models', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Get paradigm/data/model
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, fit_responses=fit_responses)
    # run2 pairs runs k and k+4 of the same session; the 8 (session, run2) pairs define the CV folds
    paradigm = paradigm.set_index(pd.Index((paradigm.index.get_level_values('run') - 1) % 4 + 1, name='run2'), append=True)
    paradigm.index = paradigm.index.swaplevel('run', 'run2')
    paradigm = paradigm.astype(np.float32).droplevel(['run', 'trial_nr', 'subject'])    

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    
    if whole_brain:
        masker = sub.get_brain_mask(epi_space=True, return_masker=True)   
    else:
        masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    if censored:
        data = data[paradigm['x'] < 26]
        paradigm = paradigm[paradigm['x'] < 26]

    all_cvr2 = []

    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run2']):

        print(f'Fitting using session {test_session} run {test_run} as test set')

        test_data, test_paradigm = data.loc[(test_session, test_run)].copy().astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        if model_label == -1:
            # Null model: predict the per-voxel mean of the training set
            train_mean = train_data.mean(axis=0)
            test_pred = pd.DataFrame(
                np.tile(train_mean.values, (len(test_data), 1)),
                index=test_data.index,
                columns=test_data.columns,
            )
        else:
            # Get model
            model = get_model(model_label)
            gd_pars = fit_model(model_label, model, train_data, train_paradigm, max_n_iterations=max_n_iterations)
            test_pred = model.predict(parameters=gd_pars, paradigm=test_paradigm)

        cv_r2 = get_rsq(test_data, test_pred)

        print(f'CV R2: {cv_r2}')

        all_cvr2.append(cv_r2)

    all_cvr2 = pd.concat(all_cvr2, axis=1)
    mean_cvr2 = all_cvr2.mean(axis=1)

    target_fn = op.join(target_dir, f'sub-{subject}_desc-cvr2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(mean_cvr2).to_filename(target_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--model_label', default=1, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--fit_responses', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--whole_brain', action='store_true')
    parser.add_argument('--censored', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, smoothed=args.smoothed, bids_folder=args.bids_folder, debug=args.debug,
         fit_responses=args.fit_responses, whole_brain=args.whole_brain, censored=args.censored)
