import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject, get_all_subject_ids
from tqdm.contrib.itertools import product
import pandas as pd


def main(roi='NPCr', bids_folder='/data/ds-neuralpriors', smoothed=True, fit_responses=False):

    key = 'encoding_models'

    target_dir = op.join(bids_folder, 'derivatives', key)
    print(f'Writing to {target_dir}')
    os.makedirs(target_dir, exist_ok=True)

    subject_ids = get_all_subject_ids()
    model_labels = list(range(0, 15))

    # model_labels = [0, 3, 4, 5] + list(range(12, 25))
    model_labels = [0,3,4,5, 14, 15, 25]
    subjects = [Subject(subject_id=subject_id) for subject_id in subject_ids]
    pars = []

    keys = []
    for sub, model_label, smoothed in product(subjects, model_labels, [True]):
        try:
            pars.append(sub.get_prf_parameters_volume2(smoothed=smoothed, model_label=model_label, roi='NPCr', response_fit=fit_responses))
            keys.append((sub.subject_id, model_label, smoothed))
        except Exception as e:
            print(f"Failed for {sub.subject_id} model {model_label}: {e}")

    pars = pd.concat(pars, keys=keys, names=['subject_id', 'model_label', 'smoothed'], axis=0)
    pars.columns.names = ['parameter', 'range']
    
    if fit_responses:
        pars.to_csv(op.join(target_dir, f'group_roi-{roi}_desc-responses_parameters.tsv'), sep='\t')
    else:
        pars.to_csv(op.join(target_dir, f'group_roi-{roi}_desc-groundtruth_parameters.tsv'), sep='\t')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('roi', default='NPCr', type=str)
    argparser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    argparser.add_argument('--fit_responses', action='store_true')
    args = argparser.parse_args()
    main(roi=args.roi, bids_folder=args.bids_folder, fit_responses=args.fit_responses)