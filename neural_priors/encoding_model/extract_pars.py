import os
import os.path as op
import argparse
from neural_priors.utils.data import Subject, get_all_subject_ids
from tqdm.contrib.itertools import product
import pandas as pd


def main(roi='NPCr', bids_folder='/data/ds-neuralpriors', smoothed=True, space='natural'):

    key = 'summary_encoding_models'

    if space == 'log':
        key += '.logspace'
    else:
        key += '.natural_space'

    if smoothed:
        key += '.smoothed'

    print(key)

    target_dir = op.join(bids_folder, 'derivatives', key)
    print(f'Writing to {target_dir}')
    os.makedirs(target_dir, exist_ok=True)

    subject_ids = get_all_subject_ids()
    model_labels = list(range(1, 11))
    subjects = [Subject(subject_id=subject_id) for subject_id in subject_ids]
    pars = []

    keys = []
    for sub, model_label in product(subjects, model_labels):
        try:
            pars.append(sub.get_prf_parameters_volume(smoothed=smoothed, model_label=model_label, roi='NPCr',
                                                      gaussian=space == 'natural'))
            keys.append((sub.subject_id, model_label))
        except Exception as e:
            print(f"Failed for {sub.subject_id} model {model_label}: {e}")


    pars = pd.concat(pars, keys=keys, names=['subject_id', 'model_label'])
    pars.columns.names = ['parameter', 'range']
    pars.to_csv(op.join(target_dir, f'group_roi-{roi}_parameters.tsv'), sep='\t')


argparser = argparse.ArgumentParser()
argparser.add_argument('roi', default='NPCr', type=str)
argparser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
argparser.add_argument('--smoothed', action='store_true')
argparser.add_argument('--log_space', action='store_const', const='log', default='natural', dest='space')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(roi=args.roi, bids_folder=args.bids_folder, smoothed=args.smoothed, space=args.space)