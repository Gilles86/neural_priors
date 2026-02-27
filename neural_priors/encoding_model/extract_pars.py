import argparse
from pathlib import Path
from neural_priors.utils.data import Subject, get_all_subject_ids
from tqdm import tqdm
import pandas as pd


def main(model_label, roi='NPCr', bids_folder='/data/ds-neuralpriors', smoothed=True, fit_responses=False, censored=False):

    target_dir = Path(bids_folder) / 'derivatives' / 'extracted_pars'
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f'Writing to {target_dir}')

    subject_ids = get_all_subject_ids()
    subjects = [Subject(subject_id=subject_id, bids_folder=bids_folder) for subject_id in subject_ids]

    pars = []
    keys = []

    for sub in tqdm(subjects, desc=f'Model {model_label}'):
        try:
            pars.append(sub.get_prf_parameters_volume(smoothed=smoothed, model_label=model_label, roi=roi, response_fit=fit_responses, censored=censored))
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_label', type=int)
    argparser.add_argument('--roi', default='NPCr', type=str)
    argparser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    argparser.add_argument('--smoothed', action='store_true')
    argparser.add_argument('--fit_responses', action='store_true')
    argparser.add_argument('--censored', action='store_true')
    args = argparser.parse_args()
    main(args.model_label, roi=args.roi, bids_folder=args.bids_folder, smoothed=args.smoothed, fit_responses=args.fit_responses, censored=args.censored)