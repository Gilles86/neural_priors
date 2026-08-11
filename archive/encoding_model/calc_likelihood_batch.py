"""
Serial local loop over models x subjects calling calc_likelihood.main();
the SLURM equivalent is slurm_jobs/submit_likelihoods.sh.
Must be run from inside this directory (flat import of calc_likelihood).
"""

import argparse
from calc_likelihood import main
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=int, nargs='+', required=True, help='List of model labels to run')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 42)), help='List of subject numbers (default: 1-41)')
    parser.add_argument('--roi', type=str, default='NPCr')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--fit_responses', action='store_true')
    parser.add_argument('--censored', action='store_true')
    return parser.parse_args()

def main_batch():
    args = parse_args()
    total = len(args.models) * len(args.subjects)
    with tqdm(total=total, desc="Likelihood calculations") as pbar:
        for model_label in args.models:
            for subject in args.subjects:
                pbar.set_postfix({"subject": f"{subject:02d}", "model": model_label})
                try:
                    main(subject, model_label, args.roi, args.bids_folder, args.smoothed, args.fit_responses, args.censored)
                except Exception as e:
                    pbar.write(f"Error processing subject {subject:02d}, model {model_label}: {e}")
                pbar.update(1)

if __name__ == '__main__':
    main_batch()
