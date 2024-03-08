import pandas as pd
import numpy as np
import argparse
import os
import os.path as op

possible_ranges = [[10, 25], [10, 40]]
n_runs = 6
n_trials_per_run = 10


def main(subject, overwrite):

    # Check if subject can be an integer
    try:
        subject = int(subject)

        if subject %2 == 1:
            range1, range2 = possible_ranges
        else:
            range2, range1 = possible_ranges

        subject = '{:02d}'.format(subject)

    except ValueError:
        range1, range2 = np.random.shuffle(possible_ranges)
        pass

    if not op.exists(op.abspath('settings')):
        os.mkdir('settings')

    target_fn = op.abspath('settings/{}.yml'.format(subject))

    if op.exists(target_fn):
        if not overwrite:
            raise ValueError('Settings file already exists. Stopping (use --force to overwrite)')
        else:
            print('Overwriting settings file')
            os.remove(target_fn)
    
    n_trials = n_runs * n_trials_per_run

    n1 = np.random.randint(range1[0], range1[1] + 1, n_trials)
    n2 = np.random.randint(range2[0], range2[1] + 1, n_trials)


# Create argument parser and __name__ == bla stuff
parser = argparse.ArgumentParser(description='Create settings file for experiment')
parser.add_argument('subject', type=str, help='Subject name')
parser.add_argument('--force', help='Force overwrite of settings file', action='store_true')

args = parser.parse_args()

main(args.subject, overwrite=args.force)

