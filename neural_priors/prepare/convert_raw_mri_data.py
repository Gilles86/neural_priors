import re
import os
import os.path as op
import shutil
import argparse
import pandas as pd
import glob
from nilearn import image
import numpy as np
import json

def main(subject, session, bids_folder='/data'):

    try:
        subject = int(subject)
        subject = f'{subject:02d}'
    except ValueError:
        pass

    sourcedata_root = op.join(bids_folder, 'sourcedata', 'mri', f'sub-{subject}', f'ses-{session}')

    t1w = glob.glob(op.join(sourcedata_root, '*_t1w*.nii'))
    flair = glob.glob(op.join(sourcedata_root, '*flair*.nii'))

    target_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'anat')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for run0, t in enumerate(t1w):
        shutil.copy(t, op.join(target_dir, f'sub-{subject}_ses-{session}_run-{run0+1}_T1w.nii'))
    
    if len(flair) == 1:
        shutil.copy(flair[0], op.join(target_dir, f'sub-{subject}_ses-{session}_FLAIR.nii'))
    else:
        raise ValueError(f"More than 1/no FLAIR {flair}!")


    # # # *** FUNCTIONAL DATA ***
    with open(op.abspath('./bold_template.json'), 'r') as f:
        json_template = json.load(f)

    reg = re.compile('.*run(?P<run>[0-9]+).*')
    funcs = glob.glob(op.join(sourcedata_root, '*run*.nii'))

    runs = [int(reg.match(fn).group(1)) for fn in funcs]

    target_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for run, fn in zip(runs, funcs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject}_ses-{session}_task-task_run-{run}_bold.nii'))

        json_sidecar = json_template
        json_sidecar['PhaseEncodingDirection'] = 'i' if (run % 2 == 1) else 'i-'

        with open(op.join(target_dir, f'sub-{subject}_ses-{session}_task-task_run-{run}_bold.json'), 'w') as f:
            json.dump(json_sidecar, f)


    # *** physio logfiles ***
    physiologs = glob.glob(op.join(sourcedata_root, '*run*scanphyslog*.log'))
    runs = [int(reg.match(fn).group(1)) for fn in physiologs]

    for run, fn in zip(runs, physiologs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject}_ses-{session}_task-task_run-{run}_physio.log'))

    # *** Fieldmaps ***
    func_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'func')
    target_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'fmap')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    with open(op.abspath('./fmap_template.json'), 'r') as f:
        json_template = json.load(f)
        print(json_template)
  
    for target_run in range(1, 8):
        bold =  op.join(func_dir, f'sub-{subject}_ses-{session}_task-task_run-{target_run}_bold.nii')

        if not op.exists(bold):
            print(f"Skipping EPI search for run {target_run}")
            continue


        source_run = target_run + 1
        index_slice = slice(5, 10)
        if source_run == 9:
            source_run = 7
            index_slice = slice(-10, -5)
        
        direction = 'RL' if (source_run % 2 == 1) else 'LR'

        epi = op.join(func_dir, f'sub-{subject}_ses-{session}_task-task_run-{source_run}_bold.nii')
        
        if not op.exists(epi):
            print(f"PROBLEM with target run {target_run}")
            if target_run % 2 == 0:
                potential_source_runs = np.arange(1, 7, 2)
            else:
                potential_source_runs = np.arange(2, 7, 2)

            distances = np.abs(target_run - potential_source_runs)
            potential_source_runs = potential_source_runs[np.argsort(distances)]

            for source_run in potential_source_runs:
                print(source_run)
                epi = op.join(func_dir, f'sub-{subject}_ses-{session}_task-task_run-{source_run}_bold.nii')
                if op.exists(epi):
                    print(f'Using {source_run} as EPI for target {target_run}')
                    print(epi)
                    if (source_run > target_run):
                        index_slice = slice(5, 10)
                    else:
                        index_slice = slice(-10, -5)
                    print(f"Index slice: {index_slice}")
                    break

        epi = image.index_img(epi, index_slice)

        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_dir-{direction}_run-{target_run}_epi.nii')
        epi.to_filename(target_fn)

        json_sidecar = json_template
        json_sidecar['PhaseEncodingDirection'] = 'i' if (source_run % 2 == 1) else 'i-'
        json_sidecar['IntendedFor'] = f'ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{target_run}_bold.nii'

        with open(op.join(target_dir, f'sub-{subject}_ses-{session}_dir-{direction}_run-{target_run}_epi.json'), 'w') as f:
            json.dump(json_sidecar, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('session', type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder)
