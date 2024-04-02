import os.path as op
import numpy as np
import pandas as pd
import os
from nilearn import image

class Subject(object):


    def __init__(self, subject_id, bids_folder='/data/ds-neuralpriors'):
        
        self.subject_id = subject_id
        self.bids_folder = bids_folder


    def get_behavioral_data(self, session=1, tasks=None, raw=False, add_info=True):

        if tasks is None:
            tasks = ['feedback', 'estimation_task']

        behavior_folder = op.join(self.bids_folder, 'sourcedata', 'behavior', f'sub-{self.subject_id}', f'ses-{session}', )


        df = []
        keys = []

        for task in tasks:
            if task == 'feedback':
                runs = [1, 5]
            elif task == 'estimation_task':
                runs = list(range(1, 9))

            for run in runs:
                try:
                    fn = op.join(behavior_folder, f'sub-{self.subject_id}_ses-{session}_task-{task}_run-{run}_events.tsv')
                    d = pd.read_csv(fn, sep='\t')

                    if (d['n'] > 25).any():
                        d['range'] = 'wide'
                    else:
                        d['range'] = 'narrow'


                    keys.append((self.subject_id, task, run))
                    df.append(d)
                except Exception as e:
                    print(f'Problem with {task} run {run}: {e}')

        df = pd.concat(df, keys=keys, names=['subject', 'task', 'run']).set_index('event_type', append=True)

        if raw:
            if add_info:
                raise ValueError('add_info is not implemented for raw data')
            return df

        df = df.xs('feedback', level='event_type')

        df['response'] = df['response'].astype(float)
        df['n'] = df['n'].astype(float)

        if add_info:
            df['error'] = df['response'] - df['n']
            df['abs_error'] = np.abs(df['error'])
            df['squared_error'] = df['error']**2

        return df

    def get_runs(self, session):
        return list(range(2, 9))


    def get_preprocessed_bold(self, session=1, space='T1w'):

        base_dir = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', f'ses-{session}', 'func')

        runs = self.get_runs(session)

        files = [op.join(base_dir, f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_space-{space}_desc-preproc_bold.nii.gz') for run in runs]

        # Check if all files exist
        for f in files:
            if not op.exists(f):
                raise ValueError(f'File {f} does not exist')

        return files

    def get_onsets(self, session=1):

        runs = self.get_runs(session)

        onsets = pd.concat([pd.read_csv(op.join(self.bids_folder, f'sub-{self.subject_id}', f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_events.tsv'), index_col='trial_nr', sep='\t') for run in runs],
                           keys=runs, names=['run'])

        return onsets


    def get_confounds(self, session=1, type='minimum'):
        runs = self.get_runs(session)

        confounds = pd.concat([pd.read_csv(op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_desc-confounds_timeseries.tsv'), sep='\t') for run in runs],
                                keys=runs, names=['run'])

        if type == 'minimum':
            confound_labels = ['cosine00', 'cosine01', 'cosine02', 'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02', 'non_steady_state_outlier03']
        elif type == 'full':
            confound_labels = confounds.columns
        
        return confounds[confound_labels]


    def get_single_trial_estimates(self, session, type='stimulus', smoothed=False):
        
        dir = 'glm.stim1.denoise'

        if smoothed:
            dir += '.smoothed'

        dir = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', f'ses-{session}', 'func')

        fn = op.join(dir, f'sub-{self.subject_id}_ses-{session}_task-task_space-T1w_desc-{stimulus}_pe.nii.gz')

        return image.load_img(fn)
