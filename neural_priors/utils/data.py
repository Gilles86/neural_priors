import os.path as op
import numpy as np
import pandas as pd
import os
from nilearn import image, surface
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask
import pkg_resources
import yaml
from braincoder.models import LogGaussianPRF, GaussianPRF

def get_all_subject_ids(only_full=True):
    with pkg_resources.resource_stream('neural_priors', '/data/subjects.yml') as stream:
        mapping = yaml.safe_load(stream)

        subjects = []

        for key in mapping.keys():
            if len(mapping[key]) > 1:
                subjects.append(key)
        
        return subjects

def get_all_behavioral_data(bids_folder='/data/ds-neuralpriors', subjects=None):
    subjects = get_all_subject_ids() if subjects is None else subjects

    df = []
    for subject in subjects:
        sub = Subject(subject, bids_folder=bids_folder)
        df.append(sub.get_behavioral_data(add_info=True))

    df = pd.concat(df)

    return df

class Subject(object):


    def __init__(self, subject_id, bids_folder='/data/ds-neuralpriors'):
        if type(subject_id) == int:
            subject_id = f'{subject_id:02d}'
        
        self.subject_id = subject_id
        self.bids_folder = bids_folder

        self.derivatives_dir = op.join(bids_folder, 'derivatives')

    def get_sessions(self):
        assert self.subject_id in get_all_subject_ids(), f'{self.subject_id} not in {get_all_subject_ids()}'
        with pkg_resources.resource_stream('neural_priors', '/data/subjects.yml') as stream:
            return yaml.safe_load(stream)[self.subject_id]

    def get_behavioral_data(self, session=None, tasks=None, raw=False, add_info=True):

        if session is None:
            data = pd.concat((self.get_behavioral_data(session, tasks, raw, add_info) for session in self.get_sessions()), keys=self.get_sessions(), names=['session'])

            if tasks is None:
                data = data.reorder_levels(['subject', 'session', 'run', 'trial_nr']).sort_index()
            else:
                data = data.reorder_levels(['subject', 'session', 'task', 'run', 'trial_nr']).sort_index()

            return data

        if tasks is None:
            tasks = ['estimation_task']

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
        df = df.droplevel(-2)

        if raw:
            if add_info:
                raise ValueError('add_info is not implemented for raw data')
            return df

        df = df.set_index('trial_nr', append=True)

        df = df.xs('feedback', level='event_type')

        if len(df.index.unique(level='task')) == 1:
            df = df.droplevel('task')

        df['response'] = df['response'].astype(float)
        df['n'] = df['n'].astype(float)

        if add_info:
            df['error'] = df['response'] - df['n']
            df['abs_error'] = np.abs(df['error'])
            df['squared_error'] = df['error']**2

        return df

    def get_runs(self, session):

        if session is None:
            return self.get_runs(1) + self.get_runs(2)

        return list(range(1, 9))


    def get_preprocessed_bold(self, session=1, runs=None, space='T1w'):

        if session is None:
            return self.get_preprocessed_bold(1, runs, space) + self.get_preprocessed_bold(2, runs, space)

        base_dir = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', f'ses-{session}', 'func')

        if runs is None:
            runs = self.get_runs(session)

        files = [op.join(base_dir, f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_space-{space}_desc-preproc_bold.nii.gz') for run in runs]

        # Check if all files exist
        for f in files:
            if not op.exists(f):
                raise ValueError(f'File {f} does not exist')

        return files

    def get_onsets(self, session=None):

        if session is None:
            sessions = [1,2]
            return pd.concat([self.get_onsets(session) for session in sessions], keys=sessions, names=['session'])

        runs = self.get_runs(session)

        onsets = pd.concat([pd.read_csv(op.join(self.bids_folder, f'sub-{self.subject_id}', f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_events.tsv'), index_col='trial_nr', sep='\t') for run in runs],
                           keys=runs, names=['run'])

        return onsets


    def get_confounds(self, session=None, type='minimum'):
        runs = self.get_runs(session)

        confounds = pd.concat([pd.read_csv(op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_desc-confounds_timeseries.tsv'), sep='\t') for run in runs],
                                keys=runs, names=['run'])

        if type == 'minimum':
            confound_labels = ['cosine00', 'cosine01', 'cosine02', 'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02', 'non_steady_state_outlier03']
        elif type == 'full':
            confound_labels = confounds.columns
        
        return confounds[confound_labels]


    def get_single_trial_estimates(self, session, type='stim', smoothed=False, roi=None,
                                   zscore_sessions=False):

        if (session is not None) and (zscore_sessions):
            raise ValueError(f'Cannot zscore across sessions wiht only one session ({session})')
        
        dir = 'glm_stim1.denoise'

        if smoothed:
            dir += '.smoothed'

        if session is None:
            dir = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func')
            fn = op.join(dir, f'sub-{self.subject_id}_task-task_space-T1w_desc-{type}_pe.nii.gz')
        else:
            dir = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', f'ses-{session}', 'func')
            fn = op.join(dir, f'sub-{self.subject_id}_ses-{session}_task-task_space-T1w_desc-{type}_pe.nii.gz')

        im = image.load_img(fn, dtype=np.float32)

        n_volumes = 240 if session is not None else 480
        assert(im.shape[3] == n_volumes), f'Expected {n_volumes} volumes, got {im.shape[3]}'

        if zscore_sessions:
            session1 = image.index_img(im, slice(0, 240))
            session2= image.index_img(im, slice(240, 480))

            session1 = image.clean_img(session1, detrend=False, standardize='zscore')
            session2 = image.clean_img(session2, detrend=False, standardize='zscore')

            im = image.concat_imgs([session1, session2])

        if roi is not None:
            mask = self.get_volume_mask(roi=roi, session=session, epi_space=True)
            masker = NiftiMasker(mask)
            im = masker.fit_transform(im)

        return im

    def get_brain_mask(self, session=None, epi_space=True, return_masker=True, debug_mask=False):
        if not epi_space:
            raise ValueError('Only EPI space is supported')

        session = 1 if session is None else session

        fn = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}',
                    f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-2_space-T1w_desc-brain_mask.nii.gz')

        mask_img = image.load_img(fn)

        if debug_mask:
            # Convert to numpy array
            mask_data = mask_img.get_fdata()

            # Create a downsampled mask: keep 1 in 100 voxels
            mask_indices = np.argwhere(mask_data > 0)
            np.random.shuffle(mask_indices)
            subsample_size = max(1, len(mask_indices) // 100)  # Ensure at least one voxel
            subsample_indices = mask_indices[:subsample_size]

            # Create a new empty mask
            debug_mask_data = np.zeros_like(mask_data)

            # Set the selected voxels to 1
            for idx in subsample_indices:
                debug_mask_data[tuple(idx)] = 1

            # Create a new Nifti image
            mask_img = image.new_img_like(mask_img, debug_mask_data)

        if return_masker:
            return NiftiMasker(mask_img=mask_img)

        return mask_img


    def get_volume_mask(self, roi=None, session=None, epi_space=False, return_masker=False):

        if session is None:
            session = 1

        base_mask = op.join(self.bids_folder, 'derivatives', f'fmriprep/sub-{self.subject_id}/ses-{session}/func/sub-{self.subject_id}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')
        base_mask = image.load_img(base_mask, dtype='int32') # To prevent weird nilearn warning

        first_run = self.get_preprocessed_bold(session=session, runs=[1])[0]
        base_mask = image.resample_to_img(base_mask, first_run, interpolation='nearest')

        if roi is None:
            if epi_space:
                return base_mask
            else:
                raise NotImplementedError

        elif roi.startswith('NPC') or roi.startswith('NF') or roi.startswith('NTO'):
            
            anat_mask = op.join(self.derivatives_dir
            ,'ips_masks',
            f'sub-{self.subject_id}',
            'anat',
            f'sub-{self.subject_id}_space-T1w_desc-{roi}_mask.nii.gz'
            )

            if epi_space:
                epi_mask = op.join(self.derivatives_dir
                                    ,'ips_masks',
                                    f'sub-{self.subject_id}',
                                    'func',
                                    f'ses-{session}',
                                    f'sub-{self.subject_id}_space-T1w_desc-{roi}_mask.nii.gz')

                if not op.exists(epi_mask):
                    if not op.exists(op.dirname(epi_mask)):
                        os.makedirs(op.dirname(epi_mask))


                    im = image.resample_to_img(image.load_img(anat_mask, dtype='int32'), image.load_img(base_mask, dtype='int32'), interpolation='nearest')
                    im.to_filename(epi_mask)

                mask = epi_mask

            else: 
                mask = anat_mask

        else:
            raise NotImplementedError

        if return_masker:
            return NiftiMasker(mask)

        return image.load_img(mask, dtype='int32')

    def get_prf_parameters_volume(self, session=None, 
            run=None,
            smoothed=False,
            fixed_baseline=False,
            cross_validated=False,
            keys=None,
            roi=None,
            range_n=None,
            return_image=False,
            gaussian=False,
            joint=False,
            model_label=1,
            wprf=False):

        if (session is not None) and (not cross_validated):
            raise ValueError('Session must be None')
        if gaussian and wprf:
            raise ValueError('Cannot have both gaussian and wprf')

        dir = 'encoding_model'

        if cross_validated:
            if run is None:
                raise Exception('Give run')

            dir += '.cv'

        if joint:
            dir += f'.joint.model{model_label}'
        else:
            dir += '.denoise'

        if wprf:
            dir += '.wprf'

        if gaussian:
            dir += '.gaussian'

        if smoothed:
            dir += '.smoothed'

        if fixed_baseline:
            dir += '.fixed_baseline'

        if range_n is not None:
            assert(range_n in ['wide', 'narrow', 'wide2']), f'range must be either "wide", "narrow", or "wide2"'
            if not joint:
                dir += f'.range_{range_n}'

        parameters = []

        if keys is None:
            if wprf:
                keys= ['cvr2']
            elif gaussian:
                keys = ['mu', 'sd', 'amplitude', 'baseline', 'r2', 'cvr2']
            else:
                keys = ['mode', 'fwhm', 'amplitude', 'baseline', 'r2', 'cvr2']

        mask = self.get_volume_mask(session=session, roi=roi, epi_space=True)
        masker = NiftiMasker(mask)

        if cross_validated:
            if session is None:
                raise ValueError('Session must be given for cross-validated data')
            fn_template = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func',
                                    'sub-{subject_id}_ses-{session}_run-{run}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
        else:
            
            if joint:
                fn_template = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func', 'sub-{subject_id}_desc-{range_n}.{parameter_key}.optim_space-T1w_pars.nii.gz')
                r2_template = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func', 'sub-{subject_id}_desc-r2.optim_space-T1w_pars.nii.gz')
                cvr2_template = op.join(self.bids_folder, 'derivatives', dir.replace('encoding_model.joint', 'encoding_model.joint.cv'), f'sub-{self.subject_id}', 'func', 'sub-{subject_id}_desc-cvr2.optim_space-T1w_pars.nii.gz')
            else:
                fn_template = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func', 'sub-{subject_id}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
                cvr2_template = op.join(self.bids_folder, 'derivatives', dir.replace('encoding_model', 'encoding_model.cv'), f'sub-{self.subject_id}', 'func', 'sub-{subject_id}_desc-cvr2.optim_space-T1w_pars.nii.gz')

        for parameter_key in keys:

            try:
                if (parameter_key == 'cvr2') and (not cross_validated):
                    fn = cvr2_template.format(parameter_key=parameter_key, run=run, session=session, subject_id=self.subject_id)
                elif joint and (parameter_key == 'r2'):
                    fn = r2_template.format(run=run, session=session, subject_id=self.subject_id)
                else:
                    fn = fn_template.format(parameter_key=parameter_key, run=run, session=session, subject_id=self.subject_id, range_n=range_n)
                
                if not hasattr(masker, "mask_img_"):
                    masker.fit(fn)

                pars = pd.Series(apply_mask(fn, mask, ensure_finite=False))
                parameters.append(pars)
            except Exception as e:
                print(f'Could not load {fn}: {e}')

        parameters =  pd.concat(parameters, axis=1, keys=keys, names=['parameter']).astype(np.float32)

        if return_image:
            return masker.inverse_transform(parameters.T)

        return parameters

    def get_surf_info(self):
        info = {'L':{}, 'R':{}}

        for hemi in ['L', 'R']:

            fs_hemi = {'L':'lh', 'R':'rh'}[hemi]

            info[hemi]['inner'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'anat', f'sub-{self.subject_id}_hemi-{hemi}_white.surf.gii')
            info[hemi]['mid'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'anat', f'sub-{self.subject_id}_hemi-{hemi}_thickness.shape.gii')
            info[hemi]['outer'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'anat', f'sub-{self.subject_id}_hemi-{hemi}_pial.surf.gii')
            # info[hemi]['inflated'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'ses-1', 'anat', f'sub-{self.subject_id}_ses-1_hemi-{hemi}_inflated.surf.gii')
            info[hemi]['curvature'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer', f'sub-{self.subject_id}', 'surf', f'{fs_hemi}.curv')

            for key in info[hemi]:
                assert(os.path.exists(info[hemi][key])), f'{info[hemi][key]} does not exist'

        return info

    def get_prf_parameters_surf(self, session, run=None, smoothed=False, cross_validated=False, hemi=None,
                                mask=None, space='fsnative', parameters=None, key=None, nilearn=True,
                                range_n=None):

        if nilearn is False:
            raise NotImplementedError

        if mask is not None:
            raise NotImplementedError

        if parameters is None:
            parameter_keys = ['mode', 'fwhm', 'amplitude', 'cvr2', 'r2']
        else:
            parameter_keys = parameters

        if hemi is None:
            prf_l = self.get_prf_parameters_surf(session, 
                    run, smoothed, cross_validated, hemi='L',
                    mask=mask, space=space, key=key, parameters=parameters, nilearn=nilearn,
                    range_n=range_n)
            prf_r = self.get_prf_parameters_surf(session, 
                    run, smoothed, cross_validated, hemi='R',
                    mask=mask, space=space, key=key, parameters=parameters, nilearn=nilearn,
                    range_n=range_n)
            
            return pd.concat((prf_l, prf_r), axis=0, 
                    keys=pd.Index(['L', 'R'], name='hemi'))


        if key is None:
            if cross_validated:
                key = 'encoding_model.cv.denoise'
            else:
                key = 'encoding_model.denoise'

            if smoothed:
                key += '.smoothed'

        if range_n is not None:
            key += f'.range_{range_n}'

        parameters = []

        if session is None:
            dir = op.join(self.bids_folder, 'derivatives', key, f'sub-{self.subject_id}', 'func')

            if run is not None:
                fn_template = op.join(dir, 'sub-{subject_id}_run-{run}_desc-{parameter_key}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii')
            else:
                fn_template = op.join(dir, 'sub-{subject_id}_desc-{parameter_key}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii')

        else:
            dir = op.join(self.bids_folder, 'derivatives', key, f'sub-{self.subject_id}', f'ses-{session}', 'func')

            if run is not None:
                fn_template = op.join(dir, 'sub-{subject_id}_ses-{session}_run-{run}_desc-{parameter_key}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii')
            else:
                fn_template = op.join(dir, 'sub-{subject_id}_ses-{session}_desc-{parameter_key}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii')

        for parameter_key in parameter_keys:

            fn = fn_template.format(parameter_key=parameter_key, run=run, session=session, subject_id=self.subject_id, hemi=hemi, space=space)

            pars = pd.Series(surface.load_surf_data(fn))
            pars.index.name = 'vertex'

            parameters.append(pars)

        return pd.concat(parameters, axis=1, keys=parameter_keys, names=['parameter'])

    def get_t1w(self):

        if self.get_sessions() is [1, 2]:
            raise NotImplementedError
        else:
            t1w = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'ses-1', 'anat', f'sub-{self.subject_id}_ses-1_desc-preproc_T1w.nii.gz')

            if not op.exists(t1w):
                raise ValueError(f'{t1w} does not exist')

            return image.load_img(t1w)

    def get_prf_predictions(self, session, smoothed=False, roi=None, range_n=None, return_image=False,
                            include_n=True, gaussian=False):

        prf_pars = self.get_prf_parameters_volume(session, cross_validated=False, smoothed=smoothed, roi=roi, range_n=range_n,
                                                return_image=False, gaussian=gaussian)

        paradigm = self.get_behavioral_data()['n']

        if gaussian:
            model = GaussianPRF(paradigm=paradigm, parameters=prf_pars)
        else:
            model = LogGaussianPRF(paradigm=paradigm, parameters=prf_pars,
                            parameterisation='mode_fwhm_natural')

        predictions = model.predict(paradigm)

        if include_n:
            predictions.set_index(paradigm, append=True, inplace=True)

        return predictions
