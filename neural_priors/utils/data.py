"""Data access layer for the neural_priors dataset
("Distributed range adaptation in human parietal encoding of numbers").

All analyses read behavioral data, single-trial fMRI estimates, ROI masks
and fitted encoding-model (nPRF) parameters through the `Subject` class,
so BIDS path conventions live in exactly one place.

Conventions:
- Subject IDs are zero-padded strings ('01'...'41'); ints are converted.
- Included subjects completed 2 sessions x 8 runs x 30 trials = 480 trials.
  sub-11 and sub-23 have only a single session and are excluded
  (see neural_priors/data/subjects.yml).
- Each run used either the narrow (10-25) or wide (10-40) numerosity range;
  the 'range' condition is inferred from the data (wide if any n > 25).
"""
import os.path as op
import numpy as np
import pandas as pd
import os
from nilearn import image, surface
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask
from importlib.resources import files
import yaml
from tqdm.contrib.itertools import product
from itertools import product as product_

def get_all_subject_ids(only_full=True):
    """Return the IDs (zero-padded strings) of all included subjects.

    Inclusion = more than one session listed in data/subjects.yml; this
    drops the single-session subjects sub-11 and sub-23."""
    with files('neural_priors').joinpath('data', 'subjects.yml').open() as stream:
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
        resource_path = files('neural_priors').joinpath('data/subjects.yml')
        with open(resource_path, 'r') as stream:
            return yaml.safe_load(stream)[self.subject_id]

    def get_behavioral_data(self, session=None, tasks=None, raw=False, add_info=True):
        """Trial-wise stimulus numerosities (n) and estimation responses from the
        PsychoPy source logs. The per-run 'range' condition is inferred from the
        data (wide if any n > 25). raw=True returns all logged events; otherwise
        one row per trial, with (abs/squared) estimation errors when add_info."""

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
        """Read the BIDS events.tsv files (stimulus + response events per run,
        written by prepare/make_events_files.py); both sessions concatenated
        when session is None."""

        if session is None:
            sessions = [1,2]
            return pd.concat([self.get_onsets(session) for session in sessions], keys=sessions, names=['session'])

        runs = self.get_runs(session)

        onsets = pd.concat([pd.read_csv(op.join(self.bids_folder, f'sub-{self.subject_id}', f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_events.tsv'), index_col='trial_nr', sep='\t') for run in runs],
                           keys=runs, names=['run'])

        return onsets


    def get_confounds(self, session=None, type='minimum'):
        """fMRIPrep confound regressors per run. 'minimum' keeps only the cosine
        drift terms and non-steady-state outlier indicators; 'full' returns all
        columns."""
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
        """GLMsingle single-trial response amplitudes (glm/fit_single_trials_denoise.py).

        type='stim' (stimulus events, the encoding-model input) or 'response'.
        session=None loads both sessions concatenated (480 trials; asserted),
        a single session yields 240. Optionally z-score each session separately
        and/or return the voxels x trials matrix for an ROI."""

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
            mask = self.get_volume_mask(roi=roi, epi_space=True)
            masker = NiftiMasker(mask)
            im = masker.fit_transform(im)

        return im

    def get_brain_mask(self, session=None, epi_space=True, return_masker=True, debug_mask=False):
        """fMRIPrep whole-brain mask on the T1w-space functional grid, used for
        whole-brain encoding-model fits. debug_mask subsamples ~1% of voxels
        for fast test runs."""
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


    def get_volume_mask(self, roi=None, session=1, epi_space=False, return_masker=False, verbose=False):
        """Retrieve a volume mask, optionally in EPI space and as a NiftiMasker.

        Anatomical ROIs (NPC*/NF*/NTO*) come from derivatives/ips_masks,
        built by surface/get_npc_mask.py from group fsaverage labels;
        NPCr (right NPC) is the paper's main ROI. The EPI-space version is
        resampled (nearest-neighbor) onto the functional grid and cached."""

        def _log(msg):
            """Helper function for verbose logging."""
            if verbose:
                print(msg)

        _log(f"Session: {session}, ROI: {roi}, epi_space: {epi_space}, return_masker: {return_masker}")

        # Load the base brain mask
        base_mask_path = op.join(
            self.bids_folder, 'derivatives', f'fmriprep/sub-{self.subject_id}/ses-{session}/func',
            f'sub-{self.subject_id}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz'
        )
        base_mask = image.load_img(base_mask_path, dtype='int32')  # Prevent weird nilearn warning

        # Resample to first functional run (ensuring affine match)
        first_run = self.get_preprocessed_bold(session=session, runs=[1])[0]
        base_mask = image.resample_to_img(base_mask, first_run, interpolation='nearest', force_resample=False, copy_header=True)
        _log("Base mask loaded and resampled.")

        # Handle case when ROI is None
        if roi is None:
            if epi_space:
                mask = base_mask
            else:
                _log("ROI is None and epi_space=False -> Raising NotImplementedError")
                raise NotImplementedError

        # Handle anatomical masks
        elif roi.startswith(('NPC', 'NF', 'NTO')):  # Tuple avoids multiple `or` conditions
            _log(f"Processing ROI: {roi}")

            anat_mask_path = op.join(
                self.derivatives_dir, 'ips_masks', f'sub-{self.subject_id}', 'anat',
                f'sub-{self.subject_id}_space-T1w_desc-{roi}_mask.nii.gz'
            )

            if epi_space:
                epi_mask_path = op.join(
                    self.derivatives_dir, 'ips_masks', f'sub-{self.subject_id}', 'func',
                    f'ses-{session}', f'sub-{self.subject_id}_space-T1w_desc-{roi}_mask.nii.gz'
                )

                if not op.exists(epi_mask_path):
                    _log(f"EPI mask does not exist: {epi_mask_path}, creating it.")

                    # Ensure parent directory exists
                    os.makedirs(op.dirname(epi_mask_path), exist_ok=True)

                    # Resample anatomical mask to EPI space and save
                    im = image.resample_to_img(
                        image.load_img(anat_mask_path, dtype='int32'),
                        image.load_img(base_mask, dtype='int32'),
                        interpolation='nearest'
                    )
                    im.to_filename(epi_mask_path)
                    _log(f"Saved new EPI mask: {epi_mask_path}")

                mask = epi_mask_path
            else:
                mask = anat_mask_path

        else:
            _log(f"Unknown ROI: {roi} -> Raising NotImplementedError")
            raise NotImplementedError

        _log(f"Final mask path: {mask}")

        # Load the final mask as a Nifti1Image
        mask = image.load_img(mask, dtype='int32')
        _log("Loaded final mask into Nifti1Image.")

        # Return either a NiftiMasker or the raw mask
        if return_masker:
            _log("Returning NiftiMasker")
            masker = NiftiMasker(mask_img=mask, resampling_target="data")  # Ensures compatibility with input images
            masker.fit()
            return masker

        _log("Returning final mask as Nifti1Image.")
        return mask


    def get_prf_parameters_volume(self, model_label, smoothed=True, roi=None, response_fit=False, return_image=False,
                                   raw=False, par_keys=None, censored=False, use_nifti=False):
        """Fitted encoding-model parameters, one row per voxel in the ROI mask.

        Columns carry a (parameter, range) MultiIndex with range in
        {'narrow', 'wide'} for condition-specific parameters (r2/cvr2 have no
        range level). Default fast path reads the pre-extracted group TSV in
        derivatives/extracted_pars/ (run encoding_model/extract_pars.py first);
        use_nifti=True (or custom par_keys) reads the per-parameter NIfTI maps
        directly. raw=True converts condition-specific back to raw model
        parameters; return_image=True gives the maps as a 4D NIfTI instead."""

        # Fast path: read from pre-extracted group TSV if available and no custom par_keys requested
        if par_keys is None and not use_nifti:
            import warnings
            desc = 'responses' if response_fit else 'groundtruth'
            if censored:
                desc += '.censored'
            tsv_path = op.join(self.bids_folder, 'derivatives', 'extracted_pars',
                               f'group_roi-{roi}_model-{model_label}_desc-{desc}_parameters.tsv')
            if op.exists(tsv_path):
                pars = pd.read_csv(tsv_path, sep='\t', header=[0, 1], index_col=[0, 1, 2, 3])
                # Filter by subject_id (level 0); use string comparison to avoid dtype issues
                mask = pars.index.get_level_values(0).astype(int) == int(self.subject_id)
                pars = pars.loc[mask].droplevel([0, 1, 2])
                pars.index = range(len(pars))
                pars.columns.names = ['parameter', 'range']

                if raw:
                    from neural_priors.encoding_model.fit_model import conditionspecific_to_raw_pars
                    try:
                        pars = conditionspecific_to_raw_pars(model_label, pars)
                    except KeyError as e:
                        # The extracted TSVs lack some raw parameters (e.g. alpha,
                        # lower_bound_range for AlphaDeltaModel labels <= 14) — those
                        # only exist as NIfTIs.
                        raise KeyError(
                            f'Raw parameter {e} not in the extracted TSV for model '
                            f'{model_label}; pass use_nifti=True to load it from the '
                            f'per-parameter NIfTI maps instead.') from e

                if return_image:
                    masker = self.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
                    return masker.inverse_transform(pars.T)

                return pars
            else:
                raise FileNotFoundError(f'Pre-extracted TSV not found at {tsv_path}. Run extract_pars.py first, or pass use_nifti=True to load from NIfTI files directly.')

        if par_keys is None:
            par_keys = ['mu', 'sd', 'delta_wide', 'amplitude', 'baseline']

            if raw:
                # AlphaDeltaModel stores alpha and lower_bound_range in files
                if model_label <= 14:
                    par_keys += ['alpha', 'lower_bound_range']
                # baseline_ratio is stored for these models
                if model_label in [11, 13, 16, 17, 19, 20]:
                    par_keys += ['baseline_ratio']

        # Create target folder
        key = f'model{model_label}'
        cv_key = f'model{model_label}.cv'

        if roi != 'NPCr':
            key += '.whole_brain'
            cv_key += '.whole_brain'

        if censored:
            key += '.censored'
            cv_key += '.censored'

        if smoothed:
            key += '.smoothed'
            cv_key += '.smoothed'

        if response_fit:
            key += '.fit_responses'
            cv_key += '.fit_responses'

        # NB: for roi=None this uses the ses-1 run-1 brain mask, whereas the
        # whole-brain fits (fit_model.py) were fitted inside get_brain_mask()'s
        # run-2 mask; voxels present in only one of the two masks read back as 0.
        masker = self.get_volume_mask(roi=roi, epi_space=True, return_masker=True)

        target_dir = op.join(self.bids_folder, 'derivatives', 'encoding_models', key, f'sub-{self.subject_id}', 'func')

        pars = []

        for par, condition in product_(par_keys, ['narrow', 'wide']):
            fn = op.join(target_dir, f'sub-{self.subject_id}_desc-{par}.{condition}.optim_space-T1w_pars.nii.gz')

            pars.append(pd.Series(masker.transform(fn).squeeze(), name=(par, condition)))

        pars.append(pd.Series(masker.transform(op.join(target_dir, f'sub-{self.subject_id}_desc-r2.optim_space-T1w_pars.nii.gz')).squeeze(), name=('r2', None)))

        cv_dir = op.join(self.bids_folder, 'derivatives', 'encoding_models',  cv_key, f'sub-{self.subject_id}', 'func')

        pars.append(pd.Series(masker.transform(op.join(cv_dir, f'sub-{self.subject_id}_desc-cvr2.optim_space-T1w_pars.nii.gz')).squeeze(), name=('cvr2', None)))

        pars = pd.concat(pars, axis=1, names=['parameter', 'range'])

        if raw:
            from neural_priors.encoding_model.fit_model import conditionspecific_to_raw_pars
            pars = conditionspecific_to_raw_pars(model_label, pars)

        if return_image:
            return masker.inverse_transform(pars.T)
        else:
            return pars


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

    def get_prf_parameters_surf(self, model_label, smoothed=False, hemi=None, space='fsnative', fit_responses=False):
        """Whole-brain nPRF parameters sampled onto the cortical surface
        (r2, cvr2, preferred numerosity mu per range condition), as produced by
        surface/sample_prf_to_surface_nilearn.py; used for the Fig. 2 maps."""

        parameter_keys = ['r2', 'cvr2', 'mu.narrow', 'mu.wide']

        if hemi is None:
            prf_l = self.get_prf_parameters_surf(model_label, smoothed, hemi='L', space=space)
            prf_r = self.get_prf_parameters_surf(model_label, smoothed, hemi='R', space=space)
            
            return pd.concat((prf_l, prf_r), axis=0, 
                    keys=pd.Index(['L', 'R'], name='hemi'))

        # Find target folder
        key = f'model{model_label}.whole_brain'

        if smoothed:
            key += '.smoothed'

        if fit_responses:
            key += '.fit_responses'

        dir = op.join(self.bids_folder, 'derivatives', 'encoding_models', key, f'sub-{self.subject_id}', 'func')

        parameters = []

        fn_template = op.join(dir, 'sub-{subject_id}_desc-{parameter_key}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii')

        for parameter_key in parameter_keys:

            fn = fn_template.format(parameter_key=parameter_key, subject_id=self.subject_id, hemi=hemi, space=space)

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

