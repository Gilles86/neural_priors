"""Sample whole-brain nPRF parameter volumes onto the cortical surface.

Per hemisphere, nilearn's vol_to_surf averages the T1w-space parameter
maps (mu, sd, amplitude, baseline per range condition, plus r2/cvr2)
along the white-to-pial axis, yielding fsnative vertex values; the
mu/sd/r2/cvr2 maps are then resampled fsnative -> fsaverage with
FreeSurfer's SurfaceTransform. The resulting .func.gii files feed
Subject.get_prf_parameters_surf() and the Fig. 2 surface maps
(visualize/visualize_*.py).
"""
import argparse
import os.path as op
from neural_priors.utils.data import Subject
from nilearn import surface
import nibabel as nb
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform
import numpy as np

def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder):

        subjects_dir = op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer')

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = in_file.replace('fsnative', 'fsaverage')
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = 'fsaverage'
        sxfm.inputs.hemi = fs_hemi

        r = sxfm.run()
        return r

def main(subject, model_label, bids_folder, smoothed, keys_to_extract=None):

    sub = Subject(subject, bids_folder=bids_folder)
    surfinfo = sub.get_surf_info()

    prf_pars_volume = sub.get_prf_parameters_volume(model_label=model_label, smoothed=smoothed, return_image=True,
                                                    par_keys = ['mu', 'sd', 'amplitude', 'baseline'])

    key = f'model{model_label}.whole_brain'

    if smoothed:
        key += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', 'encoding_models', key, f'sub-{subject}', 'func')

    print(f'Writing to {target_dir}')

    par_keys = ['mu.narrow', 'mu.wide',
                'sd.narrow', 'sd.wide',
                'amplitude.narrow', 'amplitude.wide',
                'baseline.narrow', 'baseline.wide',
                'r2', 'cvr2']

    #	mu	sd	delta_wide	amplitude	baseline	r2	cvr2

    if keys_to_extract is None:
        keys_to_extract = par_keys

    for hemi in ['L', 'R']:
        samples = surface.vol_to_surf(prf_pars_volume, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
        samples = samples.astype(np.float32)
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        for ix, par in enumerate(par_keys):
            if par in keys_to_extract:
                im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples[:, ix])])
                
                target_fn =  op.join(target_dir, f'sub-{subject}_desc-{par}.optim.nilearn_space-fsnative_hemi-{hemi}.func.gii')

                nb.save(im, target_fn)

                if par not in ['amplitude.narrow', 'amplitude.wide', 'baseline.narrow', 'baseline.wide']:
                    # Transform to fsaverage
                    print(f'Transforming {target_fn} to fsaverage')
                    transform_fsaverage(target_fn, fs_hemi, f'sub-{subject}', bids_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('model_label', default=1, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.model_label, bids_folder=args.bids_folder, smoothed=args.smoothed)
