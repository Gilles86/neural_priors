import os
import os.path as op
import argparse
from nipype.interfaces.freesurfer import SurfaceTransform
import nipype.pipeline.engine as pe
from nilearn import surface
from neuropythy.freesurfer import subject as fs_subject
from neuropythy.io import load, save
from neuropythy.mri import (is_image, is_image_spec, image_clear, to_image)
import numpy as np

def main(subject, bids_folder, roi='npc1'):

    subjects_dir = op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer')

    def transform_surface(in_file,
            out_file, 
            target_subject,
            hemi,
            source_subject='fsaverage'):

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = out_file
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = target_subject
        sxfm.inputs.hemi = hemi

        r = sxfm.run()
        return r


    mask_data = []
    # for hemi, fs_hemi in [('L', 'lh'), ('R', 'rh')]:
    for hemi, fs_hemi in [('L', 'lh'), ('R', 'rh')]:
        in_file = op.join(bids_folder, 'derivatives', 'surface_masks', f'desc-{roi.upper()}_{hemi}_space-fsaverage_hemi-{fs_hemi}.label.gii')

        out_file = op.join(subjects_dir, f'sub-{subject}', 'surf', f'{fs_hemi}.{roi}.mgz')

        transform_surface(in_file, out_file, f'sub-{subject}', fs_hemi)
        mask_data.append(surface.load_surf_data(out_file))



    target_dir = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}',
            'anat')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    target_fn = op.join(target_dir, f'sub-{subject}_space-T1w_desc-{roi}_mask.nii.gz')
    sub = fs_subject(op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer', f'sub-{subject}'))
    im = load(op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat', f'sub-{subject}_desc-preproc_T1w.nii.gz'))
    im = to_image(image_clear(im, fill=0.0), dtype=np.int32)

    print('Generating volume...')
    new_im = sub.cortex_to_image(tuple(mask_data),
            im,
            hemi=None,
            method='nearest',
            fill=0.0)

    print('Exporting volume file: %s' % target_fn)
    save(target_fn, new_im)
    print('surface_to_image complete!')

    target_fn = op.join(target_dir, f'sub-{subject}_space-T1w_desc-{roi}l_mask.nii.gz')
    print('Generating volume...')
    new_im = sub.cortex_to_image(mask_data[0],
            im,
            hemi='lh',
            method='nearest',
            fill=0.0)

    print('Exporting volume file: %s' % target_fn)
    save(target_fn, new_im)
    print('surface_to_image complete!')

    target_fn = op.join(target_dir, f'sub-{subject}_space-T1w_desc-{roi}r_mask.nii.gz')
    print('Generating volume...')
    new_im = sub.cortex_to_image((np.zeros_like(mask_data[0]), mask_data[1]),
            im,
            hemi=None,
            method='nearest',
            fill=0.0)

    print('Exporting volume file: %s' % target_fn)
    save(target_fn, new_im)
    print('surface_to_image complete!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--roi', default='NPC')
    parser.add_argument(
        '--bids_folder', default='/data/ds-neuralpriors')
    args = parser.parse_args()

    main(args.subject, args.bids_folder, roi=args.roi)
