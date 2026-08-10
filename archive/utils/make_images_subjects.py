import os.path as op
import os
import sys
import numpy as np
import nibabel as nib
import imageio

def save_slices_as_png(img_data, output_folder, orientation):
    # Create directory for orientation if it doesn't exist
    dir_path = os.path.join(output_folder, orientation)
    os.makedirs(dir_path, exist_ok=True)

    # Determine the axis for slicing
    axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    axis = axes[orientation]

    # Number of slices
    num_slices = img_data.shape[axis]

    # Generate and save each slice as a PNG file
    for i in range(num_slices):
        slice_data = np.rot90(np.take(img_data, i, axis=axis))
        image_path = os.path.join(dir_path, f'{orientation}_slice_{i:03d}.png')
        imageio.imwrite(image_path, slice_data)

def create_gif_from_slices(input_folder, output_gif):
    # Gather all slice images from the folder
    images = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.png'):
            file_path = os.path.join(input_folder, filename)
            images.append(imageio.imread(file_path))
    # Write the images to a GIF file
    imageio.mimsave(output_gif, images, duration=0.1)

def main(subject_number):
    # Format the file path with the subject number
    nifti_file = f'/data/ds-neuralpriors/derivatives/fmriprep/sub-{subject_number}/ses-1/anat/sub-{subject_number}_ses-1_desc-preproc_T1w.nii.gz'
    img = nib.load(nifti_file)
    img_data = img.get_fdata()

    output_folder = f'/data/ds-neuralpriors/derivatives/subject_images/sub-{subject_number}/'

    if not op.exists(output_folder):
        os.makedirs(output_folder) 

    # Process each orientation
    for orientation in ['sagittal', 'coronal', 'axial']:
        save_slices_as_png(img_data, output_folder, orientation)
        input_folder = os.path.join(output_folder, orientation)
        output_gif = os.path.join(output_folder, f'{orientation}.gif')
        create_gif_from_slices(input_folder, output_gif)

    print(f"PNG slices and GIF animations have been created for subject {subject_number}.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <subject_number>")
        sys.exit(1)
    subject_number = sys.argv[1]
    main(subject_number)
