"""
Group-level pycortex webGL viewer on fsaverage (Fig. 2a): shows the
whole-brain group GIfTIs (r2, cvr2_count, preferred numerosity per condition)
written by combine_subjects.ipynb, thresholded at r2 > 0.01 and
cvr2_count > 0.2. Run from this directory (`from utils import ...`).
"""

import cortex
import numpy as np
import matplotlib.pyplot as plt
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import pandas as pd
import argparse
from pathlib import Path
from nilearn import surface



def main(model_labels, bids_folder='/data/ds-neuralpriors', vmin=10, vmax=25, cvr2_thr=0.0):

    ds = {}

    bids_folder = Path(bids_folder)

    vmin_vmax = {'r2': (0.0, 0.05), 'cvr2_count': (0.1, 1.0),
                 'mu.narrow': (vmin, vmax), 'mu.wide': (vmin, vmax)}

    thresholds = {'r2': 0.01, 'cvr2_count': 0.2}

    cmaps = {'r2': 'inferno', 'cvr2_count': 'magma',
                'mu.narrow': 'nipy_spectral', 'mu.wide': 'nipy_spectral'}

    for model_label in model_labels:

        pars = {}

        for parameter in ['r2', 'cvr2_count', 'mu.narrow', 'mu.wide']:

            pars[parameter] = []

            for hemi in ['L', 'R']:
                fn = bids_folder / f'derivatives/encoding_models/model{model_label}.whole_brain.smoothed/group_{parameter}_hemi-{hemi}.pars.gii'
                pars[parameter].append(surface.load_surf_data(fn))

            pars[parameter] = np.concatenate(pars[parameter], axis=0)

        pars = pd.DataFrame(pars).astype(np.float32)

        for p in pars.columns:

            if p in ['cvr2_count']:
                mask = pars[p] > thresholds[p]
            else:
                mask = pars['r2'] > thresholds['r2']

            ds[f'model{model_label}.{p}'] = get_alpha_vertex(
                pars[p].values, mask.values, vmin=vmin_vmax[p][0], vmax=vmin_vmax[p][1],
                subject=f'fsaverage', cmap=cmaps[p])

    cortex.webgl.show(ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize PRF parameters for a subject.')
    parser.add_argument('model_labels', type=int, nargs='+', help='Model label to visualize')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-neuralpriors', help='BIDS folder path')
    parser.add_argument('--vmin', type=float, default=5.0, help='Minimum value for visualization')
    parser.add_argument('--vmax', type=float, default=25.0, help='Maximum value for visualization')
    parser.add_argument('--cvr2_thr', type=float, default=0.0, help='Threshold for cvr2')

    args = parser.parse_args()
    main(args.model_labels, args.bids_folder, args.vmin, args.vmax, args.cvr2_thr)