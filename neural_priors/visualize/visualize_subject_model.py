import cortex
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import pandas as pd
import argparse


# --- custom μ colormap "laura" — warm orange (low μ) → cool teal (high μ) --
# Registered on matplotlib so `mpl.colormaps['laura']` resolves it from
# anywhere (including get_alpha_vertex).
_LAURA_CMAP = LinearSegmentedColormap.from_list(
    'laura', [
        '#ee7a04', '#ea8660', '#e19288', '#d3a0a6', '#c4aeba',
        '#b3bbc7', '#a2c7cf', '#95d3d0', '#8eddca',
    ], N=256,
)
if 'laura' not in mpl.colormaps:
    mpl.colormaps.register(_LAURA_CMAP, name='laura')


def main(subject, model_labels, bids_folder='/data/ds-neuralpriors',
         vmin=5, vmax=25, cvr2_thr=0.0, cmap='nipy_spectral'):

    sub = Subject(subject_id=subject)


    
    ds = {}

    for model_label in model_labels:
        pars = sub.get_prf_parameters_surf(model_label=model_label, smoothed=True)
        # mask voxels whose preferred μ is outside [vmin, vmax] in EITHER
        # condition.  Without the upper bound, μ > vmax voxels would still be
        # rendered, saturated at the top of the colormap (the red blobs you
        # noticed in S9).
        out_of_range = ((pars['mu.narrow'] < vmin) | (pars['mu.wide'] < vmin)
                        | (pars['mu.narrow'] > vmax) | (pars['mu.wide'] > vmax))
        pars.loc[out_of_range, 'cvr2'] = -1.

        ds[f'{subject}.model{model_label}.cvr2_thr'] = get_alpha_vertex(pars['cvr2'].values, (pars['cvr2'] > cvr2_thr).values, vmin=0.0, vmax=.05, subject=f'neuralpriors.sub-{subject}', cmap='plasma')

        ds[f'{subject}.model{model_label}.mu.narrow'] = get_alpha_vertex(pars['mu.narrow'].values, (pars['cvr2'] > cvr2_thr).values, vmin=vmin, vmax=vmax, subject=f'neuralpriors.sub-{subject}', cmap=cmap)


        # if mu.narrow is not identical to mu.wide, we can visualize both
        if not np.allclose(pars['mu.narrow'].values, pars['mu.wide'].values):
            ds[f'{subject}.model{model_label}.mu.wide'] = get_alpha_vertex(pars['mu.wide'].values, (pars['cvr2'] > cvr2_thr).values, vmin=vmin, vmax=vmax, subject=f'neuralpriors.sub-{subject}', cmap=cmap)
        else:
            ds[f'{subject}.mu.wide'] = ds[f'{subject}.model{model_label}.mu.narrow']

    cortex.webgl.show(ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize PRF parameters for a subject.')
    parser.add_argument('subject', type=str, help='Subject ID')
    parser.add_argument('model_label', type=int, nargs='+', help='Model label to visualize')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-neuralpriors', help='BIDS folder path')
    parser.add_argument('--vmin', type=float, default=5.0, help='Minimum value for visualization')
    parser.add_argument('--vmax', type=float, default=25.0, help='Maximum value for visualization')
    parser.add_argument('--cvr2_thr', type=float, default=0.0, help='Threshold for cvr2')
    parser.add_argument('--cmap', type=str, default='nipy_spectral',
                        help=("Colormap for μ. Default: 'nipy_spectral'. "
                              "Custom warm→cool palette: 'laura'. "
                              "Other matplotlib names also work, e.g. 'viridis', 'turbo', 'plasma'."))

    args = parser.parse_args()
    main(args.subject, args.model_label, args.bids_folder, args.vmin, args.vmax,
         args.cvr2_thr, args.cmap)