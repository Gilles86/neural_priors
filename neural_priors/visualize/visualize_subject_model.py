import cortex
import numpy as np
import matplotlib.pyplot as plt
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import pandas as pd
import argparse



def main(subject, model_labels, bids_folder='/data/ds-neuralpriors', vmin=5, vmax=25, cvr2_thr=0.0):

    sub = Subject(subject_id=subject)


    
    ds = {}

    for model_label in model_labels:
        pars = sub.get_prf_parameters_surf2(model_label=model_label, smoothed=True)
        pars.loc[(pars['mu.narrow'] < vmin) | (pars['mu.wide'] < vmin), 'cvr2'] = -1.

        ds[f'{subject}.model{model_label}.cvr2_thr'] = get_alpha_vertex(pars['cvr2'].values, (pars['cvr2'] > cvr2_thr).values, vmin=0.0, vmax=.05, subject=f'neuralpriors.sub-{subject}', cmap='plasma')

        ds[f'{subject}.model{model_label}.mu.narrow'] = get_alpha_vertex(pars['mu.narrow'].values, (pars['cvr2'] > cvr2_thr).values, vmin=vmin, vmax=25, subject=f'neuralpriors.sub-{subject}', cmap='nipy_spectral')
    

        # if mu.narrow is not identical to mu.wide, we can visualize both
        if not np.allclose(pars['mu.narrow'].values, pars['mu.wide'].values):
            ds[f'{subject}.model{model_label}.mu.wide'] = get_alpha_vertex(pars['mu.wide'].values, (pars['cvr2'] > cvr2_thr).values, vmin=vmin, vmax=25, subject=f'neuralpriors.sub-{subject}', cmap='nipy_spectral')
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

    args = parser.parse_args()
    main(args.subject, args.model_label, args.bids_folder, args.vmin, args.vmax, args.cvr2_thr)