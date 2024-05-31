import argparse
import cortex
import os.path as op
from cortex import webgl 
from nilearn import image
from nilearn import surface
import numpy as np
from utils import get_alpha_vertex
import matplotlib.pyplot as plt
import pandas as pd

from neural_priors.utils.data import Subject


def main(subject, bids_folder, use_cvr2=True, threshold=None, filter_extreme_prfs=True, smoothed=False, fsnative=False,
         vmin=1, vmax=40, show_colorbar=False):

    print(use_cvr2, threshold)

    sub = Subject(subject, bids_folder=bids_folder)

    if fsnative:
        space = 'fsnative'
    else:
        space = 'fsaverage'

    fs_subject = 'fsaverage' if not fsnative else f'neuralpriors.sub-{subject}'

    vertices = {}

    if use_cvr2 and (threshold is None):
        threshold = 0.0
    elif not use_cvr2 and (threshold is None):
        threshold = 0.05
    
    # for session in [1]:
    keys = ['both', 'wide', 'narrow']
    keys_ = [None, 'wide', 'narrow']

    prf_pars = []

    for key, key_ in zip(keys, keys_):
        prf_pars.append(sub.get_prf_parameters_surf(session=None, run=None,  smoothed=smoothed, nilearn=True, space=space,
                                               range_n=key_))

    prf_pars = pd.concat(prf_pars, axis=1, keys=keys)
    print(prf_pars.columns)



    if use_cvr2:
        mask1 = (prf_pars[[('wide', 'cvr2'), ('narrow', 'cvr2')]] > threshold).any(axis=1).values
    else:
        mask1 = (prf_pars[('both', 'r2')] > threshold).values

    if filter_extreme_prfs:
        pass

    for key in keys:
        if use_cvr2:
            mask2 = (prf_pars[(key, 'cvr2')] > threshold).values
        else:
            mask2 = (prf_pars[(key, 'r2')] > threshold).values

        for label, mask in enumerate([mask1, mask2]):
            vertices[f'mode_range-{key}.{label+1}'] = get_alpha_vertex(prf_pars[(key, 'mode')].values, mask, vmin=vmin, vmax=vmax, subject=fs_subject) 
            vertices[f'fwhm_range-{key}.{label+1}'] = get_alpha_vertex(prf_pars[(key, 'fwhm')].values, mask, vmin=1, vmax=50, subject=fs_subject) 
            vertices[f'amplitude_range-{key}.{label+1}'] = get_alpha_vertex(prf_pars[(key, 'amplitude')].values, mask, vmin=0, vmax=5, cmap='viridis', subject=fs_subject) 
            vertices[f'r2_range-{key}.{label+1}'] = get_alpha_vertex(prf_pars[(key, 'r2')].values, mask, cmap='hot', vmin=threshold, vmax=0.15, subject=fs_subject)
            vertices[f'cvr2_range-{key}.{label+1}'] = get_alpha_vertex(prf_pars[(key, 'cvr2')].values, mask, cmap='hot', vmin=0.0, vmax=0.15, subject=fs_subject)

    vertices = {k: v for k, v in sorted(vertices.items(), key=lambda item: item[0])}
    print(vertices)
    webgl.show(vertices)

    if show_colorbar:
        x = np.linspace(0, 1, 101, True)
        im = plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
            extent=[vmin, vmax, 0, 1], aspect=1.*(vmax-vmin) / 20.,
            origin='lower')
        print(im.get_extent())
        plt.yticks([])
        plt.tight_layout()

        # ns = np.array([5, 7, 10, 14, 20, 28, 40, 56, 80])
        # ns = ns[ns <= vmax]
        # plt.xticks(ns)
        plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--fsnative', action='store_true')
    parser.add_argument('--unsmoothed', dest='smoothed', action='store_false')
    parser.add_argument('--threshold_r2', dest='use_cvr2', action='store_false')
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--no_mu_filter', dest='filter_extreme_prfs', action='store_false')
    parser.add_argument('--show_colorbar', action='store_true')
    args = parser.parse_args()
    main(args.subject, bids_folder=args.bids_folder, use_cvr2=args.use_cvr2, threshold=args.threshold, smoothed=args.smoothed, fsnative=args.fsnative, filter_extreme_prfs=args.filter_extreme_prfs, show_colorbar=args.show_colorbar)