import argparse
import cortex
import os.path as op
from cortex import webgl 
from nilearn import image
from nilearn import surface
import numpy as np
from utils import get_alpha_vertex
import matplotlib.pyplot as plt

from neural_priors.utils.data import Subject


def main(subject, bids_folder, use_cvr2=True, threshold=None, filter_extreme_prfs=True, smoothed=False, fsnative=False,
         vmin=5, vmax=40, show_colorbar=False):

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
    
    for session in [1]:
        prf_pars = sub.get_prf_parameters_surf(session, None,  smoothed=smoothed, nilearn=True, space=space)
        print(prf_pars.head())
        print(prf_pars.describe())

        if use_cvr2:
            mask = (prf_pars['cvr2']  > threshold).values
        else:
            mask = (prf_pars['r2']  > threshold).values

        if filter_extreme_prfs:
            print("Filtering extreme prfs")
            mask = mask & (prf_pars['mu'] > vmin).values & (prf_pars['mu'] < vmax).values

        mu_vertex = get_alpha_vertex(prf_pars['mu'].values, mask, vmin=vmin, vmax=vmax, subject=fs_subject) 
        sd_vertex = get_alpha_vertex(prf_pars['sd'].values, mask, vmin=5, vmax=20, subject=fs_subject) 
        r2_vertex = get_alpha_vertex(prf_pars['r2'].values, mask, cmap='hot', vmin=threshold, vmax=0.15, subject=fs_subject)
        cvr2_vertex = get_alpha_vertex(prf_pars['cvr2'].values, mask, cmap='hot', vmin=0.0, vmax=0.15, subject=fs_subject)

        vertices[f"mu_vertex_session_{session}"] = mu_vertex
        vertices[f"sd_vertex_session_{session}"] = sd_vertex
        vertices[f"r2_vertex_session_{session}"] = r2_vertex
        vertices[f"cvr2_vertex_session_{session}"] = cvr2_vertex

    vertices = {k: v for k, v in sorted(vertices.items(), key=lambda item: item[0])}
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