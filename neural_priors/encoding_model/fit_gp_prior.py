"""Cross-validated LogGaussianPRF fit on NPC with optional GP prior on `mu`.

Compares two fitting regimes on the same data and folds:

  * classical — per-vertex MSE fit (braincoder ``ParameterFitter``)
  * bayes    — hierarchical Bayesian fit with a Gaussian-Process prior
                on ``mu`` over geodesic distance on the cortical surface
                (braincoder ``BayesianParameterFitter``)

For both, cross-validated R² is computed by leaving one
(session, run2) combination out, fitting on the remaining trials, and
predicting the held-out trials with the fitted parameters. The two
regimes are compared on identical folds for a fair head-to-head.

Distance handling
-----------------
NPC is masked in volume space (single-trial GLM estimates), so we have
voxel-centroid mm coordinates in T1w space. For each voxel centroid we
find the nearest vertex on the hemisphere's white-matter surface
(fmriprep's ``*_hemi-{L,R}_white.surf.gii`` is in T1w space, so the
frames agree directly), then compute pairwise geodesic distances by
running Dijkstra on the triangulated mesh, restricted to the matched
vertices. This gives a proper on-cortex distance metric rather than the
straight-line Euclidean one that would cross sulci.

Notes
-----
* The model is the built-in braincoder ``LogGaussianPRF``
  (``mu_sd_natural`` parameterisation) — not the custom AlphaDelta zoo,
  so the comparison is purely about the prior, not the model variant.
* Narrow vs wide stimulus ranges are handled by restricting to one
  range at a time (default ``wide``, since it covers the full numerosity
  span). Pooling both ranges with a single tuning curve assumes no
  range-context effect on the population code; we sidestep that here.
  Use ``--range both`` to pool anyway, or ``--range narrow`` for the
  narrow set.
* Single hemisphere at a time. For bilateral NPC, run twice (NPCl, NPCr).
"""

import argparse
import os
import os.path as op
import pickle

import numpy as np
import pandas as pd
import nibabel as nib

from neural_priors.utils.data import Subject
from neural_priors.encoding_model.fit_model import get_paradigm

from scipy.spatial import cKDTree

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.optimize.gp_prior import GeodesicGPPrior
from braincoder.optimize.bayesian_fitter import BayesianParameterFitter
from braincoder.utils import get_rsq
from braincoder.utils.cortex import geodesic_distance_matrix


def voxel_centroids_mm(masker):
    """Scanner-space (mm) coordinates of the in-mask voxels."""
    mask_img = masker.mask_img_
    affine = mask_img.affine
    mask_data = mask_img.get_fdata().astype(bool)
    ijk = np.argwhere(mask_data)
    xyz = nib.affines.apply_affine(affine, ijk).astype(np.float32)
    return xyz


def load_white_surface(sub, hemi_letter):
    """Load fmriprep's white-matter GIfTI surface for one hemisphere.

    Returns ``(vertices_mm, faces)`` in the same T1w space as the EPI
    mask, so coordinates can be matched directly against voxel
    centroids without any extra registration step.
    """
    surf_info = sub.get_surf_info()
    gii = nib.load(surf_info[hemi_letter]['inner'])  # white
    vertices = gii.darrays[0].data.astype(np.float32)
    faces = gii.darrays[1].data.astype(np.int64)
    return vertices, faces


def cortical_distance_matrix(xyz, vertices, faces, progressbar=True):
    """Project voxel centroids to nearest surface vertex, return geodesics.

    Also returns the matched vertex indices and the voxel→vertex
    snap-distance (useful as a sanity check — anything more than ~3 mm
    suggests a coregistration problem).
    """
    tree = cKDTree(vertices)
    snap_dist, vtx_idx = tree.query(xyz, k=1)
    vtx_idx = np.asarray(vtx_idx, dtype=int)
    snap_dist = np.asarray(snap_dist, dtype=np.float32)

    # Deduplicate: occasionally multiple voxels project to the same vertex
    # (cortex is folded, voxels are 3-D). Keep the first occurrence; the
    # GP prior is over unique vertices, so we'll map duplicates back later.
    unique_vtx, inverse = np.unique(vtx_idx, return_inverse=True)
    D_unique = geodesic_distance_matrix(
        vertices, faces, source_indices=unique_vtx,
        progressbar=progressbar)

    # Expand back to voxel-indexed (n_vox, n_vox) by re-indexing through `inverse`.
    D = D_unique[np.ix_(inverse, inverse)].astype(np.float32)
    np.fill_diagonal(D, 0.0)
    return D, vtx_idx, snap_dist


def _roi_to_hemi_letter(roi):
    """Decide which hemisphere a single-hemi NPC ROI lives on."""
    if roi.endswith('r') or roi.endswith('R'):
        return 'R'
    if roi.endswith('l') or roi.endswith('L'):
        return 'L'
    raise ValueError(
        f"Cannot determine hemisphere from roi={roi!r}. "
        f"Use NPCl or NPCr (single hemi); for bilateral, run twice.")


def initial_pars(n_vx, paradigm):
    """Naive per-voxel init: mu at paradigm mean, sd at paradigm range/4."""
    x = paradigm.astype(np.float32)
    mu_init = float(x.mean())
    sd_init = max(float((x.max() - x.min()) / 4.0), 1.0)
    return pd.DataFrame({
        'mu':        np.full(n_vx, mu_init, dtype=np.float32),
        'sd':        np.full(n_vx, sd_init, dtype=np.float32),
        'amplitude': np.ones(n_vx, dtype=np.float32),
        'baseline':  np.zeros(n_vx, dtype=np.float32),
    })


def load_data(subject, bids_folder, roi='NPCr', stim_range='wide'):
    """Load paradigm + masked single-trial estimates indexed by (session, run2).

    Restricts to one stimulus range ('wide', 'narrow') or pools both
    ('both'). The (session, run2) index is used for leave-one-out CV.
    """
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm_full = get_paradigm(sub, fit_responses=False)
    # 'range' was mapped {narrow: False, wide: True} in get_paradigm.
    if stim_range == 'wide':
        keep = np.asarray(paradigm_full['range'] == 1.0)
    elif stim_range == 'narrow':
        keep = np.asarray(paradigm_full['range'] == 0.0)
    elif stim_range == 'both':
        keep = np.ones(len(paradigm_full), dtype=bool)
    else:
        raise ValueError(
            f"--range must be wide/narrow/both, got {stim_range!r}")
    paradigm_full = paradigm_full.loc[keep]

    # Add the run2 level the existing cv pipeline uses.
    runs = paradigm_full.index.get_level_values('run')
    paradigm_full = paradigm_full.set_index(
        pd.Index((runs - 1) % 4 + 1, name='run2'), append=True)
    paradigm_full.index = paradigm_full.index.swaplevel('run', 'run2')
    paradigm_full = paradigm_full.droplevel(['run', 'trial_nr', 'subject'])
    paradigm = paradigm_full['x'].astype(np.float32)

    # Mask single-trial estimates; subset to the same trials we kept above.
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data_img = sub.get_single_trial_estimates(session=None, smoothed=False)
    data_2d = masker.fit_transform(data_img).astype(np.float32)
    data_full = pd.DataFrame(data_2d, index=get_paradigm(sub).index)
    # Subset to the same trials we kept; reindex to paradigm's CV index.
    data = data_full.iloc[keep].copy()
    data.index = paradigm.index
    data.columns.name = 'voxel'

    xyz = voxel_centroids_mm(masker)
    return paradigm, data, masker, xyz, sub


PRIOR_PARAMS = ['mu', 'sd', 'amplitude', 'baseline']


def fit_fold_classical(model, train_data, train_par, init_pars,
                        max_iter, progressbar):
    fitter = ParameterFitter(model, train_data, train_par, log_dir=False)
    pars = fitter.fit(init_pars=init_pars, max_n_iterations=max_iter,
                      progressbar=progressbar)
    return pars, float(fitter.r2.mean())


def _build_priors(distance_matrix, classical_pars, lengthscale_init=5.0):
    """One GP prior per regularized parameter; variance seeded from classical."""
    priors = {}
    for name in PRIOR_PARAMS:
        v = float(np.var(classical_pars[name].values))
        v_init = max(v, 1e-4)               # avoid zero-variance init
        nugget_init = max(v_init * 0.1, 1e-4)
        priors[name] = GeodesicGPPrior(
            distance_matrix,
            lengthscale_init=lengthscale_init,
            variance_init=v_init,
            nugget_init=nugget_init)
    return priors


def fit_fold_bayes(model, train_data, train_par, distance_matrix,
                   classical_pars, max_iter, progressbar):
    """Stage-2+3 fit with GP priors on every parameter in PRIOR_PARAMS."""
    priors = _build_priors(distance_matrix, classical_pars)
    fitter = BayesianParameterFitter(
        model, train_data, train_par, priors=priors)
    # Reuse the classical fold's parameters as the stage-1 result and
    # skip straight to stages 2 and 3.
    fitter.classical_estimates = classical_pars
    fitter.fit_hyperparameters(progressbar=progressbar)
    fitter.fit_map(max_n_iterations=max_iter, progressbar=progressbar)
    return (fitter.map_estimates,
            {name: priors[name].hyperparameters for name in PRIOR_PARAMS},
            fitter.map_sigma)


def fit_fold_bayes_no_prior(model, train_data, train_par, classical_pars,
                             max_iter, progressbar):
    """Same Gaussian-likelihood + per-vertex sigma loop as fit_fold_bayes,
    but with no GP prior. This isolates the contribution of the prior
    from the contribution of the noise model / likelihood formulation.
    """
    fitter = BayesianParameterFitter(
        model, train_data, train_par, priors={})
    fitter.classical_estimates = classical_pars
    fitter.fit_map(max_n_iterations=max_iter, progressbar=progressbar)
    return fitter.map_estimates, fitter.map_sigma


def _run_one_range(subject, bids_folder, roi, stim_range, D, sub, masker,
                    max_iter, debug, output_dir):
    """Fit classical + bayes across all folds, for a single stimulus range."""
    paradigm, data, _, _, _ = load_data(
        subject, bids_folder, roi=roi, stim_range=stim_range)
    n_vx = data.shape[1]
    print(f'\n>>> stim_range={stim_range}: '
          f'{data.shape[0]} trials × {n_vx} voxels')

    init = initial_pars(n_vx, paradigm)
    model = LogGaussianPRF(paradigm=paradigm.to_frame())

    folds = list(paradigm.index.unique())
    if debug:
        folds = folds[:2]
        print(f'DEBUG: restricting to {len(folds)} folds')

    fold_results = []
    for fold in folds:
        sess, run2 = fold
        print(f'\n=== Fold (session={sess}, run2={run2}) ===')

        test_data = data.loc[fold]
        test_par = paradigm.loc[fold]
        train_data = data.drop(fold)
        train_par = paradigm.drop(fold)

        # 1. Classical SSQ fit
        cls_pars, cls_train_r2 = fit_fold_classical(
            model, train_data, train_par, init,
            max_iter=max_iter, progressbar=False)
        cls_pred = model.predict(
            parameters=cls_pars, paradigm=test_par.to_frame())
        cls_cvr2 = get_rsq(test_data, cls_pred)

        # 2. No-prior ML fit: same Gaussian-likelihood loop as Bayes,
        #    same init pars, no GP prior — isolates the prior's effect.
        ml_pars, ml_sigma = fit_fold_bayes_no_prior(
            model, train_data, train_par, cls_pars,
            max_iter=max_iter, progressbar=False)
        ml_pred = model.predict(
            parameters=ml_pars, paradigm=test_par.to_frame())
        ml_cvr2 = get_rsq(test_data, ml_pred)

        # 3. Bayesian fit with GP priors on every parameter
        map_pars, hyperpars_dict, sigma = fit_fold_bayes(
            model, train_data, train_par, D, cls_pars,
            max_iter=max_iter, progressbar=False)
        map_pred = model.predict(
            parameters=map_pars, paradigm=test_par.to_frame())
        map_cvr2 = get_rsq(test_data, map_pred)

        print(f'  classical: train R² {cls_train_r2:.3f} | '
              f'cvR² mean {float(cls_cvr2.mean()):.3f}')
        print(f'  ml(no-prior): cvR² mean {float(ml_cvr2.mean()):.3f}')
        print(f'  bayes      : cvR² mean {float(map_cvr2.mean()):.3f}')
        for name, hp in hyperpars_dict.items():
            print(f'    prior[{name}]: l={hp["lengthscale"]:.2f} mm, '
                  f'v={hp["variance"]:.3f}, nug={hp["nugget"]:.3f}')

        fold_results.append({
            'session': sess,
            'run2': run2,
            'classical_params': cls_pars,
            'ml_params': ml_pars,
            'bayes_params': map_pars,
            'classical_cvr2': cls_cvr2,
            'ml_cvr2': ml_cvr2,
            'bayes_cvr2': map_cvr2,
            'hyperparameters': hyperpars_dict,
            'ml_sigma': ml_sigma,
            'bayes_sigma': sigma,
        })

    # ------- Save outputs per range -------
    suffix = stim_range
    with open(op.join(output_dir,
                       f'sub-{subject}_range-{suffix}_desc-folds.pkl'),
              'wb') as f:
        pickle.dump({
            'subject': subject,
            'roi': roi,
            'stim_range': stim_range,
            'folds': fold_results,
        }, f)

    rows = []
    for r in fold_results:
        for method, cvr2 in (('classical', r['classical_cvr2']),
                              ('ml',        r['ml_cvr2']),
                              ('bayes',     r['bayes_cvr2'])):
            for vox, val in cvr2.items():
                rows.append(dict(
                    session=r['session'], run2=r['run2'],
                    voxel=int(vox), method=method,
                    stim_range=stim_range, cvr2=float(val)))
    cvr2_long = pd.DataFrame(rows)
    cvr2_long.to_csv(op.join(
        output_dir, f'sub-{subject}_range-{suffix}_desc-cvr2.tsv'),
        sep='\t', index=False)

    hp_rows = []
    for r in fold_results:
        for pname, hp in r['hyperparameters'].items():
            hp_rows.append(dict(session=r['session'], run2=r['run2'],
                                 parameter=pname, **hp))
    pd.DataFrame(hp_rows).to_csv(op.join(
        output_dir, f'sub-{subject}_range-{suffix}_desc-hyperpars.tsv'),
        sep='\t', index=False)

    summary = cvr2_long.groupby('method')['cvr2'].agg(['mean', 'median'])
    print(f'\n[stim_range={stim_range}] CV R² summary:')
    print(summary.to_string())
    return cvr2_long


def main(subject, bids_folder, roi='NPCr', stim_range='both',
         max_iter=2000, debug=False, output_dir=None):
    if debug:
        max_iter = 200

    # Load once (data, mask, voxel centroids) — distance matrix is shared
    # across ranges because the voxel set doesn't depend on the paradigm.
    _, _, masker, xyz, sub = load_data(
        subject, bids_folder, roi=roi, stim_range='both')
    n_vx = xyz.shape[0]
    print(f'ROI={roi}: {n_vx} voxels')

    # Geodesic distances via nearest-vertex projection.
    hemi = _roi_to_hemi_letter(roi)
    vertices, faces = load_white_surface(sub, hemi)
    D, vtx_idx, snap_dist = cortical_distance_matrix(
        xyz, vertices, faces, progressbar=not debug)
    print(f'Nearest-vertex snap distance: median {np.median(snap_dist):.2f} mm,'
          f' max {snap_dist.max():.2f} mm')
    if snap_dist.max() > 5.0:
        print('WARNING: max snap distance > 5 mm — check coregistration')
    print(f'Cortical distance matrix: {D.shape}, '
          f'median off-diag {np.median(D[D > 0]):.1f} mm, '
          f'max {D.max():.1f} mm')

    if output_dir is None:
        output_dir = op.join(
            bids_folder, 'derivatives', 'encoding_models',
            f'gp_prior_roi-{roi}', f'sub-{subject}', 'func')
    os.makedirs(output_dir, exist_ok=True)
    np.save(op.join(output_dir, f'sub-{subject}_desc-distance.npy'), D)
    np.save(op.join(output_dir, f'sub-{subject}_desc-vertex_idx.npy'),
             vtx_idx)

    ranges = ['narrow', 'wide'] if stim_range == 'both' else [stim_range]
    all_cvr2 = []
    for r in ranges:
        all_cvr2.append(_run_one_range(
            subject, bids_folder, roi, r, D, sub, masker,
            max_iter, debug, output_dir))

    pd.concat(all_cvr2, ignore_index=True).to_csv(
        op.join(output_dir, f'sub-{subject}_desc-cvr2_all.tsv'),
        sep='\t', index=False)
    print(f'\nWrote outputs to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--roi', default='NPCr')
    parser.add_argument('--range', dest='stim_range',
                        choices=['narrow', 'wide', 'both'], default='both',
                        help='Which stimulus range(s) to fit. "both" runs '
                             'narrow and wide separately.')
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    main(args.subject, args.bids_folder,
         roi=args.roi, stim_range=args.stim_range,
         max_iter=args.max_iter,
         debug=args.debug, output_dir=args.output_dir)
