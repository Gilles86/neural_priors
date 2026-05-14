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


def load_data(subject, bids_folder, roi='NPCr', stim_range='wide',
               smoothed=False):
    """Load paradigm + masked single-trial estimates indexed by (session, run2).

    Restricts to one stimulus range ('wide', 'narrow') or pools both
    ('both'). ``smoothed`` switches between the unsmoothed and spatially-
    smoothed single-trial estimates (subject pipeline already has both).
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
    data_img = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
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


def _build_priors(distance_matrix, classical_pars):
    """One GP prior per regularized parameter; variance seeded from classical.

    Initial lengthscale defaults to ~25% of the median pairwise distance
    so the kernel is non-degenerate from step 0 (cf. ‘RBF with l << d
    looks like a delta function’). Adam will adjust it during stage 2.
    """
    offdiag = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)]
    lengthscale_init = max(float(np.median(offdiag)) * 0.25, 1.0)
    priors = {}
    for name in PRIOR_PARAMS:
        v = float(np.var(classical_pars[name].values))
        v_init = max(v, 1e-4)
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


def _fdr_significant_voxels(train_data, train_pred, alpha=0.05,
                             min_voxels=100):
    """Voxels passing a tail-FDR threshold from a 2-Gaussian mixture on
    logit(R²), via :func:`braincoder.utils.stats.fit_r2_mixture`.

    Empirical-Bayes FDR — the noise and signal components are learned
    from this fold's R² distribution, not assumed to follow the
    parametric F-null. The threshold is the smallest R² at which the
    tail false-discovery rate is ≤ ``alpha``. Falls back to top
    ``min_voxels`` by R² if the fit is degenerate or the threshold is
    out of range.

    The fallback default of 100 is calibrated for noisy single-trial
    GLM data where the mixture often can't surface a clean signal
    component (~ all-noise-looking R² distribution): top-10 was too
    thin to give the residual-noise fit anything to work with.

    Returns
    -------
    keep : 1-D int array of voxel indices.
    info : dict with mixture summary + actual R² threshold + fallback flag.
    """
    from braincoder.utils.stats import fit_r2_mixture, r2_fdr_threshold

    train_data = np.asarray(train_data, dtype=np.float64)
    train_pred = np.asarray(train_pred, dtype=np.float64)
    ss_res = np.sum((train_data - train_pred) ** 2, axis=0)
    ss_tot = np.sum(
        (train_data - train_data.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2_safe = np.nan_to_num(r2, nan=-np.inf)

    fit = None
    threshold = float('inf')
    try:
        fit = fit_r2_mixture(r2)
        threshold = r2_fdr_threshold(fit, alpha=alpha)
    except ValueError:
        pass  # too few voxels — falls through to top-N

    keep = np.where(np.isfinite(r2) & (r2 > threshold))[0]
    fallback = False
    if len(keep) < min_voxels:
        keep = np.argsort(-r2_safe)[:min_voxels]
        fallback = True

    info = dict(fit) if fit is not None else {}
    info['alpha'] = float(alpha)
    info['r2_threshold'] = float(threshold)
    info['n_kept'] = int(len(keep))
    info['fallback'] = bool(fallback)
    info['r2'] = r2.astype(np.float32)
    return np.sort(keep), info


def _save_r2_mixture_diagnostic(fold_results, output_dir, subject,
                                 stim_range, alpha=0.05):
    """Multi-panel PDF: rows = methods, cols = folds. Each panel shows
    the training R² histogram + the fitted logit-Gaussian mixture +
    FDR threshold for that (fold, method).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from braincoder.utils.stats import plot_r2_mixture

    methods = ['classical', 'ml', 'bayes']
    n_folds = len(fold_results)
    if n_folds == 0:
        return
    fig, axes = plt.subplots(
        len(methods), n_folds,
        figsize=(2.8 * n_folds, 2.4 * len(methods)),
        squeeze=False)
    for i, method in enumerate(methods):
        for j, r in enumerate(fold_results):
            ax = axes[i, j]
            d = r['decoding'].get(method, {})
            info = d.get('fdr_info', {}) or {}
            r2 = info.get('r2')
            title = (f'{method} | s{r["session"]}-r{r["run2"]}'
                      if i == 0 else f's{r["session"]}-r{r["run2"]}')
            if r2 is None or 'signal_mu' not in info or info.get('fallback'):
                ax.text(0.5, 0.5, info.get('reason',
                        'fallback' if info.get('fallback') else 'no fit'),
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=8)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            plot_r2_mixture(info, r2=r2, alpha=alpha, ax=ax, title=title)
            ax.title.set_size(8)
            if j > 0:
                ax.set_ylabel('')
            if i < len(methods) - 1:
                ax.set_xlabel('')
            ax.legend_.remove() if ax.get_legend() else None
            if j == 0:
                ax.set_ylabel(f'{method}\nDensity', fontsize=9)

    fig.suptitle(f'sub-{subject} range={stim_range}: '
                  f'logit-Gaussian R² mixture per fold (α={alpha})',
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fn = op.join(output_dir,
                 f'sub-{subject}_range-{stim_range}_desc-r2_mixture.pdf')
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    return fn


def _decode_test_trials(params, train_data, train_par, test_data,
                         sig_voxels, stim_grid, max_resid_iter=2000):
    """Posterior-mean decode test trials using braincoder's standard
    Student-t residual noise model + ``get_stimulus_pdf``.

    Pipeline (matches ``neural_priors/encoding_model/decode.py`` and
    ``tms_risk/encoding_model/decode_select_voxels_cv.py``):

      1. Build a fresh ``LogGaussianPRF`` on the FDR-significant voxels
         with the method's fitted parameters baked in.
      2. ``init_pseudoWWT(stim_grid, params)`` precomputes the
         basis-weight matrix the residual fitter needs.
      3. ``ResidualFitter`` fits a multivariate Student-t noise model
         (full covariance ``omega`` and degrees of freedom ``dof``)
         on the training residuals.
      4. ``get_stimulus_pdf`` evaluates the un-normalized posterior over
         the numerosity grid for each test trial.
      5. Readout = posterior mean over ``stim_grid``.

    Returns ``None`` if no voxels pass FDR.
    """
    if len(sig_voxels) == 0:
        return None

    from braincoder.models import LogGaussianPRF
    from braincoder.optimize import ResidualFitter

    sig_voxels = np.asarray(sig_voxels, dtype=int)
    sig_params = params.iloc[sig_voxels]
    train_data_sig = train_data.iloc[:, sig_voxels].astype(np.float32)
    test_data_sig = test_data.iloc[:, sig_voxels].astype(np.float32)

    m = LogGaussianPRF(paradigm=train_par.to_frame(), parameters=sig_params)
    m.init_pseudoWWT(stim_grid, sig_params)

    residfit = ResidualFitter(m, train_data_sig, train_par,
                               parameters=sig_params)
    omega, dof = residfit.fit(
        init_sigma2=0.1, method='t',
        max_n_iterations=max_resid_iter, learning_rate=0.05,
        progressbar=False)

    pdf = m.get_stimulus_pdf(test_data_sig, stim_grid, sig_params,
                              omega=omega, dof=dof)
    # pdf columns are the stimulus grid values (floats)
    cols = pdf.columns.astype(float).values
    decoded = (pdf.values * cols[None, :]).sum(axis=1) / pdf.values.sum(axis=1)
    return decoded.astype(np.float32)


def fit_fold_bayes_no_prior(model, train_data, train_par, init_pars,
                             max_iter, progressbar):
    """Same Gaussian-likelihood + per-vertex sigma loop as fit_fold_bayes,
    but with no GP prior. This isolates the contribution of the prior
    from the contribution of the noise model / likelihood formulation.

    Uses the *same* naive init as the classical fit (not a classical
    warm start), so the head-to-head with classical is fair: equal
    optimization budget, equal starting point, only the loss differs
    (Gaussian likelihood with per-vertex sigma vs SSQ).
    """
    fitter = BayesianParameterFitter(
        model, train_data, train_par, priors={})
    # Stage-1 only needed for sigma^2 init from residuals; reuse the
    # naive init pars as if they were the classical fit so fit_map's
    # initialization path works without running another classical loop.
    fitter.classical_estimates = init_pars
    fitter.fit_map(max_n_iterations=max_iter,
                    init_pars=init_pars,
                    progressbar=progressbar)
    return fitter.map_estimates, fitter.map_sigma


def _run_one_range(subject, bids_folder, roi, stim_range, D, sub, masker,
                    max_iter, debug, output_dir, smoothed=False):
    """Fit classical + bayes across all folds, for a single stimulus range."""
    paradigm, data, _, _, _ = load_data(
        subject, bids_folder, roi=roi, stim_range=stim_range,
        smoothed=smoothed)
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
        cls_train_pred = model.predict(
            parameters=cls_pars, paradigm=train_par.to_frame())

        # 2. No-prior ML fit
        ml_pars, ml_sigma = fit_fold_bayes_no_prior(
            model, train_data, train_par, init,
            max_iter=max_iter, progressbar=False)
        ml_pred = model.predict(
            parameters=ml_pars, paradigm=test_par.to_frame())
        ml_cvr2 = get_rsq(test_data, ml_pred)
        ml_train_pred = model.predict(
            parameters=ml_pars, paradigm=train_par.to_frame())

        # 3. Bayesian fit with GP priors on every parameter
        map_pars, hyperpars_dict, sigma = fit_fold_bayes(
            model, train_data, train_par, D, cls_pars,
            max_iter=max_iter, progressbar=False)
        map_pred = model.predict(
            parameters=map_pars, paradigm=test_par.to_frame())
        map_cvr2 = get_rsq(test_data, map_pred)
        map_train_pred = model.predict(
            parameters=map_pars, paradigm=train_par.to_frame())

        # --- FDR (Beta-mixture) voxel selection + posterior-mean decoding ---
        stim_grid = np.linspace(
            float(paradigm.min()), float(paradigm.max()), 201,
            dtype=np.float32)
        true_test = test_par.values.astype(np.float32)
        decoding = {}
        decode_iter = 200 if debug else 2000
        for method, train_pred_df, fit_pars in (
                ('classical', cls_train_pred, cls_pars),
                ('ml',        ml_train_pred,  ml_pars),
                ('bayes',     map_train_pred, map_pars)):
            sig, fdr_info = _fdr_significant_voxels(
                train_data.values, train_pred_df.values)
            decoded = _decode_test_trials(
                fit_pars, train_data, train_par, test_data,
                sig, stim_grid, max_resid_iter=decode_iter)
            if decoded is None:
                decoding[method] = dict(n_sig=0, mae=np.nan,
                                         median_ae=np.nan, decoded=None,
                                         fdr_info=fdr_info)
            else:
                err = np.abs(decoded - true_test)
                decoding[method] = dict(
                    n_sig=int(len(sig)),
                    mae=float(np.mean(err)),
                    median_ae=float(np.median(err)),
                    decoded=decoded,
                    true=true_test,
                    fdr_info=fdr_info)

        print(f'  classical: train R² {cls_train_r2:.3f} | '
              f'cvR² mean {float(cls_cvr2.mean()):.3f} | '
              f'decode {decoding["classical"]["n_sig"]} vx '
              f'medAE {decoding["classical"]["median_ae"]:.2f}')
        print(f'  ml       : cvR² mean {float(ml_cvr2.mean()):.3f} | '
              f'decode {decoding["ml"]["n_sig"]} vx '
              f'medAE {decoding["ml"]["median_ae"]:.2f}')
        print(f'  bayes    : cvR² mean {float(map_cvr2.mean()):.3f} | '
              f'decode {decoding["bayes"]["n_sig"]} vx '
              f'medAE {decoding["bayes"]["median_ae"]:.2f}')
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
            'decoding': decoding,
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

    # Decoding summary: one row per (fold, method). Mixture params
    # included so we can sanity-check the empirical-null fit.
    dec_rows = []
    for r in fold_results:
        for method, d in r['decoding'].items():
            info = d.get('fdr_info', {})
            dec_rows.append(dict(
                session=r['session'], run2=r['run2'],
                method=method, n_sig_voxels=d['n_sig'],
                mae=d['mae'], median_ae=d['median_ae'],
                stim_range=stim_range,
                fdr_fallback=info.get('fallback', False),
                fdr_r2_threshold=info.get('r2_threshold', np.nan),
                fdr_noise_mean_r2=info.get('noise_mean_r2', np.nan),
                fdr_signal_mean_r2=info.get('signal_mean_r2', np.nan),
                fdr_signal_weight=info.get('signal_weight', np.nan)))
    pd.DataFrame(dec_rows).to_csv(op.join(
        output_dir, f'sub-{subject}_range-{suffix}_desc-decoding.tsv'),
        sep='\t', index=False)

    # Per-trial decoded values for later scatterplots.
    trial_rows = []
    for r in fold_results:
        for method, d in r['decoding'].items():
            if d['decoded'] is None:
                continue
            for tr, (dec, tru) in enumerate(zip(d['decoded'], d['true'])):
                trial_rows.append(dict(
                    session=r['session'], run2=r['run2'],
                    method=method, trial=tr,
                    decoded=float(dec), true=float(tru),
                    stim_range=stim_range))
    pd.DataFrame(trial_rows).to_csv(op.join(
        output_dir, f'sub-{subject}_range-{suffix}_desc-decoded_trials.tsv'),
        sep='\t', index=False)

    summary = cvr2_long.groupby('method')['cvr2'].agg(['mean', 'median'])
    print(f'\n[stim_range={stim_range}] CV R² summary:')
    print(summary.to_string())

    try:
        pdf_path = _save_r2_mixture_diagnostic(
            fold_results, output_dir, subject, stim_range)
        if pdf_path:
            print(f'Wrote R² mixture diagnostic: {pdf_path}')
    except Exception as e:
        # Plotting is a nice-to-have; never let it crash the fit job.
        print(f'WARNING: failed to write R² mixture diagnostic: {e!r}')

    return cvr2_long


def main(subject, bids_folder, roi='NPCr', stim_range='both',
         smoothed=False, max_iter=2000, debug=False, output_dir=None):
    if debug:
        max_iter = 200

    # Load once (data, mask, voxel centroids) — distance matrix is shared
    # across ranges because the voxel set doesn't depend on the paradigm.
    _, _, masker, xyz, sub = load_data(
        subject, bids_folder, roi=roi, stim_range='both', smoothed=smoothed)
    n_vx = xyz.shape[0]
    smooth_tag = 'smoothed' if smoothed else 'unsmoothed'
    print(f'ROI={roi} ({smooth_tag}): {n_vx} voxels')

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
        key = f'gp_prior_roi-{roi}'
        if smoothed:
            key += '.smoothed'
        output_dir = op.join(
            bids_folder, 'derivatives', 'encoding_models',
            key, f'sub-{subject}', 'func')
    os.makedirs(output_dir, exist_ok=True)
    np.save(op.join(output_dir, f'sub-{subject}_desc-distance.npy'), D)
    np.save(op.join(output_dir, f'sub-{subject}_desc-vertex_idx.npy'),
             vtx_idx)

    ranges = ['narrow', 'wide'] if stim_range == 'both' else [stim_range]
    all_cvr2 = []
    for r in ranges:
        all_cvr2.append(_run_one_range(
            subject, bids_folder, roi, r, D, sub, masker,
            max_iter, debug, output_dir, smoothed=smoothed))

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
    parser.add_argument('--smoothed', action='store_true',
                        help='Use the spatially-smoothed single-trial '
                             'estimates instead of the unsmoothed pipeline. '
                             'Output goes under gp_prior_roi-{ROI}.smoothed/.')
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    main(args.subject, args.bids_folder,
         roi=args.roi, stim_range=args.stim_range,
         smoothed=args.smoothed,
         max_iter=args.max_iter,
         debug=args.debug, output_dir=args.output_dir)
