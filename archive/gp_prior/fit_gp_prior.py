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
import datetime
import json
import os
import os.path as op
import pickle
import subprocess

import numpy as np
import pandas as pd
import nibabel as nib

from neural_priors.utils.data import Subject
from neural_priors.encoding_model.fit_model import (
    get_paradigm, get_model, fit_model as fit_model_npr)

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


DEFAULT_PRIOR_PARAMS = ['mu', 'sd', 'amplitude', 'baseline']
ALLOWED_PRIOR_PARAMS = ['mu', 'sd', 'amplitude', 'baseline']


def fit_fold_classical(model, train_data, train_par, init_pars,
                        max_iter, progressbar):
    fitter = ParameterFitter(model, train_data, train_par, log_dir=False)
    pars = fitter.fit(init_pars=init_pars, max_n_iterations=max_iter,
                      progressbar=progressbar)
    return pars, float(fitter.r2.mean())


def _build_priors(distance_matrix, classical_pars, prior_params):
    """Build one ``GeodesicGPPrior`` per name in ``prior_params``.

    Variance is seeded from each parameter's empirical variance on the
    classical estimates (so a parameter with native scale ~30 starts
    with a big variance and one near 0 with a small one). Initial
    lengthscale defaults to ~25% of the median pairwise distance so
    the kernel is non-degenerate from step 0 (cf. 'RBF with l << d
    looks like a delta function'). Adam then adjusts l/v/n during
    stage 2.

    Caller-supplied ``prior_params`` lets us run subsets — the
    paper-faithful ``--prior_params mu``, or the all-four default.
    Empty list returns ``{}`` (i.e., pure ML mode with no GP prior).
    """
    if not prior_params:
        return {}
    missing = [n for n in prior_params if n not in classical_pars.columns]
    if missing:
        raise ValueError(
            f"prior_params {missing} not in classical_pars columns "
            f"{list(classical_pars.columns)}")

    offdiag = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)]
    lengthscale_init = max(float(np.median(offdiag)) * 0.25, 1.0)
    priors = {}
    for name in prior_params:
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
                   classical_pars, max_iter, progressbar,
                   shared_lengthscale=False,
                   prior_params=None,
                   joint_hyperparams=False):
    """Stage-2+3 fit with GP priors on the requested PRF parameters.

    ``prior_params``: which parameters get a GP prior. Defaults to
    ``DEFAULT_PRIOR_PARAMS`` (all four — mu, sd, amplitude, baseline).
    Use ``['mu']`` for the paper-faithful single-parameter variant.

    ``shared_lengthscale``: tie all priors' lengthscales to one shared
    Variable.

    ``joint_hyperparams``: type-II MAP — skip stage 2 (no MLE on noisy
    classical estimates) and co-optimize hyperparameters with model
    parameters in stage 3. The prior's ``-½ log|K(ψ)|`` term provides
    automatic regularization on ψ, sidestepping stage-2's
    over-smoothing bias.
    """
    if prior_params is None:
        prior_params = list(DEFAULT_PRIOR_PARAMS)
    priors = _build_priors(distance_matrix, classical_pars, prior_params)
    fitter = BayesianParameterFitter(
        model, train_data, train_par, priors=priors)
    fitter.classical_estimates = classical_pars
    if joint_hyperparams:
        if shared_lengthscale:
            fitter.tie_lengthscales()
    else:
        fitter.fit_hyperparameters(progressbar=progressbar,
                                    shared_lengthscale=shared_lengthscale)
    fitter.fit_map(max_n_iterations=max_iter, progressbar=progressbar,
                    joint_hyperparams=joint_hyperparams)
    return (fitter.map_estimates,
            {name: priors[name].hyperparameters for name in prior_params},
            fitter.map_sigma)


def fit_brain_r2_threshold(subject, bids_folder, p_threshold=0.5,
                             wb_model_label=15):
    """Fit logit-Gaussian R² mixture on whole-brain cvR² from the
    existing neural_priors pipeline and return the R² value at which
    P(signal | r²) first crosses ``p_threshold`` (default 0.5 = Bayes
    classification boundary between signal and noise components).

    Whole-brain cvR² gives a much cleaner empirical-null distribution
    than NPC alone (lots of genuinely-noise voxels in white matter,
    ventricles, non-task regions), so the mixture finds a sensible
    signal/noise split rather than collapsing to fallback. The
    resulting threshold is then applied to our per-fold NPC training
    R²s for voxel selection.

    Default ``wb_model_label=15`` is the user's standard
    LinearScalingModel whole-brain fit. The R² scale isn't identical to
    our LogGaussianPRF but the bimodal *shape* of the noise/signal
    distribution is what matters for the threshold.
    """
    from braincoder.utils.stats import fit_r2_mixture, r2_p_signal_threshold
    from nilearn import image

    key = f'model{wb_model_label}.cv.whole_brain.smoothed'
    fn = op.join(bids_folder, 'derivatives', 'encoding_models', key,
                  f'sub-{subject}', 'func',
                  f'sub-{subject}_desc-cvr2.optim_space-T1w_pars.nii.gz')
    if not op.exists(fn):
        print(f'WARNING: no whole-brain cvR² at {fn}; '
              f'falling back to per-fold NPC mixture')
        return None, None
    cvr2 = image.load_img(fn).get_fdata().ravel()
    cvr2 = cvr2[np.isfinite(cvr2)]
    try:
        fit = fit_r2_mixture(cvr2)
        threshold = r2_p_signal_threshold(fit, p=p_threshold)
        print(f'Whole-brain mixture: signal R² mean {fit["signal_mean_r2"]:.3f}, '
              f'noise R² mean {fit["noise_mean_r2"]:.3f}, '
              f'signal weight {fit["signal_weight"]:.2f}, '
              f'p_signal≥{p_threshold} R² threshold {threshold:.4f} '
              f'(source: {key})')
        return threshold, fit
    except Exception as e:
        print(f'WARNING: whole-brain mixture fit failed: {e!r}')
        return None, None


def _fdr_significant_voxels(train_data, train_pred, p_threshold=0.5,
                             min_voxels=100, brain_threshold=None):
    """Voxels passing ``P(signal | r²) ≥ p_threshold`` under a
    2-Gaussian mixture on logit(R²), via
    :func:`braincoder.utils.stats.fit_r2_mixture` +
    :func:`braincoder.utils.stats.r2_posterior_signal`.

    Two modes:

    * ``brain_threshold`` provided: use that R² threshold directly
      (computed once per subject from a whole-brain mixture — see
      :func:`fit_brain_r2_threshold`). The local NPC mixture is *not*
      fitted; we just compare each voxel's training R² to the
      pre-computed threshold.
    * ``brain_threshold`` is ``None`` (default): fit a per-fold local
      mixture on the NPC training R²s, compute each voxel's
      P(signal | r²), and keep those with ≥ p_threshold (default 0.5
      = "more likely signal than noise"). Falls back to top
      ``min_voxels`` by R² if the fit is degenerate.

    The local fallback default of 100 is calibrated for noisy
    single-trial GLM data on small ROIs where the mixture can be too
    degenerate to find a usable threshold.

    Returns
    -------
    keep : 1-D int array of voxel indices.
    info : dict with mixture summary + actual R² threshold + fallback flag.
    """
    from braincoder.utils.stats import (
        fit_r2_mixture, r2_posterior_signal, r2_p_signal_threshold)

    train_data = np.asarray(train_data, dtype=np.float64)
    train_pred = np.asarray(train_pred, dtype=np.float64)
    ss_res = np.sum((train_data - train_pred) ** 2, axis=0)
    ss_tot = np.sum(
        (train_data - train_data.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2_safe = np.nan_to_num(r2, nan=-np.inf)

    fit = None
    p_signal = None
    if brain_threshold is not None and np.isfinite(brain_threshold):
        threshold = float(brain_threshold)
        source = 'whole_brain'
        keep = np.where(np.isfinite(r2) & (r2 > threshold))[0]
    else:
        threshold = float('inf')
        source = 'npc_local'
        try:
            fit = fit_r2_mixture(r2)
            threshold = r2_p_signal_threshold(fit, p=p_threshold)
            p_signal = r2_posterior_signal(r2, fit)
            keep = np.where(np.isfinite(r2) & (p_signal >= p_threshold))[0]
        except ValueError:
            keep = np.where(np.isfinite(r2) & (r2 > threshold))[0]

    fallback = False
    if len(keep) < min_voxels:
        keep = np.argsort(-r2_safe)[:min_voxels]
        fallback = True

    info = dict(fit) if fit is not None else {}
    info['p_threshold'] = float(p_threshold)
    info['r2_threshold'] = float(threshold)
    info['n_kept'] = int(len(keep))
    info['fallback'] = bool(fallback)
    info['source'] = source
    info['r2'] = r2.astype(np.float32)
    if p_signal is not None:
        info['p_signal'] = p_signal.astype(np.float32)
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
            # FDR info is the same regardless of omega-variant; grab
            # whichever exists (plain is always present).
            d = r['decoding'].get((method, 'plain'),
                                   r['decoding'].get(method, {}))
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
                         sig_voxels, stim_grid, max_resid_iter=2000,
                         distance_matrix=None, use_wwt=True,
                         model_factory=None):
    """Posterior-mean decode test trials using braincoder's Student-t
    residual noise model + ``get_stimulus_pdf``.

    Pipeline:

      1. Build a fresh ``LogGaussianPRF`` on the FDR-significant voxels
         with the method's fitted parameters baked in.
      2. ``init_pseudoWWT(stim_grid, params)`` precomputes the
         basis-weight matrix the residual fitter needs.
      3. ``ResidualFitter`` fits a multivariate Student-t noise model
         on the training residuals. When ``distance_matrix`` is given,
         Ω uses braincoder's distance-modulated form
             Ω = ρ (α exp(-β·D) ττᵀ + (1-α) ττᵀ) + (1-ρ) diag(τ²) + σ² WWᵀ
         which exploits known spatial structure of fMRI noise and is
         far more constrained than a free-form Ω fit from few trials.
      4. ``get_stimulus_pdf`` evaluates the un-normalized posterior over
         the numerosity grid for each test trial.
      5. Readout = posterior mean over ``stim_grid``.

    Returns ``(decoded, omega_stats)`` — ``(None, {})`` if no voxels
    pass FDR. ``omega_stats`` is a dict of the fitted residual-fitter
    parameters (σ², ρ, α, β, dof, mean(τ)), logged so we can introspect
    how much the σ²·WᵀW term was contributing in each fold/method.
    """
    if len(sig_voxels) == 0:
        return None, {}

    from braincoder.models import LogGaussianPRF
    from braincoder.optimize import ResidualFitter

    sig_voxels = np.asarray(sig_voxels, dtype=int)
    sig_params = params.iloc[sig_voxels]
    train_data_sig = train_data.iloc[:, sig_voxels].astype(np.float32)
    test_data_sig = test_data.iloc[:, sig_voxels].astype(np.float32)

    if model_factory is None:
        # Default: 1D-paradigm LogGaussianPRF (single stim_range).
        m = LogGaussianPRF(paradigm=train_par.to_frame(), parameters=sig_params)
    else:
        # Caller-supplied factory — used for multi-range models like
        # neural_priors LinearScalingModel where paradigm is a
        # 2-column DataFrame [x, range].
        m = model_factory(train_par, sig_params)
    m.init_pseudoWWT(stim_grid, sig_params)

    residfit = ResidualFitter(m, train_data_sig, train_par,
                               parameters=sig_params)
    # Subset the precomputed cortical-distance matrix to the selected voxels.
    D_sig = None
    if distance_matrix is not None:
        D_sig = np.asarray(distance_matrix)[np.ix_(sig_voxels, sig_voxels)]
        D_sig = D_sig.astype(np.float32)

    omega, dof = residfit.fit(
        init_sigma2=0.1, method='t', D=D_sig,
        max_n_iterations=max_resid_iter, learning_rate=0.05,
        use_wwt=use_wwt,
        progressbar=False)

    # Snapshot the fitted residual-fitter parameters so we can see
    # post-hoc how much variance the σ²·WᵀW term was absorbing
    # relative to the parametric ρ/α/β components.
    omega_pars = dict(residfit.fitted_omega_parameters or {})

    def _scalar(x):
        x = np.asarray(x)
        return float(x.mean()) if x.size else float('nan')

    omega_stats = {
        'omega_sigma2': _scalar(omega_pars.get('sigma2', np.nan)),
        'omega_rho':    _scalar(omega_pars.get('rho',    np.nan)),
        'omega_alpha':  _scalar(omega_pars.get('alpha',  np.nan)),
        'omega_beta':   _scalar(omega_pars.get('beta',   np.nan)),
        'omega_dof':    _scalar(omega_pars.get('dof',    dof if dof is not None else np.nan)),
        'omega_tau_mean': _scalar(omega_pars.get('tau',  np.nan)),
    }

    pdf = m.get_stimulus_pdf(test_data_sig, stim_grid, sig_params,
                              omega=omega, dof=dof)
    # pdf columns are the stimulus grid values (floats)
    cols = pdf.columns.astype(float).values
    decoded = (pdf.values * cols[None, :]).sum(axis=1) / pdf.values.sum(axis=1)
    return decoded.astype(np.float32), omega_stats


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
                    max_iter, debug, output_dir, smoothed=False,
                    brain_threshold=None, shared_lengthscale=False,
                    prior_params=None, joint_hyperparams=False,
                    use_wwt=True):
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

        # 3. Bayesian fit with GP priors on the requested parameters
        map_pars, hyperpars_dict, sigma = fit_fold_bayes(
            model, train_data, train_par, D, cls_pars,
            max_iter=max_iter, progressbar=False,
            shared_lengthscale=shared_lengthscale,
            prior_params=prior_params,
            joint_hyperparams=joint_hyperparams)
        map_pred = model.predict(
            parameters=map_pars, paradigm=test_par.to_frame())
        map_cvr2 = get_rsq(test_data, map_pred)
        map_train_pred = model.predict(
            parameters=map_pars, paradigm=train_par.to_frame())

        # --- FDR voxel selection + posterior-mean decoding ---
        # We run the decoder twice per (fold, method): once with a free-
        # form Ω, once with the distance-modulated Ω (braincoder's
        # _get_omega_distance — Ω = ρ (α exp(-β D) ττᵀ + (1-α) ττᵀ) +
        # (1-ρ) diag(τ²) + σ² WWᵀ). Saved with an `omega` column so the
        # analysis can compare them head-to-head.
        stim_grid = np.linspace(
            float(paradigm.min()), float(paradigm.max()), 201,
            dtype=np.float32)
        true_test = test_par.values.astype(np.float32)
        decoding = {}                       # keys: (method, omega_variant)
        decode_iter = 200 if debug else 2000
        for method, train_pred_df, fit_pars in (
                ('classical', cls_train_pred, cls_pars),
                ('ml',        ml_train_pred,  ml_pars),
                ('bayes',     map_train_pred, map_pars)):
            sig, fdr_info = _fdr_significant_voxels(
                train_data.values, train_pred_df.values,
                brain_threshold=brain_threshold)
            for omega_variant, D_arg in (('plain', None),
                                          ('distance', D)):
                decoded, omega_stats = _decode_test_trials(
                    fit_pars, train_data, train_par, test_data,
                    sig, stim_grid, max_resid_iter=decode_iter,
                    distance_matrix=D_arg, use_wwt=use_wwt)
                key = (method, omega_variant)
                if decoded is None:
                    decoding[key] = dict(
                        n_sig=0, mae=np.nan, median_ae=np.nan,
                        mae_log=np.nan, median_ae_log=np.nan, r=np.nan,
                        decoded=None, fdr_info=fdr_info,
                        omega_stats=omega_stats)
                    continue
                err = np.abs(decoded - true_test)
                err_log = np.abs(np.log(np.clip(decoded, 1e-6, None))
                                  - np.log(np.clip(true_test, 1e-6, None)))
                if (np.std(decoded) < 1e-9 or np.std(true_test) < 1e-9
                        or len(decoded) < 3):
                    r_fold = float('nan')
                else:
                    r_fold = float(np.corrcoef(decoded, true_test)[0, 1])
                decoding[key] = dict(
                    n_sig=int(len(sig)),
                    mae=float(np.mean(err)),
                    median_ae=float(np.median(err)),
                    mae_log=float(np.mean(err_log)),
                    median_ae_log=float(np.median(err_log)),
                    r=r_fold,
                    decoded=decoded,
                    true=true_test,
                    fdr_info=fdr_info,
                    omega_stats=omega_stats)

        def _summarize(name, cvr2, dec_plain, dec_dist):
            return (f'  {name:9s}: cvR² {float(cvr2.mean()):+.3f} | '
                    f'n_sig {dec_plain["n_sig"]} | '
                    f'plain medAE {dec_plain["median_ae"]:.2f} '
                    f'r {dec_plain["r"]:+.3f} | '
                    f'dist medAE {dec_dist["median_ae"]:.2f} '
                    f'r {dec_dist["r"]:+.3f}')
        print(f'  (classical train R² {cls_train_r2:.3f})')
        print(_summarize('classical', cls_cvr2,
                         decoding[('classical', 'plain')],
                         decoding[('classical', 'distance')]))
        print(_summarize('ml',        ml_cvr2,
                         decoding[('ml', 'plain')],
                         decoding[('ml', 'distance')]))
        print(_summarize('bayes',     map_cvr2,
                         decoding[('bayes', 'plain')],
                         decoding[('bayes', 'distance')]))
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

    # Decoding summary: one row per (fold, method). Natural + log-space
    # errors and within-fold Pearson r are all saved. The analysis
    # notebook should average r across folds rather than pooling trials
    # — pooling confounds within-fold structure (range context, drift)
    # with the decoder's actual decoding power.
    dec_rows = []
    for r in fold_results:
        for key, d in r['decoding'].items():
            method, omega_variant = key
            info = d.get('fdr_info', {}) or {}
            o = d.get('omega_stats', {}) or {}
            dec_rows.append(dict(
                session=r['session'], run2=r['run2'],
                method=method, omega=omega_variant,
                n_sig_voxels=d['n_sig'],
                mae=d['mae'], median_ae=d['median_ae'],
                mae_log=d.get('mae_log', np.nan),
                median_ae_log=d.get('median_ae_log', np.nan),
                r=d.get('r', np.nan),
                stim_range=stim_range,
                fdr_fallback=info.get('fallback', False),
                fdr_r2_threshold=info.get('r2_threshold', np.nan),
                fdr_noise_mean_r2=info.get('noise_mean_r2', np.nan),
                fdr_signal_mean_r2=info.get('signal_mean_r2', np.nan),
                fdr_signal_weight=info.get('signal_weight', np.nan),
                omega_sigma2=o.get('omega_sigma2', np.nan),
                omega_rho=o.get('omega_rho', np.nan),
                omega_alpha=o.get('omega_alpha', np.nan),
                omega_beta=o.get('omega_beta', np.nan),
                omega_dof=o.get('omega_dof', np.nan),
                omega_tau_mean=o.get('omega_tau_mean', np.nan)))
    pd.DataFrame(dec_rows).to_csv(op.join(
        output_dir, f'sub-{subject}_range-{suffix}_desc-decoding.tsv'),
        sep='\t', index=False)

    # Per-trial decoded values for later scatterplots.
    trial_rows = []
    for r in fold_results:
        for key, d in r['decoding'].items():
            method, omega_variant = key
            if d['decoded'] is None:
                continue
            for tr, (dec, tru) in enumerate(zip(d['decoded'], d['true'])):
                trial_rows.append(dict(
                    session=r['session'], run2=r['run2'],
                    method=method, omega=omega_variant, trial=tr,
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


def _git_sha(path):
    """Return short git SHA at ``path``, or ``None`` if not a repo."""
    try:
        out = subprocess.check_output(
            ['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None


def _write_manifest(output_dir, subject, manifest):
    """Write a per-subject manifest JSON capturing the exact run config.

    Includes git SHAs (neural_priors + braincoder), CLI args, voxel-
    selection rule, prior coupling, and timestamps. Future-you (or
    a reviewer) can stand on any TSV and reproduce or audit it.
    """
    fn = op.join(output_dir, f'sub-{subject}_desc-manifest.json')
    with open(fn, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return fn


def _load_data_joint(subject, bids_folder, roi='NPCr', smoothed=False):
    """Load both stim ranges + masked single-trial estimates, indexed by
    (session, run2). Paradigm is a 2-column DataFrame [x, range] that
    LinearScalingModel-family models expect.

    Mirrors the data flow in ``fit_model_cv.py`` exactly so a model 15
    fit here matches the production CV pipeline byte-for-byte.
    """
    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = get_paradigm(sub, fit_responses=False)
    paradigm = paradigm.set_index(
        pd.Index((paradigm.index.get_level_values('run') - 1) % 4 + 1,
                  name='run2'),
        append=True)
    paradigm.index = paradigm.index.swaplevel('run', 'run2')
    paradigm = paradigm.astype(np.float32).droplevel(['run', 'trial_nr',
                                                        'subject'])

    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data_img = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    data = pd.DataFrame(masker.fit_transform(data_img),
                         index=paradigm.index).astype(np.float32)
    data.columns.name = 'voxel'

    xyz = voxel_centroids_mm(masker)
    return paradigm, data, masker, xyz, sub


# Per-model-label config: which params to fix or share when running the
# neural_priors fit_model() pipeline. Kept here (rather than reading from
# fit_model.py) so the model-15 setup is explicit in this file.
M15_FIXED_PARS = ['lower_bound_range', 'delta_wide']
M15_BAYES_FIXED_PARS = M15_FIXED_PARS + ['sd_wide_scale']


def _wire_lsm_transforms(model):
    """Point ``_transform_parameters_forward/backward`` at the model's own
    ``_forward2/_backward2`` if it has them.

    ``LinearScalingModel`` (and similar custom subclasses) build their
    own 7-parameter softplus/identity transforms via
    ``create_transform_functions`` and store them as ``_forward2`` /
    ``_backward2``, but they never assign these to the standard
    ``_transform_parameters_forward`` / ``_backward`` attributes that
    braincoder's ``ParameterFitter`` actually calls. The parent
    AlphaGaussianPRF's 5-parameter ``self.transformations`` would then
    silently strip 2 columns of the 7-parameter init array, producing
    ``IndexError`` when ``shared_pars`` indexes beyond column 4. This
    fix mirrors the pattern in ``braincoder.models.prf_2d`` that does
    the same assignment explicitly.
    """
    if hasattr(model, '_transform_parameters_forward2'):
        model._transform_parameters_forward = model._transform_parameters_forward2
        model._transform_parameters_backward = model._transform_parameters_backward2
    return model


def _run_joint(subject, bids_folder, roi, model_label, D, sub, masker,
                max_iter, debug, output_dir, smoothed=False,
                brain_threshold=None,
                shared_lengthscale=False,
                prior_params=None,
                joint_hyperparams=False,
                use_wwt=True):
    """CV fit of a multi-range model (e.g. model 15) with optional GP prior.

    Uses neural_priors' ``fit_model.fit_model()`` verbatim for the
    classical step, so the encoding fit is byte-for-byte the production
    pipeline. The bayes step then wraps the same model in
    ``BayesianParameterFitter`` with the requested prior(s).

    Decoding runs per held-out range (narrow and wide separately), each
    against a 2D stim grid carrying the range flag.
    """
    paradigm, data, _, _, _ = _load_data_joint(
        subject, bids_folder, roi=roi, smoothed=smoothed)
    n_vx = data.shape[1]
    print(f'\n>>> model {model_label} joint narrow+wide: '
          f'{data.shape[0]} trials × {n_vx} voxels')

    folds = list(paradigm.groupby(level=['session', 'run2']).groups.keys())
    if debug:
        folds = folds[:2]
        print(f'DEBUG: restricting to {len(folds)} folds')

    if prior_params is None:
        prior_params = ['mu_narrow']

    # Validate prior params against the model's actual parameter list
    probe_model = _wire_lsm_transforms(get_model(model_label))
    invalid = [p for p in prior_params if p not in probe_model.parameter_labels]
    if invalid:
        raise ValueError(
            f"prior_params {invalid} not in model_{model_label}'s "
            f"parameter_labels {probe_model.parameter_labels}")
    print(f'Prior on: {prior_params}; '
          f'model parameters = {probe_model.parameter_labels}')

    def _decoder_factory(train_par_, params_):
        m = _wire_lsm_transforms(get_model(model_label))
        m.paradigm = m.get_paradigm(train_par_)
        m.parameters = params_
        return m

    fold_results = []
    for fold in folds:
        sess, run2 = fold
        print(f'\n=== Fold (session={sess}, run2={run2}) ===')

        test_data = data.loc[fold].copy().astype(np.float32)
        test_par = paradigm.loc[fold].copy().astype(np.float32)
        train_data = data.drop(fold).copy()
        train_par = paradigm.drop(fold).copy()

        # ---- 1. Classical via fit_model() (grid + iterative, with the
        # right fixed_pars and shared_pars for this model_label).
        model = _wire_lsm_transforms(get_model(model_label))
        cls_pars = fit_model_npr(model_label, model, train_data, train_par,
                                  max_n_iterations=max_iter)
        cls_pred = model.predict(parameters=cls_pars, paradigm=train_par)
        cls_train_r2 = float(get_rsq(train_data, cls_pred).mean())
        cls_pred_test = model.predict(parameters=cls_pars, paradigm=test_par)
        cls_cvr2 = get_rsq(test_data, cls_pred_test)
        cls_train_pred = cls_pred

        # ---- 2. ML: BayesianParameterFitter with no priors. Init from
        # the classical pars. lower_bound_range + delta_wide held fixed
        # (their grid value); other params get refit per voxel.
        ml_fitter = BayesianParameterFitter(model, train_data, train_par,
                                              priors={})
        ml_fitter.classical_estimates = cls_pars
        ml_fitter.fit_map(max_n_iterations=max_iter, init_pars=cls_pars,
                           fixed_pars=M15_FIXED_PARS, progressbar=False)
        ml_pars = ml_fitter.map_estimates
        ml_sigma = ml_fitter.map_sigma
        ml_pred_test = model.predict(parameters=ml_pars, paradigm=test_par)
        ml_cvr2 = get_rsq(test_data, ml_pred_test)
        ml_train_pred = model.predict(parameters=ml_pars, paradigm=train_par)

        # ---- 3. Bayes: GP prior(s) on requested param(s). Also hold
        # sd_wide_scale fixed at the classical-shared value (it was fit
        # as one value across all voxels in stage 1).
        priors = _build_priors(D, cls_pars, prior_params)
        bayes_fitter = BayesianParameterFitter(model, train_data, train_par,
                                                 priors=priors)
        bayes_fitter.classical_estimates = cls_pars
        if joint_hyperparams:
            if shared_lengthscale:
                bayes_fitter.tie_lengthscales()
        else:
            bayes_fitter.fit_hyperparameters(
                progressbar=False, shared_lengthscale=shared_lengthscale)
        bayes_fitter.fit_map(max_n_iterations=max_iter, init_pars=cls_pars,
                              fixed_pars=M15_BAYES_FIXED_PARS,
                              joint_hyperparams=joint_hyperparams,
                              progressbar=False)
        map_pars = bayes_fitter.map_estimates
        sigma = bayes_fitter.map_sigma
        hyperpars_dict = {name: priors[name].hyperparameters
                          for name in prior_params}
        map_pred_test = model.predict(parameters=map_pars, paradigm=test_par)
        map_cvr2 = get_rsq(test_data, map_pred_test)
        map_train_pred = model.predict(parameters=map_pars, paradigm=train_par)

        # ---- 4. FDR voxel selection (train R²) + per-range decoding
        decoding = {}
        decode_iter = 200 if debug else 2000
        for method, train_pred_df, fit_pars in (
                ('classical', cls_train_pred, cls_pars),
                ('ml',        ml_train_pred,  ml_pars),
                ('bayes',     map_train_pred, map_pars)):
            sig, fdr_info = _fdr_significant_voxels(
                train_data.values, train_pred_df.values,
                brain_threshold=brain_threshold)

            for tr_range, range_label in ((0.0, 'narrow'), (1.0, 'wide')):
                test_mask = (test_par['range'] == tr_range).values
                if not test_mask.any():
                    continue
                test_data_r = test_data.loc[test_mask]
                test_par_r = test_par.loc[test_mask]
                true_test = test_par_r['x'].values.astype(np.float32)

                # 2D stim grid: numerosity range from train data of this
                # range, with the range flag set.
                train_mask = (train_par['range'] == tr_range).values
                if not train_mask.any():
                    continue
                stim_lo = float(train_par.loc[train_mask, 'x'].min())
                stim_hi = float(train_par.loc[train_mask, 'x'].max())
                stim_xs = np.linspace(stim_lo, stim_hi, 201, dtype=np.float32)
                stim_grid = np.column_stack([
                    stim_xs, np.full_like(stim_xs, tr_range)]).astype(np.float32)

                for omega_variant, D_arg in (('plain', None),
                                              ('distance', D)):
                    decoded, omega_stats = _decode_test_trials(
                        fit_pars, train_data, train_par, test_data_r,
                        sig, stim_grid, max_resid_iter=decode_iter,
                        distance_matrix=D_arg, use_wwt=use_wwt,
                        model_factory=_decoder_factory)
                    key = (method, omega_variant, range_label)
                    if decoded is None:
                        decoding[key] = dict(
                            n_sig=0, mae=np.nan, median_ae=np.nan,
                            mae_log=np.nan, median_ae_log=np.nan, r=np.nan,
                            decoded=None, fdr_info=fdr_info,
                            omega_stats=omega_stats)
                        continue
                    err = np.abs(decoded - true_test)
                    err_log = np.abs(np.log(np.clip(decoded, 1e-6, None))
                                      - np.log(np.clip(true_test, 1e-6, None)))
                    if (np.std(decoded) < 1e-9 or np.std(true_test) < 1e-9
                            or len(decoded) < 3):
                        r_fold = float('nan')
                    else:
                        r_fold = float(np.corrcoef(decoded, true_test)[0, 1])
                    decoding[key] = dict(
                        n_sig=int(len(sig)),
                        mae=float(np.mean(err)),
                        median_ae=float(np.median(err)),
                        mae_log=float(np.mean(err_log)),
                        median_ae_log=float(np.median(err_log)),
                        r=r_fold,
                        decoded=decoded,
                        true=true_test,
                        fdr_info=fdr_info,
                        omega_stats=omega_stats)

        print(f'  (classical train R² {cls_train_r2:+.3f})')
        for name, cvr2 in (('classical', cls_cvr2), ('ml', ml_cvr2),
                            ('bayes', map_cvr2)):
            summary = []
            for rng in ('narrow', 'wide'):
                for om in ('plain', 'distance'):
                    d = decoding.get((name, om, rng), {})
                    r = d.get('r', float('nan'))
                    summary.append(f'{rng[:1]}/{om[:1]} r {r:+.3f}')
            print(f'  {name:9s}: cvR² {float(cvr2.mean()):+.3f} | ' +
                  ' | '.join(summary))
        for name, hp in hyperpars_dict.items():
            print(f'    prior[{name}]: l={hp["lengthscale"]:.2f} mm, '
                  f'v={hp["variance"]:.3f}, nug={hp["nugget"]:.3f}')

        fold_results.append({
            'session': sess, 'run2': run2,
            'classical_params': cls_pars, 'ml_params': ml_pars,
            'bayes_params': map_pars,
            'classical_cvr2': cls_cvr2, 'ml_cvr2': ml_cvr2,
            'bayes_cvr2': map_cvr2,
            'hyperparameters': hyperpars_dict,
            'ml_sigma': ml_sigma, 'bayes_sigma': sigma,
            'decoding': decoding,
        })

    # ---- Write outputs ----
    suffix = f'm{model_label}'
    with open(op.join(output_dir,
                       f'sub-{subject}_model-{suffix}_desc-folds.pkl'),
              'wb') as f:
        pickle.dump({
            'subject': subject, 'roi': roi, 'model_label': model_label,
            'folds': fold_results,
        }, f)

    # cvR² long-form per voxel × fold × method
    rows = []
    for r in fold_results:
        for method, cvr2 in (('classical', r['classical_cvr2']),
                              ('ml',        r['ml_cvr2']),
                              ('bayes',     r['bayes_cvr2'])):
            for vox, val in cvr2.items():
                rows.append(dict(
                    session=r['session'], run2=r['run2'],
                    voxel=int(vox), method=method,
                    model_label=int(model_label), cvr2=float(val)))
    cvr2_long = pd.DataFrame(rows)
    cvr2_long.to_csv(op.join(
        output_dir, f'sub-{subject}_model-{suffix}_desc-cvr2.tsv'),
        sep='\t', index=False)

    # Hyperparams
    hp_rows = []
    for r in fold_results:
        for pname, hp in r['hyperparameters'].items():
            hp_rows.append(dict(session=r['session'], run2=r['run2'],
                                 parameter=pname, **hp))
    if hp_rows:
        pd.DataFrame(hp_rows).to_csv(op.join(
            output_dir, f'sub-{subject}_model-{suffix}_desc-hyperpars.tsv'),
            sep='\t', index=False)

    # Decoding: one row per (fold, method, omega, range). Has the same
    # column schema as the LogGaussianPRF path plus a 'stim_range' column.
    dec_rows = []
    for r in fold_results:
        for key, d in r['decoding'].items():
            method, omega_variant, range_label = key
            info = d.get('fdr_info', {}) or {}
            o = d.get('omega_stats', {}) or {}
            dec_rows.append(dict(
                session=r['session'], run2=r['run2'],
                method=method, omega=omega_variant,
                n_sig_voxels=d['n_sig'],
                mae=d['mae'], median_ae=d['median_ae'],
                mae_log=d.get('mae_log', np.nan),
                median_ae_log=d.get('median_ae_log', np.nan),
                r=d.get('r', np.nan),
                stim_range=range_label,
                model_label=int(model_label),
                fdr_fallback=info.get('fallback', False),
                fdr_r2_threshold=info.get('r2_threshold', np.nan),
                fdr_noise_mean_r2=info.get('noise_mean_r2', np.nan),
                fdr_signal_mean_r2=info.get('signal_mean_r2', np.nan),
                fdr_signal_weight=info.get('signal_weight', np.nan),
                omega_sigma2=o.get('omega_sigma2', np.nan),
                omega_rho=o.get('omega_rho', np.nan),
                omega_alpha=o.get('omega_alpha', np.nan),
                omega_beta=o.get('omega_beta', np.nan),
                omega_dof=o.get('omega_dof', np.nan),
                omega_tau_mean=o.get('omega_tau_mean', np.nan)))
    pd.DataFrame(dec_rows).to_csv(op.join(
        output_dir, f'sub-{subject}_model-{suffix}_desc-decoding.tsv'),
        sep='\t', index=False)

    return cvr2_long


def main(subject, bids_folder, roi='NPCr', stim_range='both',
         smoothed=False, max_iter=2000, debug=False, output_dir=None,
         wb_model_label=15, use_brain_threshold=False,
         tag='default', shared_lengthscale=False,
         prior_params=None, joint_hyperparams=False,
         use_wwt=True, model_label=None):
    if debug:
        max_iter = 200
    if prior_params is None:
        # Model 15 (and other LinearScalingModel variants) has 'mu_narrow'
        # rather than 'mu'. Default to a sensible single-parameter prior
        # in either case.
        if model_label is None:
            prior_params = list(DEFAULT_PRIOR_PARAMS)
        else:
            prior_params = ['mu_narrow']
    if model_label is None:
        invalid = [p for p in prior_params if p not in ALLOWED_PRIOR_PARAMS]
        if invalid:
            raise ValueError(
                f"prior_params {invalid} not in allowed list "
                f"{ALLOWED_PRIOR_PARAMS}")
    # If model_label is set, _run_joint validates against the actual
    # model.parameter_labels after instantiation.

    run_started = datetime.datetime.utcnow().isoformat() + 'Z'
    import braincoder as _bc
    manifest = {
        'subject':           subject,
        'roi':               roi,
        'stim_range':        stim_range,
        'smoothed':          bool(smoothed),
        'tag':               tag,
        'shared_lengthscale': bool(shared_lengthscale),
        'joint_hyperparams': bool(joint_hyperparams),
        'use_wwt':            bool(use_wwt),
        'model_label':       (int(model_label) if model_label is not None
                              else None),
        'use_brain_threshold': bool(use_brain_threshold),
        'wb_model_label':    int(wb_model_label),
        'max_iter':          int(max_iter),
        'debug':             bool(debug),
        'prior_params':      list(prior_params),
        'voxel_selection':   ('whole_brain_p>=0.5' if use_brain_threshold
                              else 'per_fold_p_signal>=0.5'),
        'git_neural_priors': _git_sha(op.dirname(op.abspath(__file__))),
        'git_braincoder':    _git_sha(op.dirname(_bc.__file__)),
        'run_started':       run_started,
    }

    # FDR voxel-selection threshold.
    #
    # Default (``use_brain_threshold=False``): per-fold NPC-local R²
    # mixture inside ``_fdr_significant_voxels`` — strictly within-fold
    # cross-validation, no subject-level information leak from outside
    # the training set.
    #
    # ``--brain_threshold`` flag: also fit a whole-brain R² mixture on
    # the existing cross-validated model{wb_model_label} cvR² map and
    # apply that single threshold to every fold's training R². Cleaner
    # bimodal mixture (lots of obvious-noise voxels in white matter /
    # ventricles), but the threshold is derived from data outside the
    # training set of any one fold, which is a subject-level
    # information leak.
    if use_brain_threshold:
        brain_threshold, brain_fit = fit_brain_r2_threshold(
            subject, bids_folder, p_threshold=0.5,
            wb_model_label=wb_model_label)
    else:
        brain_threshold, brain_fit = None, None
        print('Voxel-selection: per-fold NPC R² mixture '
              '(p_signal≥0.5; --brain_threshold to use whole-brain instead)')

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
        # Experiment tag is its own directory level so the same
        # (roi, smoothing) combination can host multiple side-by-side
        # experiments. Glob-friendly: `exp-*` lists all experiments.
        output_dir = op.join(
            bids_folder, 'derivatives', 'encoding_models',
            key, f'exp-{tag}', f'sub-{subject}', 'func')
    os.makedirs(output_dir, exist_ok=True)
    manifest['output_dir'] = output_dir
    _write_manifest(output_dir, subject, manifest)
    np.save(op.join(output_dir, f'sub-{subject}_desc-distance.npy'), D)
    np.save(op.join(output_dir, f'sub-{subject}_desc-vertex_idx.npy'),
             vtx_idx)

    # Save the whole-brain mixture summary so it's part of the subject's
    # output (one record per subject).
    if brain_fit is not None:
        wb_record = dict(brain_fit)
        wb_record['p_threshold'] = 0.5
        wb_record['r2_threshold'] = float(brain_threshold)
        wb_record['source'] = f'model{wb_model_label}.cv.whole_brain.smoothed'
        pd.DataFrame([wb_record]).to_csv(
            op.join(output_dir,
                     f'sub-{subject}_desc-wholebrain_r2_mixture.tsv'),
            sep='\t', index=False)

    if model_label is not None:
        # Multi-range neural_priors model (e.g., LinearScalingModel
        # model 15). Fits narrow + wide jointly in one CV pass.
        all_cvr2 = [_run_joint(
            subject, bids_folder, roi, model_label, D, sub, masker,
            max_iter, debug, output_dir, smoothed=smoothed,
            brain_threshold=brain_threshold,
            shared_lengthscale=shared_lengthscale,
            prior_params=prior_params,
            joint_hyperparams=joint_hyperparams,
            use_wwt=use_wwt)]
        ranges = ['joint']
    else:
        ranges = ['narrow', 'wide'] if stim_range == 'both' else [stim_range]
        all_cvr2 = []
        for r in ranges:
            all_cvr2.append(_run_one_range(
                subject, bids_folder, roi, r, D, sub, masker,
                max_iter, debug, output_dir, smoothed=smoothed,
                brain_threshold=brain_threshold,
                shared_lengthscale=shared_lengthscale,
                prior_params=prior_params,
                joint_hyperparams=joint_hyperparams,
                use_wwt=use_wwt))

    pd.concat(all_cvr2, ignore_index=True).to_csv(
        op.join(output_dir, f'sub-{subject}_desc-cvr2_all.tsv'),
        sep='\t', index=False)

    # Update manifest with finish time + list of TSVs that landed.
    manifest['run_finished'] = datetime.datetime.utcnow().isoformat() + 'Z'
    manifest['ranges_completed'] = ranges
    _write_manifest(output_dir, subject, manifest)
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
    parser.add_argument('--wb_model_label', type=int, default=15,
                        help='Which neural_priors whole-brain model to '
                             'source cvR² from for the FDR mixture '
                             '(default 15, LinearScalingModel). Only used '
                             'if --brain_threshold is passed.')
    parser.add_argument('--brain_threshold', dest='use_brain_threshold',
                        action='store_true',
                        help='Use subject-level whole-brain R² mixture to '
                             'set the FDR voxel-selection threshold. Default '
                             'is per-fold NPC-local mixture (strict within-'
                             'fold CV; recommended).')
    parser.add_argument('--tag', default='default',
                        help='Experiment tag. Outputs land under '
                             'gp_prior_roi-{ROI}[.smoothed]/exp-{tag}/. '
                             'Each subject gets a _desc-manifest.json '
                             'recording CLI args, git SHAs, and run '
                             'timestamps. Use distinct tags for variants '
                             'you want to compare side-by-side.')
    parser.add_argument('--shared_lengthscale', action='store_true',
                        help='Tie GP-prior lengthscales across all four '
                             'parameters (mu, sd, amplitude, baseline) to '
                             'one shared value via joint MLE. Use when you '
                             'want a single "cortical topographic scale" '
                             'rather than per-parameter scales.')
    parser.add_argument('--prior_params', nargs='+',
                        default=None,
                        metavar='PARAM',
                        help='Which PRF parameters get a GP prior. Default '
                             'is all four (mu, sd, amplitude, baseline). '
                             'Paper-faithful single-parameter recipe: '
                             '--prior_params mu. Empty list (passable via '
                             "'--prior_params' with no args) gives the "
                             'no-prior ML mode.')
    parser.add_argument('--joint_hyperparams', action='store_true',
                        help='Type-II MAP: skip stage 2 (no MLE on noisy '
                             'classical estimates) and co-optimize the GP '
                             'hyperparameters with the model parameters in '
                             'stage 3. The prior\'s -½ log|K(ψ)| term '
                             'automatically penalizes over-smoothing of ψ.')
    parser.add_argument('--model_label', type=int, default=None,
                        help='Use a neural_priors fit_model.py model '
                             '(e.g. 15 = LinearScalingModel narrow+wide '
                             'with shared sd_wide_scale and fixed '
                             'lower_bound_range / delta_wide). When set, '
                             'narrow and wide trials are fit jointly in '
                             'one CV pass instead of separately. Default '
                             'prior_params becomes [mu_narrow]. The '
                             '--range argument is ignored.')
    parser.add_argument('--no_wwt', dest='use_wwt', action='store_false',
                        help='Strip the σ²·WᵀW term from the decoder Ω. '
                             'WᵀW is a voxel-voxel tuning-similarity '
                             'matrix that gets injected into the noise '
                             'covariance, potentially absorbing whatever '
                             'spatial-smoothness gain the GP prior brings. '
                             'Setting this lets us test whether WᵀW is '
                             'doing real spatial-noise modeling or just '
                             'creating cancellation.')
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    main(args.subject, args.bids_folder,
         roi=args.roi, stim_range=args.stim_range,
         smoothed=args.smoothed,
         wb_model_label=args.wb_model_label,
         use_brain_threshold=args.use_brain_threshold,
         tag=args.tag,
         shared_lengthscale=args.shared_lengthscale,
         prior_params=args.prior_params,
         joint_hyperparams=args.joint_hyperparams,
         use_wwt=args.use_wwt,
         model_label=args.model_label,
         max_iter=args.max_iter,
         debug=args.debug, output_dir=args.output_dir)
