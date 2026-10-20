"""
Bayesian decoding of numerosity from single-trial fMRI amplitudes.

Overview
--------
For each subject, this script implements leave-one-run-out cross-validation
(folded by the run2 index, which groups original runs 1+5, 2+6, 3+7, 4+8).
In each fold:

  1. Fit encoding model on the training set (grid search + gradient descent).
  2. Select the best n_voxels by training-set R² (or by cross-validated R²
     when n_voxels=0).
  3. Fit a multivariate Student-t residual noise model (ResidualFitter) on the
     training set.  This yields a noise covariance matrix omega and degrees of
     freedom dof.
  4. For each test trial, evaluate the likelihood P(data | n, range) over the
     full grid of possible stimuli (all numerosities × {narrow, wide}) using
     the fitted encoding model and noise model.  This unnormalised likelihood
     is the posterior PDF over stimulus identity (assuming a flat prior).

Output
------
One TSV per subject written to
  derivatives/decoding2/model{N}.smoothed[.flags]/sub-{id}/func/
  sub-{id}_mask-{roi}_nvoxels-{n}_pars.tsv

Rows: one per test trial across all folds.
Row index (5 levels): session, run, trial_nr, x (true numerosity), range
  (0.0=narrow, 1.0=wide).  Sorted to original presentation order in the
  analysis notebook via sort_index(level=['session','run','trial_nr']).
Columns: MultiIndex (n, range_condition) over the decoded stimulus grid,
  where the first header row is the numerosity value and the second is
  '0.0' (narrow) or '1.0' (wide).

Key flags
---------
--n_voxels 0        Select voxels by cross-validated R² > 0 (all positive).
--n_voxels N        Select the top N voxels by training-set R².
--spherical_noise   Fit isotropic (scalar) noise rather than full covariance.
--separate_sigmas   Fit separate noise models for narrow and wide conditions.
--fit_responses     Use the response-fitted encoding model (fitted to the
                    participant's estimates) instead
                    of the true presented numerosity.

Opt-in noise-model study (``--noise_model lowrank_cv``; methods study, does
not change any default or any production output)
-------------------------------------------------------------------------------
Replaces the ResidualFitter noise model by braincoder's ``LowRankNoise``
(branch ``noise-models``): Omega = D (F F' + Psi) D with rank k and a
likelihood temperature T (posterior ~ likelihood^(1/T)) chosen jointly by
nested leave-one-run-out CV *inside the training runs* on the held-out log
posterior of the true stimulus. Everything else (encoding fit, cvr2 voxel
selection read from the production cache, folds, grid) is the production
path, and the production ResidualFitter noise model is fitted on the *same*
fold parameters as a paired reference. Requires ``--tag``; writes to
derivatives/decoding2/<production key>.<tag>/ (never the production tree):

  *_desc-lowrank_loglik.tsv   untempered LowRankNoise log-likelihood grid
  *_desc-refnoise_loglik.tsv  ResidualFitter (production settings) log-lik.
  *_pars.tsv                  tempered LowRankNoise posterior (sums to 1)
  *_desc-noisefit_log.tsv     per fold: k, T, dof, n_voxels, timings
  *_desc-cvscores.tsv         inner-CV score for every (k, T)
  *_desc-gdpars.tsv           per-fold encoding parameters

The production env (neural_priors2: py3.9, keras 2, braincoder ed48b10) cannot
import the keras-3 braincoder of the noise-models branch, and the model classes
in models.py depend on the old braincoder internals. ``LowRankNoise`` is pure
numpy/scipy, so its module is loaded by file path from a separate checkout
($BRAINCODER_NOISE, default ~/git/braincoder-noise); the production braincoder
is untouched.
"""

import argparse
from neural_priors.utils.data import Subject
from fit_model import get_paradigm, get_model, fit_model, fit_model_cv
from pathlib import Path
import numpy as np
from braincoder.utils import get_rsq
import pandas as pd
import os
import os.path as op
from braincoder.optimize import ParameterFitter
from braincoder.models import AlphaGaussianPRF
from neural_priors.encoding_model.models import AlphaDeltaModel
from braincoder.optimize import ResidualFitter
import pingouin as pg
import importlib.util
import time


def load_lowrank_noise():
    """Return braincoder.noise.LowRankNoise from the noise-models checkout.

    Tries a regular import first (an env whose braincoder already has it),
    otherwise loads braincoder/noise.py by path from $BRAINCODER_NOISE
    (default ~/git/braincoder-noise). noise.py only needs numpy/scipy/pandas
    (its keras import is local to log_prob_backend, which is not used here).
    """
    try:
        from braincoder.noise import LowRankNoise
        return LowRankNoise
    except ImportError:
        pass
    root = os.environ.get('BRAINCODER_NOISE', op.expanduser('~/git/braincoder-noise'))
    path = op.join(root, 'braincoder', 'noise.py')
    spec = importlib.util.spec_from_file_location('braincoder_noise_standalone', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f'LowRankNoise loaded from {path}')
    return mod.LowRankNoise


def _grid_frame(grid, like):
    return pd.DataFrame(np.asarray(grid, dtype=np.float32), columns=like.columns)


def add_grid_predictor(model, paradigm_like):
    """Give an old-braincoder model the predict_stimulus_grid() LowRankNoise uses.

    For these (non-HRF) models predicting the grid as a paradigm is the same
    computation as get_stimulus_pdf's per-grid-point prediction.
    """
    if hasattr(model, 'predict_stimulus_grid'):
        return model

    def predict_stimulus_grid(grid, parameters=None, weights=None):
        pred = model.predict(paradigm=_grid_frame(grid, paradigm_like), parameters=parameters)
        return np.asarray(pred.values, dtype=np.float64)

    model.predict_stimulus_grid = predict_stimulus_grid
    return model


def mvt_grid_loglik(test_data, grid_pred, omega, dof):
    """log p(y_t | s) for every test trial t and grid stimulus s (numpy, float64).

    Multivariate Student-t with scale matrix omega (dof None/inf = Gaussian):
    the residual distribution braincoder's get_stimulus_pdf uses, but kept in
    log space (the production pdf, exp(ll - max), underflows to exact zeros).
    """
    Y = np.asarray(test_data, dtype=np.float64)
    P = np.asarray(grid_pred, dtype=np.float64)
    L = np.linalg.cholesky(np.asarray(omega, dtype=np.float64))
    V = Y.shape[1]
    logdet = 2 * np.log(np.diag(L)).sum()
    from scipy.linalg import solve_triangular
    Zy = solve_triangular(L, Y.T, lower=True)          # V x n
    Zp = solve_triangular(L, P.T, lower=True)          # V x S
    d2 = (Zy ** 2).sum(0)[:, None] + (Zp ** 2).sum(0)[None, :] - 2 * Zy.T @ Zp
    d2 = np.clip(d2, 0, None)
    from scipy.special import gammaln
    if dof is None or not np.isfinite(dof):
        return -0.5 * (V * np.log(2 * np.pi) + logdet + d2)
    dof = float(dof)
    return (gammaln((dof + V) / 2.) - gammaln(dof / 2.) - 0.5 * V * np.log(dof * np.pi)
            - 0.5 * logdet - 0.5 * (dof + V) * np.log1p(d2 / dof))


def tempered_posterior(ll, temperature):
    """Normalised (sum = 1) posterior over the grid from a log-lik table."""
    ll = np.asarray(ll, dtype=np.float64)
    if np.isinf(temperature):
        return np.full_like(ll, 1.0 / ll.shape[1])
    a = ll / temperature
    a = a - a.max(1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(1, keepdims=True)

def get_decoding_paradigm(sub, fit_responses=False, drop_levels=True):
    """
    Return the stimulus paradigm DataFrame with a 'run2' cross-validation index.

    Why run2 exists — the folding logic
    ------------------------------------
    Each session has 8 runs.  Each run is entirely narrow OR entirely wide
    (determined by the stimulus range).  Crucially, the design is arranged so
    that run k and run k+4 are *opposite* conditions within the same session:
    if run 1 is narrow, run 5 is wide; if run 2 is wide, run 6 is narrow; etc.

    We want each cross-validation fold to contain *both* narrow and wide trials,
    for two reasons:
      - The residual noise model must see both conditions during training.
      - The test set should contain both so we can evaluate decoding for each.

    The solution: group runs into 4 folds by pairing each run with its
    complement 4 runs later:

        run2 = (run - 1) % 4 + 1

        original run  →  run2 (fold)
        ─────────────────────────────
        1             →  1   (narrow)  ┐ fold 1 test set:
        5             →  1   (wide)    ┘ one narrow + one wide run
        2             →  2   (wide)    ┐ fold 2 test set:
        6             →  2   (narrow)  ┘ one narrow + one wide run
        3             →  3   (narrow)  ┐ fold 3 test set:
        7             →  3   (wide)    ┘ ...
        4             →  4   (wide)    ┐ fold 4 test set:
        8             →  4   (narrow)  ┘ ...

    The cross-validation loop iterates over (session, run2) pairs.  For each
    fold, the test set is data.loc[(session, run2)], which selects both original
    runs (e.g. runs 1 and 5 for fold 1), giving a balanced narrow+wide test set.
    Everything else is the training set.

    Parameters
    ----------
    drop_levels : bool
        True (default): drop run, trial_nr, subject from the index, leaving
        only [session, run2].  Used for the data DataFrame during fitting.
        False: keep all levels [subject, session, run2, trial_nr, run].  Used
        to tag saved PDF rows with their original presentation identity so the
        output file can be sorted back to (session, run, trial_nr) order.
    """
        # Get paradigm/data/model
    paradigm = get_paradigm(sub, fit_responses=fit_responses)
    paradigm = paradigm.set_index(pd.Index((paradigm.index.get_level_values('run') - 1) % 4 + 1, name='run2'), append=True)
    paradigm.index = paradigm.index.swaplevel('run', 'run2')
    paradigm = paradigm.astype(np.float32)
    
    if drop_levels:
        paradigm = paradigm.droplevel(['run', 'trial_nr', 'subject'])
    
    return paradigm

def main(subject, model_label=3, roi='NPCr', bids_folder='/data/ds-neural_priors', smoothed=True, debug=False, fit_responses=False,
         n_voxels=100, spherical_noise=False, separate_sigmas=False, subtract_wt_baseline=False, lambd=0.0,
         noise_model='residualfitter', tag='', lowrank_k='cv', inner_params='fixed', inner_groups='run',
         k_grid=None):
    """
    Run leave-one-run-out Bayesian decoding for one subject.

    Cross-validation folds are defined by run2 (see get_decoding_paradigm).
    For each fold the encoding model is re-fit from scratch on the training
    runs, a Student-t noise model is estimated, and P(data | stimulus) is
    evaluated over the full (numerosity x range) grid for every test trial.

    Parameters
    ----------
    subject : str
        Zero-padded subject ID, e.g. '01'.
    model_label : int
        Encoding model variant.  Only 15, 18, and 31 support decoding.
    roi : str
        ROI label passed to Subject.get_volume_mask() (e.g. 'NPCr').
    bids_folder : str
        Root of the BIDS dataset.
    smoothed : bool
        Use spatially smoothed single-trial estimates.
    debug : bool
        Cap iterations at 100 for quick testing.
    fit_responses : bool
        Decode behavioural responses rather than true numerosities.
    n_voxels : int
        Number of voxels to use (ranked by training R²).
        0 = use all voxels with cross-validated R² > 0.
    spherical_noise : bool
        Constrain the noise covariance to be isotropic (scalar sigma²·I).
    separate_sigmas : bool
        Fit independent noise models for narrow and wide conditions.
    """
    assert model_label in [15, 18, 31], 'Only models 15, 18, and 31 are supported for decoding'

    if noise_model != 'residualfitter':
        if not tag:
            raise ValueError('--noise_model other than residualfitter needs a --tag '
                             '(never write into the production derivatives tree)')
        if separate_sigmas:
            raise NotImplementedError('lowrank_cv is implemented for the joint (not separate_sigmas) path')
        LowRankNoise = load_lowrank_noise()


    sub = Subject(subject_id=subject, bids_folder=bids_folder)
    bids_folder = Path(bids_folder)

    max_n_iterations = 100 if debug else 2000

    # Create target folder
    key = f'model{model_label}'

    if smoothed:
        key += '.smoothed'

    if spherical_noise:
        key += '.spherical_noise'

    if fit_responses:
        key += '.fit_responses'

    if separate_sigmas:
        key += '.separate_sigmas'

    if subtract_wt_baseline:
        key += '.subtract_wt_baseline'

    if lambd > 0.0:
        key += f'.lambd-{lambd}'

    if tag:
        key += f'.{tag}'

    target_dir = bids_folder / 'derivatives' / 'decoding2' / key / f'sub-{subject}' / 'func'

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    # Canonical directory for cached per-fold cvr2 results.
    # Only model_label, smoothed, and fit_responses affect the encoding model
    # fit and therefore the cvr2 — noise settings (lambd, spherical_noise, etc.)
    # do not.  We cache here so different decoding runs can reuse the same cvr2.
    cvr2_key = f'model{model_label}'
    if smoothed:
        cvr2_key += '.smoothed'
    if fit_responses:
        cvr2_key += '.fit_responses'
    cvr2_cache_dir = bids_folder / 'derivatives' / 'decoding2' / cvr2_key / f'sub-{subject}' / 'func'

    # Get paradigm/data/model
    paradigm = get_decoding_paradigm(sub, fit_responses=fit_responses)
    # Full paradigm keeps run/trial_nr so we can tag each PDF row with its original identity
    paradigm_full = get_decoding_paradigm(sub, fit_responses=fit_responses, drop_levels=False)

    data = sub.get_single_trial_estimates(session=None, smoothed=smoothed)
    masker = sub.get_volume_mask(roi=roi, epi_space=True, return_masker=True)
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index).astype(np.float32)

    # all_cvr2 = []

    # Build the stimulus grid: every (numerosity, range_condition) combination.
    # Result shape: (2 * n_unique_stimuli, 2), alternating narrow/wide per numerosity:
    #   [(n1, 0), (n1, 1), (n2, 0), (n2, 1), ...]
    # This is passed to init_pseudoWWT and get_stimulus_pdf so the decoder
    # evaluates P(data | n, range) jointly over both conditions.
    stimulus_range = np.sort(paradigm['x'].unique())
    stimulus_range = np.stack([np.repeat(stimulus_range, 2), np.stack(np.tile([0, 1], len(stimulus_range)), axis=0)], axis=1)
    stimulus_mi = pd.MultiIndex.from_arrays(stimulus_range.T, names=['n', 'range'])

    pdfs = []
    lowrank_lls, ref_lls, noise_log, cv_tables, gd_tables = [], [], [], [], []

    # Each fold holds out one run2 group (= 2 original runs, one per session)
    # as the test set; all remaining runs form the training set.
    for (test_session, test_run), _ in paradigm.groupby(level=['session', 'run2']):

        print(f'Fitting using session {test_session} run {test_run} as test set')

        test_data, test_paradigm = data.loc[(test_session, test_run)].copy().astype(np.float32), paradigm.loc[(test_session, test_run)].copy().astype(np.float32)
        train_data, train_paradigm = data.drop((test_session, test_run)).copy(), paradigm.drop((test_session, test_run)).copy()

        # Get original run/trial_nr for this fold so we can save a sortable index
        fold_mask = ((paradigm_full.index.get_level_values('session') == test_session) &
                     (paradigm_full.index.get_level_values('run2') == test_run))
        test_paradigm_full = paradigm_full[fold_mask]

        print(test_paradigm)

        # Get model
        model = get_model(model_label)

        # Cross-validate to get number of (and which) voxels
        if n_voxels == 0:

            cvr2_fn = cvr2_cache_dir / f'sub-{subject}_ses-{test_session}_run2-{test_run}_mask-{roi}_desc-cvr2_pars.tsv'

            if cvr2_fn.exists():
                print(f'Loading cached cvr2 from {cvr2_fn}')
                cvr2 = pd.read_csv(cvr2_fn, sep='\t', index_col=0).squeeze('columns')
            else:
                print('Cross-validating to get number of voxels')
                cvr2 = fit_model_cv(train_data, train_paradigm, model_label, max_n_iterations=max_n_iterations)
                if tag:
                    # tagged (study) runs never add files to the production tree
                    cvr2_fn = target_dir / cvr2_fn.name
                else:
                    cvr2_cache_dir.mkdir(parents=True, exist_ok=True)
                cvr2.to_csv(cvr2_fn, sep='\t')

            print(cvr2)
            r2_mask = cvr2 > 0.0

            print(f'Selecting {np.sum(r2_mask)} voxels with cvr2 > 0.0')

            print(train_data)
            train_data = train_data.loc[:, r2_mask]
            print(train_data)
            test_data = test_data.loc[:, r2_mask]
            gd_pars = fit_model(model_label, model, train_data.loc[:, r2_mask], train_paradigm, max_n_iterations=max_n_iterations)        

        else:

            gd_pars = fit_model(model_label, model, train_data, train_paradigm, max_n_iterations=max_n_iterations)        

            pred = model.predict(paradigm=train_paradigm, parameters=gd_pars)

            r2 = get_rsq(train_data, pred)
            print(r2.describe())

            r2 = r2[r2 < 1.0]
            r2_mask = r2.sort_values(ascending=False).index[:n_voxels]

            gd_pars = gd_pars.loc[r2_mask]
            model.apply_mask(r2_mask)

            train_data = train_data[r2_mask]
            test_data = test_data[r2_mask]


        if separate_sigmas:

            train_narrow_ix = train_paradigm['range'] == 0.0
            train_wide_ix = train_paradigm['range'] == 1.0

            train_data_narrow = train_data[train_narrow_ix]
            train_data_wide = train_data[train_wide_ix]
            train_paradigm_narrow = train_paradigm[train_narrow_ix]
            train_paradigm_wide = train_paradigm[train_wide_ix]

            
            test_narrow_ix = test_paradigm['range'] == 0.0
            test_wide_ix = test_paradigm['range'] == 1.0
            test_data_narrow = test_data[test_narrow_ix]
            test_data_wide = test_data[test_wide_ix]


            stimulus_range_narrow = stimulus_range[::2]   # rows where range==0
            stimulus_range_wide = stimulus_range[1::2]    # rows where range==1

            # Get pdf for narrow condition
            model.init_pseudoWWT(stimulus_range_narrow, gd_pars, subtract_baseline=subtract_wt_baseline)

            residfit_narrow = ResidualFitter(model, train_data_narrow,
                                            train_paradigm_narrow, parameters=gd_pars, lambd=lambd)
            omega_narrow, dof_narrow = residfit_narrow.fit(init_sigma2=0.1,
                    init_dof=10.0,
                    method='t',
                    learning_rate=0.05,
                    spherical=spherical_noise,
                    max_n_iterations=20000 if not debug else 100)

            print('DOF narrow', dof_narrow)
            pdf_narrow = model.get_stimulus_pdf(test_data_narrow, stimulus_range,
                    gd_pars,
                    omega=omega_narrow,
                    dof=dof_narrow,
                    normalize=False)
            

            # Get pdf for wide condition
            model.init_pseudoWWT(stimulus_range_wide, gd_pars, subtract_baseline=subtract_wt_baseline)
            residfit_wide = ResidualFitter(model, train_data_wide,
                                            train_paradigm_wide, parameters=gd_pars, lambd=lambd)
            omega_wide, dof_wide = residfit_wide.fit(init_sigma2=0.1,
                    init_dof=10.0,
                    method='t',
                    learning_rate=0.05,
                    spherical=spherical_noise,
                    max_n_iterations=20000 if not debug else 100)

            print('DOF wide', dof_wide)
            pdf_wide = model.get_stimulus_pdf(test_data_wide, stimulus_range,
                    gd_pars,
                    omega=omega_wide,
                    dof=dof_wide,
                    normalize=False)

            pdf = np.zeros((len(test_data), len(stimulus_range)),  dtype=np.float32)

            pdf[test_narrow_ix, :] = pdf_narrow
            pdf[test_wide_ix, :] = pdf_wide

            pdf = pd.DataFrame(pdf, index=test_data.index, columns=stimulus_mi)
            pdf.index = pd.MultiIndex.from_arrays([
                test_paradigm_full.index.get_level_values('session'),
                test_paradigm_full.index.get_level_values('run'),
                test_paradigm_full.index.get_level_values('trial_nr'),
                test_paradigm['x'].values,
                test_paradigm['range'].values,
            ], names=['session', 'run', 'trial_nr', 'x', 'range'])
            pdfs.append(pdf)

        else:
            model.init_pseudoWWT(stimulus_range, gd_pars, subtract_baseline=subtract_wt_baseline)

            residfit = ResidualFitter(model, train_data,
                                        train_paradigm, parameters=gd_pars, lambd=lambd)

            omega, dof = residfit.fit(init_sigma2=0.1,
                    init_dof=10.0,
                    method='t',
                    learning_rate=0.05,
                    spherical=spherical_noise,
                    max_n_iterations=20000 if not debug else 100)

            print('DOF', dof)

            pdf = model.get_stimulus_pdf(test_data, stimulus_range,
                    gd_pars,
                    omega=omega,
                    dof=dof,
                    normalize=False)

            pdf.columns = stimulus_mi
            pdf.index = pd.MultiIndex.from_arrays([
                test_paradigm_full.index.get_level_values('session'),
                test_paradigm_full.index.get_level_values('run'),
                test_paradigm_full.index.get_level_values('trial_nr'),
                test_paradigm['x'].values,
                test_paradigm['range'].values,
            ], names=['session', 'run', 'trial_nr', 'x', 'range'])
            print(pdf)

            if noise_model == 'lowrank_cv':
                train_full = paradigm_full[~fold_mask]
                assert np.array_equal(train_full['x'].values, train_paradigm['x'].values)
                groups = (train_full.index.get_level_values('session').astype(int).astype(str) + '_' +
                          (train_full.index.get_level_values('run') if inner_groups == 'run'
                           else train_full.index.get_level_values('run2')).astype(int).astype(str))
                groups = np.asarray(groups)
                add_grid_predictor(model, train_paradigm)
                grid_pred = model.predict_stimulus_grid(stimulus_range, parameters=gd_pars)

                # Production noise model on the same parameters, kept in log space
                ref_ll = mvt_grid_loglik(test_data.values, grid_pred, omega, dof)

                if inner_params == 'refit':
                    def inner(d, p, g):
                        return fit_model(model_label, get_model(model_label), d, p,
                                         max_n_iterations=max_n_iterations)
                else:
                    inner = None
                t0 = time.time()
                noise = LowRankNoise.fit(model, train_data, train_paradigm, gd_pars,
                                         stimulus_range=stimulus_range, groups=groups,
                                         inner_parameters=inner,
                                         k=lowrank_k if lowrank_k == 'cv' else int(lowrank_k),
                                         **({} if k_grid is None else dict(k_grid=tuple(k_grid))))
                t_fit = time.time() - t0
                print(noise.summary())
                lr_ll = noise.grid_log_likelihood(test_data, grid_pred)

                idx = pdf.index
                lowrank_lls.append(pd.DataFrame(lr_ll, index=idx, columns=stimulus_mi))
                ref_lls.append(pd.DataFrame(ref_ll, index=idx, columns=stimulus_mi))
                pdf = pd.DataFrame(tempered_posterior(lr_ll, noise.temperature),
                                   index=idx, columns=stimulus_mi)
                noise_log.append(dict(session=test_session, run2=test_run,
                                      n_voxels=train_data.shape[1], k=noise.k,
                                      temperature=noise.temperature,
                                      dof=np.nan if noise.dof is None else noise.dof,
                                      ref_dof=float(np.asarray(dof)),
                                      cv_best=float(noise.cv_scores_.values.max()),
                                      cv_k0_best=float(noise.cv_scores_.loc[0].max()),
                                      cv_T1_best=float(noise.cv_scores_[1.0].max()),
                                      n_groups=len(np.unique(groups)),
                                      inner_params=inner_params, fit_seconds=t_fit))
                cv = noise.cv_scores_.stack().rename('cv_score').reset_index()
                cv.insert(0, 'run2', test_run)
                cv.insert(0, 'session', test_session)
                cv_tables.append(cv)
                gd_tables.append(gd_pars.assign(session=test_session, run2=test_run))

            pdfs.append(pdf)

    pdfs = pd.concat(pdfs)        

    target_fn = op.join(target_dir, f'sub-{subject}_mask-{roi}_nvoxels-{n_voxels}_pars.tsv')
    pdfs.to_csv(target_fn, sep='\t')

    if noise_model == 'lowrank_cv':
        base = op.join(target_dir, f'sub-{subject}_mask-{roi}_nvoxels-{n_voxels}')
        pd.concat(lowrank_lls).to_csv(base + '_desc-lowrank_loglik.tsv', sep='\t')
        pd.concat(ref_lls).to_csv(base + '_desc-refnoise_loglik.tsv', sep='\t')
        pd.DataFrame(noise_log).to_csv(base + '_desc-noisefit_log.tsv', sep='\t', index=False)
        pd.concat(cv_tables).to_csv(base + '_desc-cvscores.tsv', sep='\t', index=False)
        pd.concat(gd_tables).to_csv(base + '_desc-gdpars.tsv', sep='\t')
        print(pd.DataFrame(noise_log).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--model_label', default=3, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-neuralpriors')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--n_voxels', type=int, default=100,
                        help='0: use all voxels with cross-validated R2 > 0 (production); N: top N voxels by training R2')
    parser.add_argument('--fit_responses', action='store_true')
    parser.add_argument('--separate_sigmas', action='store_true',
                        help='Whether to fit separate sigmas for narrow and wide conditions')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--spherical_noise', action='store_true')
    parser.add_argument('--subtract_wt_baseline', action='store_true',
                        help='Subtract per-voxel baseline from basis predictions before computing WWT')
    parser.add_argument('--lambd', type=float, default=0.0,
                        help='Blend weight for empirical sample covariance (0=parametric only, 1=empirical only)')
    parser.add_argument('--noise_model', default='residualfitter', choices=['residualfitter', 'lowrank_cv'],
                        help='residualfitter = production; lowrank_cv = braincoder LowRankNoise with '
                             'nested-CV rank and temperature (opt-in study; needs --tag)')
    parser.add_argument('--tag', default='',
                        help='suffix for a separate derivatives tree (required for lowrank_cv)')
    parser.add_argument('--lowrank_k', default='cv',
                        help="lowrank_cv only: 'cv' (default) or a fixed rank (T still by CV)")
    parser.add_argument('--inner_params', default='fixed', choices=['fixed', 'refit'],
                        help='lowrank_cv only: encoding parameters inside the inner CV '
                             '(fixed = outer-fold fit; refit = refit model per inner fold)')
    parser.add_argument('--inner_groups', default='run', choices=['run', 'run2'],
                        help='lowrank_cv only: inner CV leaves out one original run (default) or one run2 pair')
    parser.add_argument('--k_grid', type=int, nargs='*', default=None,
                        help='lowrank_cv only: candidate ranks (default: braincoder DEFAULT_K_GRID)')
    args = parser.parse_args()

    main(subject=args.subject, model_label=args.model_label, bids_folder=args.bids_folder, smoothed=args.smoothed, debug=args.debug, fit_responses=args.fit_responses, n_voxels=args.n_voxels, spherical_noise=args.spherical_noise, separate_sigmas=args.separate_sigmas, subtract_wt_baseline=args.subtract_wt_baseline, lambd=args.lambd,
         noise_model=args.noise_model, tag=args.tag, lowrank_k=args.lowrank_k,
         inner_params=args.inner_params, inner_groups=args.inner_groups,
         k_grid=args.k_grid)
