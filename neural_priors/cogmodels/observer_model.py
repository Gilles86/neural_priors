"""Bayesian-observer model of the estimation reports with two range conditions.

Thin extension of ``bauer.models.EstimationTaskModel`` (bauer branch
``estimation-model``, commit 4f90289; used read-only via PYTHONPATH) for the
neural_priors design, where the stimulus range and the response slider differ between
conditions: narrow n in 10..25 and wide n in 10..40, each slider spanning exactly its
range. The base class has one prior support and one response range per model; this
subclass makes both condition-specific:

- **Prior**: discrete truncated Gaussian on the internal (f_alpha) scale with mass only
  on the condition's integers (narrow 10..25, wide 10..40).
- **Responses**: posterior-mean estimate + Gaussian motor noise sigma_r (items),
  rounded to the integers and censored at the condition's slider ends (the probability
  beyond 25 is folded into 25 for narrow trials).
- **Lapse**: uniform over the condition's slider positions (16 vs 31).
- **Condition offsets**: every parameter in ``condition_parameters`` gets a
  hierarchical narrow-minus-wide offset on its untransformed scale,
  x[s, c] = x_s + (1/2 - c) (delta + delta_s), delta_s ~ N(0, delta_sd), c = 0 narrow,
  1 wide (as in the base class, but with prior scales per parameter, ``COND_SCALE``).
  ``{p}_cond_mu`` is the group narrow-minus-wide difference (untransformed scale);
  ``{p}_by_condition`` the participant values per condition.

- **Measurement grid**: spans f(10) - 5 nu_f .. f(40) + 5 nu_f (200 points), i.e. it
  follows the noise. Against a brute-force reference (20,001-point measurement
  integral) the option probabilities agree to |dP| <= 1.6e-4 over alpha -1.5..1.5,
  nu .06..0.5, sigma_r .6..2.5, both conditions (the base class's fixed f(1)..f(50)
  grid was off by up to .09 at alpha = -1, nu = .12, sigma_r = .8).

Everything else (encoding f_alpha(n) = (n^alpha - 1)/alpha with noise nu in log units at
n_ref = 12, the measurement/estimate grid and its inversion, censoring of unfinished
reports through a ``censor`` column, the hierarchical non-centred priors) is the base
class's.
"""
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from bauer.models import (EstimationTaskModel, PowerSpace, LogSpace, LinearSpace,
                          TruncatedGaussianPrior)
from bauer.models.estimation_task import _norm_cdf  # noqa: F401  (kept for reference)

RANGES = {'narrow': (10, 25), 'wide': (10, 40)}
LEVELS = ['narrow', 'wide']            # c = 0, 1 (sorted, as in the base class)

# prior s.d. of the group narrow-minus-wide offset (untransformed scale) and of its
# between-participant s.d. (HalfNormal)
COND_SCALE = {'alpha': (0.5, 0.25), 'nu': (0.5, 0.25), 'prior_mu': (10.0, 3.0),
              'prior_sd': (0.5, 0.25), 'sigma_r': (0.5, 0.25), 'p_lapse': None}


def space_for(name):
    return {'log': LogSpace, 'linear': LinearSpace, 'power': PowerSpace}[name]()


class RangeEstimationModel(EstimationTaskModel):

    def __init__(self, paradigm=None, encoding_space=None, ranges=RANGES,
                 condition_parameters=(), meas_halfwidth_sd=5.0, **kwargs):
        self.ranges = ranges
        self.meas_halfwidth_sd = float(meas_halfwidth_sd)
        lo = min(r[0] for r in ranges.values())
        hi = max(r[1] for r in ranges.values())
        kwargs.setdefault('motor_noise', True)
        kwargs.setdefault('lapse', True)
        super().__init__(paradigm, encoding_space=encoding_space,
                         prior=TruncatedGaussianPrior(lo, hi, discrete=True),
                         response_range=(lo, hi), condition='range',
                         condition_parameters=condition_parameters, **kwargs)
        H = len(self.hyp_grid)
        R = len(self.response_options)
        self.cond_prior_mask = np.zeros((2, H))
        self.cond_fold = np.zeros((2, R, R))
        self.cond_lapse = np.zeros((2, R))
        for c, lev in enumerate(LEVELS):
            a, b = ranges[lev]
            self.cond_prior_mask[c] = ((self.hyp_grid >= a) & (self.hyp_grid <= b)).astype(float)
            for i, o in enumerate(self.response_options):
                self.cond_fold[c, i, int(np.clip(o, a, b) - lo)] = 1.0
            ok = (self.response_options >= a) & (self.response_options <= b)
            self.cond_lapse[c] = ok / ok.sum()

    def get_free_parameters(self):
        pars = super().get_free_parameters()
        if 'prior_mu' in pars:
            # centres of 10..25 and 10..40 are 17.5 and 25 (base default N(12, 4) is
            # tuned to the retinonumeral 5..24)
            pars['prior_mu'] = dict(mu_intercept=20.0, sigma_intercept=10.0,
                                    cauchy_sigma_intercept=4.0, transform='softplus')
        return pars

    # ---- condition offsets with per-parameter scales --------------------------------
    def _condition_inputs(self, inputs, S):
        model = pm.Model.get_context()
        sign = pt.as_tensor_variable(np.array([.5, -.5]))
        out = {}
        for k, v in inputs.items():
            if k not in self.condition_parameters:
                out[k] = pt.repeat(v, 2)
                continue
            tr = self.free_parameters[k].get('transform', 'identity')
            if tr not in ('identity', 'softplus'):
                raise NotImplementedError(f'condition offsets for {tr} parameters')
            base = v if tr == 'identity' else model[f'{k}_untransformed']
            mu_scale, sd_scale = COND_SCALE[k]
            d_mu = pm.Normal(f'{k}_cond_mu', 0.0, mu_scale)
            d_sd = pm.HalfNormal(f'{k}_cond_sd', sd_scale)
            d_off = pm.Normal(f'{k}_cond_offset', 0.0, 1.0, dims=('subject',))
            delta = d_mu + d_sd * d_off
            x = base[:, None] + sign[None, :] * delta[:, None]
            x = x if tr == 'identity' else pt.softplus(x)
            pm.Deterministic(f'{k}_by_condition', x, dims=('subject', 'condition'))
            out[k] = x.reshape((2 * S,))
        return out

    # ---- condition-specific prior support -------------------------------------------
    def estimate_on_grid(self, inputs, S):
        """As the base class, with the prior mask of row r = 2s + c set by c."""
        alpha = self._alpha(inputs, S)
        nu_f = self._nu_f(inputs, alpha)
        h = pt.as_tensor_variable(self.hyp_grid)
        fh = self._f(h, alpha)
        # measurement grid adapted to the noise: f(10) - k nu_f .. f(40) + k nu_f
        # (the base class's fixed f(1)..f(50) span is too coarse for small nu at
        # alpha << 0 and the hypothesis-range span too narrow for large nu)
        f_ends = self._f(pt.as_tensor_variable(self.hyp_grid[[0, -1]]), alpha)
        k = self.meas_halfwidth_sd
        lo = f_ends[:, :1] - k * nu_f[:, None]
        hi = f_ends[:, 1:] + k * nu_f[:, None]
        t = pt.as_tensor_variable(np.linspace(0.0, 1.0, self.n_measurement_grid))
        m = lo + t[None, :] * (hi - lo)
        loglik = -0.5 * ((m[:, :, None] - fh[:, None, :]) / nu_f[:, None, None]) ** 2
        mask = np.tile(self.cond_prior_mask, (S // 2, 1))                  # (S, H)
        logmask = pt.as_tensor_variable(np.log(mask + 1e-300))
        mu_f = self.encoding_space.transform(inputs['prior_mu'], alpha)
        sd_f = inputs['prior_sd'] * inputs['prior_mu'] ** alpha
        logprior = (-0.5 * ((fh - mu_f[:, None]) / sd_f[:, None]) ** 2
                    + (alpha[:, None] - 1) * pt.log(h)[None, :]) + logmask
        logpost = loglik + logprior[:, None, :]
        logpost = logpost - pt.max(logpost, axis=2, keepdims=True)
        post = pt.exp(logpost)
        post = post / pt.sum(post, axis=2, keepdims=True)
        n_hat = pt.sum(post * h[None, None, :], axis=2)
        return m, n_hat, alpha, nu_f

    # ---- condition-specific slider ends ---------------------------------------------
    def _option_table_grid(self, inputs, n_unique, S, shift_nu=None):
        table = super()._option_table_grid(inputs, n_unique, S, shift_nu)    # (S, K, R)
        fold = pt.as_tensor_variable(np.tile(self.cond_fold, (S // 2, 1, 1)))  # (S, R, R)
        return pt.batched_dot(table, fold)

    def trial_probabilities(self, inputs, n, subject_ix, S, n_ix=None, n_unique=None,
                            shift_nu=None):
        table = self._option_table_grid(inputs, n_unique, S, shift_nu)
        p = table[subject_ix, n_ix]
        if self.lapse:
            lam = inputs['p_lapse'][subject_ix][:, None]
            L = pt.as_tensor_variable(self.cond_lapse)[subject_ix % 2]
            p = (1 - lam) * p + lam * L
        p = pt.clip(p, 1e-12, 1.0)
        return p / pt.sum(p, axis=1, keepdims=True)

    # ---- prediction with per-condition parameters (simulation, recovery) -----------
    def option_probs(self, paradigm, pars):
        """(T, R) response-option probabilities.

        ``pars``: DataFrame indexed by (subject, range) with one column per free
        parameter (participant x condition values on the natural scale).
        """
        subj = paradigm.index.get_level_values('subject')
        subjects = pd.unique(subj)
        cond = (np.asarray(paradigm['range']) == 'wide').astype(int)
        s_ix = pd.Index(subjects).get_indexer(subj)
        rows = 2 * s_ix + cond
        S2 = 2 * len(subjects)
        idx = pd.MultiIndex.from_product([subjects, LEVELS])
        P = pars.loc[idx]
        inputs = {k: pt.as_tensor_variable(P[k].to_numpy(float)) for k in self.free_parameters}
        n = paradigm['n'].to_numpy(float)
        n_unique, n_ix = np.unique(n, return_inverse=True)
        p = self.trial_probabilities(inputs, pt.as_tensor_variable(n),
                                     pt.as_tensor_variable(rows), S2,
                                     n_ix=pt.as_tensor_variable(n_ix), n_unique=n_unique).eval()
        # exact zeros outside the condition's slider
        lo_hi = np.array([self.ranges[l] for l in LEVELS])[cond]
        o = self.response_options[None, :]
        p = np.where((o >= lo_hi[:, :1]) & (o <= lo_hi[:, 1:]), p, 0.0)
        return p / p.sum(1, keepdims=True)

    def simulate_responses(self, paradigm, pars, n_samples=1, rng=None):
        rng = np.random.default_rng(rng)
        p = self.option_probs(paradigm, pars)
        cdf = np.cumsum(p, axis=1)
        u = rng.random((n_samples, len(p), 1))
        ix = np.minimum((u > cdf[None]).sum(-1), p.shape[1] - 1)
        return self.response_options[ix].T                                   # (T, n_samples)


def posterior_by_condition(idata, model, stat='mean'):
    """Participant x condition parameter table from a fit (natural scale)."""
    post = idata.posterior
    cols = {}
    for k in model.free_parameters:
        name = f'{k}_by_condition' if f'{k}_by_condition' in post else k
        v = post[name]
        v = v.mean(('chain', 'draw')) if stat == 'mean' else v.median(('chain', 'draw'))
        s = v.to_series()
        if 'condition' not in v.dims:
            s = pd.concat({lev: s for lev in LEVELS}, names=['range']).swaplevel()
        cols[k] = s
    T = pd.DataFrame(cols)
    T.index = T.index.set_names(['subject', 'range'])
    T.index = T.index.set_levels([T.index.levels[0].astype(str), T.index.levels[1]])
    return T
