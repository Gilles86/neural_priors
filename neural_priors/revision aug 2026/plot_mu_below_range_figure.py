"""Supplementary figure: how well nPRF preferred numerosity is recovered,
combining real empirical data with the parameter-recovery simulation
(Reviewer 4, point 4; notes/response_mu_below_range.md). Redesigned after
Arthur's draft (notes/figS11_v1.png): panel a is empirical, panels b/c are
simulation-based.

Panel a: EMPIRICAL data -- per-voxel recovered preferred numerosity in the
narrow vs. wide condition, from model 3 (free per-voxel shift; NPCr voxels,
cross-validated R^2 > 0, 39 participants, n=7,330). Dashed line: identity.
Solid line: OLS regression -- its slope above 1 is the per-voxel signature
of the same range-adaptation shift reported at the population level.

Panel b: Spearman rho between generative and recovered mu_narrow, by
generative-mu bin, from the parameter-recovery SIMULATION (noise=0.8,
194,212 simulated voxels; same table as plot_mu_below_range_summary.py).
Panel c: median absolute recovery error, same bins, same simulation. No
error bars on either b or c: rho has no natural SE, and neither does a
median without bootstrapping, so none is fabricated. No n= annotations
(shown in the printed console table instead).

Data note: panel a (empirical, model 3) and panels b/c (simulated
recovery, model 15) are DIFFERENT populations/models -- panel a shows the
real per-voxel shift, panels b/c show how well the fitting procedure can
recover a known ground truth. Keep the axis/title wording explicit about
this so the two are not conflated.

Writes figS_mu_below_range.pdf/.png to <bids>/derivatives/figures/.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

bids_folder = Path('/data/ds-neuralpriors')

COL_IN = '#3A8F3A'
COL_OUT = '#C0334D'
COL_REGRESSION = '#E07B39'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
sns.set_context('paper')

BINS = [-np.inf, 5, 10, 25, np.inf]
BIN_LABELS = ['< 5', '5–10', '10–25', '> 25']
BIN_COLORS = [COL_OUT, COL_OUT, COL_IN, COL_OUT]
DESIGN_MAP = {'Full': 'Full design (10–25 and 10–40)', 'Censored': 'Censored design (10–25 only)'}
MU_LIM = 10


def load_empirical(model_label=3):
    """Below-range voxels only, in BOTH conditions (0 <= recovered mu < 10
    for mu_narrow AND mu_wide) -- this is the population Reviewer 4's
    concern is about, and what Arthur's draft (figS11_v1.png) actually
    plots (its axes top out at 10 because the population is restricted,
    not because of a cosmetic axis crop). Restricting mu_wide too (not just
    mu_narrow) keeps the plotted population, the regression-fit population,
    and the reported n all the same set -- a handful of voxels with
    mu_narrow<10 but a wildly large recovered mu_wide (up to >100, almost
    certainly degenerate fits) would otherwise sit off-panel yet still
    distort the fitted line."""
    path = bids_folder / 'derivatives' / 'extracted_pars' / f'group_roi-NPCr_model-{model_label}_desc-groundtruth_parameters.tsv'
    df = pd.read_csv(path, sep='\t', index_col=[0, 1, 2, 3], header=[0, 1])
    df = df[df[('cvr2', 'nan')] > 0]
    df = pd.DataFrame({'mu_narrow': df[('mu', 'narrow')].values,
                       'mu_wide': df[('mu', 'wide')].values})
    return df[(df['mu_narrow'] >= 0) & (df['mu_narrow'] < 10) &
             (df['mu_wide'] >= 0) & (df['mu_wide'] < 10)]


def deming_slope_through_origin(x, y, delta=1.0):
    """Deming (errors-in-variables) regression slope through the origin,
    assuming equal error variance in x and y (delta=1) -- appropriate here
    because BOTH mu_narrow and mu_wide are noisy fitted quantities, not one
    clean predictor + one noisy outcome, which is what OLS assumes."""
    Sxx, Syy, Sxy = np.sum(x ** 2), np.sum(y ** 2), np.sum(x * y)
    return ((Syy - delta * Sxx) + np.sqrt((Syy - delta * Sxx) ** 2 + 4 * delta * Sxy ** 2)) / (2 * Sxy)


def load_simulation():
    files = list(bids_folder.glob('simulated_recovery_sd/*/noise_0.8/design_*/iteration-*_pervoxel.csv'))
    df = pd.concat([pd.read_csv(f).assign(
        design=f.parent.name.replace('design_', '').replace('_subjectwise', ''))
        for f in files], ignore_index=True)
    df['design'] = df['design'].str.capitalize().map(DESIGN_MAP)
    df['bin'] = pd.cut(df['gen_mu_narrow'], bins=BINS, labels=BIN_LABELS)
    return df


def compute_table(df):
    rows = []
    for label in BIN_LABELS:
        sub = df[df['bin'] == label]
        rho, _ = stats.spearmanr(sub['gen_mu_narrow'], sub['est_mu_narrow'])
        abs_err = (sub['est_mu_narrow'] - sub['gen_mu_narrow']).abs()
        err = sub['est_mu_narrow'] - sub['gen_mu_narrow']
        rows.append(dict(bin=label, n=len(sub), rho=rho,
                         mae_median=abs_err.median(), bias=err.median(),
                         mae_mean=abs_err.mean(), mae_sem=abs_err.std(ddof=1) / np.sqrt(len(sub))))
    table = pd.DataFrame(rows).set_index('bin').loc[BIN_LABELS]

    below = df[df['gen_mu_narrow'] < 10]
    inrange = df[(df['gen_mu_narrow'] > 10) & (df['gen_mu_narrow'] <= 25)]
    print(table.to_string(float_format=lambda x: f'{x:.2f}'))
    print(f'below-range classified est<10: {(below["est_mu_narrow"] < 10).mean():.1%} (n={len(below)})')
    print(f'in-range classified est in [10,25]: '
          f'{((inrange["est_mu_narrow"] >= 10) & (inrange["est_mu_narrow"] <= 25)).mean():.1%} (n={len(inrange)})')
    return table


def compute_range_split_bias(df):
    """Bias (recovered - generative mu_narrow) split by whether the voxel's
    generative mu_narrow falls inside the presented stimulus range [10, 25]
    or outside it (< 10 or > 25) -- the two-way version of compute_table()'s
    per-bin bias, for the 'estimates remained unbiased on average' claim."""
    in_range = df[(df['gen_mu_narrow'] >= 10) & (df['gen_mu_narrow'] <= 25)]
    out_range = df[(df['gen_mu_narrow'] < 10) | (df['gen_mu_narrow'] > 25)]

    print()
    for label, sub in [('overall (pooled)', df), ('in-range (10<=mu<=25)', in_range),
                       ('out-of-range (mu<10 or mu>25)', out_range)]:
        err = sub['est_mu_narrow'] - sub['gen_mu_narrow']
        t, p = stats.ttest_1samp(err, 0)
        print(f'{label}: median bias = {err.median():.4f}, mean bias = {err.mean():.4f} '
              f'(SD={err.std():.4f}, SEM={err.std()/np.sqrt(len(sub)):.4f}), '
              f't={t:.2f}, p={p:.3g}, n={len(sub)}')


def plot_empirical_panel(ax, fig, df):
    """Empirical (real-data) per-voxel narrow-vs-wide recovered mu, model 3,
    below-range voxels in BOTH conditions (0 <= mu < 10). Regression is
    Deming (errors-in-variables, delta=1), THROUGH THE ORIGIN, matching
    Arthur's draft -- OLS is the wrong tool here since both mu_narrow and
    mu_wide are noisy fitted quantities, not one clean predictor; a plain
    OLS slope would be attenuated by the noise in mu_narrow. Fit on the
    same population that's plotted (mu_wide restricted to <10 too, not
    just mu_narrow) so the line isn't pulled around by a few off-panel
    voxels with degenerate, wildly large recovered mu_wide.
    Identity/regression labels are placed in the empty upper-left region,
    off both lines entirely, so neither label crosses a line (scientific-
    figures skill: no overlapping ink)."""
    hb = ax.hexbin(df['mu_narrow'], df['mu_wide'], gridsize=35, extent=(0, MU_LIM, 0, MU_LIM),
                   linewidths=0., cmap='Blues', mincnt=1, edgecolors='none',
                   norm=mpl.colors.PowerNorm(0.4, vmin=0))

    ax.axline((0, 0), slope=1, color='0.35', ls='--', lw=.9, zorder=100)

    slope = deming_slope_through_origin(df['mu_narrow'].values, df['mu_wide'].values)
    x_fit = np.array([0, MU_LIM])
    ax.plot(x_fit, slope * x_fit, color=COL_REGRESSION, ls='-', lw=1.3, zorder=101)

    ax.text(0.3, MU_LIM - 0.3, 'Identity', fontsize=7, color='0.35', ha='left', va='top')
    ax.text(0.3, MU_LIM - 1.3, 'Regression', fontsize=7, color=COL_REGRESSION, ha='left', va='top')
    ax.text(0.3, MU_LIM - 2.3, f'slope = {slope:.2f}', fontsize=7, color=COL_REGRESSION, ha='left', va='top')

    cbar_ax = ax.inset_axes([0.72, 0.08, .03, .18])
    cb = fig.colorbar(hb, cax=cbar_ax, label='Voxel count', ticks=[1, int(hb.get_array().max())])
    cb.ax.yaxis.set_label_position('left')
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.ax.yaxis.label.set_size(6.5)
    cb.outline.set_visible(False)

    ax.set_title(f'Empirical data, below-range voxels\n(model 3, free per-voxel shift, n={len(df):,})', fontsize=8.5)
    ax.set_xlim(0, MU_LIM)
    ax.set_ylim(0, MU_LIM)
    ax.set_xticks([0, 2.5, 5, 7.5, 10])
    ax.set_yticks([0, 2.5, 5, 7.5, 10])
    ax.set_aspect('equal')
    ax.set_xlabel('Recovered μ (narrow condition)')
    ax.set_ylabel('Recovered μ (wide condition)')


def _bin_point_plot(ax, table, col, ylabel, err_col=None, fmt='{:.2f}'):
    x = np.arange(len(table))
    ax.plot(x, table[col], color='0.6', lw=1., zorder=1)
    if err_col is not None:
        ax.errorbar(x, table[col], yerr=table[err_col], fmt='none',
                   ecolor='0.4', elinewidth=1., capsize=2, zorder=1.5)
    ax.scatter(x, table[col], s=55, color=BIN_COLORS, zorder=2,
              edgecolors='white', linewidths=0.6)

    headroom = (table[col].max() - table[col].min()) * 0.12 or 0.05
    for xi, v in zip(x, table[col]):
        ax.text(xi, v + headroom, fmt.format(v), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(table.index, fontsize=7)
    ax.set_xlim(-0.5, len(table) - 0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Generative μ (simulation,\nboth designs pooled)')
    return headroom


def plot_rho_points(ax, table):
    """Spearman rho per generative-mu bin (simulation). No error bars: each
    value is a single Spearman correlation computed over all pooled voxels
    in that bin, not a mean over independent replicates, so there is no
    natural SE."""
    _bin_point_plot(ax, table, 'rho', 'Spearman ρ\n(generative vs. recovered)')
    ax.axhline(0, color='0.7', lw=0.6, zorder=0)
    ax.set_ylim(-0.3, 1.05)


def plot_mae_points(ax, table):
    """Median absolute recovery error per generative-mu bin (simulation).
    No error bars: a median has no natural SE without bootstrapping, so
    none is fabricated -- same rationale as panel b's correlation."""
    _bin_point_plot(ax, table, 'mae_median', 'Median absolute error\n(generative vs. recovered μ)')
    ax.set_ylim(0, table['mae_median'].max() * 1.3)


def main():
    empirical = load_empirical()
    sim = load_simulation()
    print(f'empirical voxels (model 3, cvr2>0): {len(empirical)}')
    print(f'simulated voxels: {len(sim)}')
    table = compute_table(sim)
    compute_range_split_bias(sim)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9),
                             gridspec_kw=dict(width_ratios=[1.15, 1., 1.]),
                             constrained_layout=True)
    plot_empirical_panel(axes[0], fig, empirical)
    plot_rho_points(axes[1], table)
    plot_mae_points(axes[2], table)

    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.2, 1.08, letter, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='bottom', ha='right')

    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=True)

    stem = bids_folder / 'derivatives' / 'figures' / 'figS_mu_below_range'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png', dpi=300)
    print('saved', stem)


if __name__ == '__main__':
    main()
