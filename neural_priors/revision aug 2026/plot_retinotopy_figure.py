"""Supplementary figure: numerosity tuning is not a retinotopic/response-bar confound.

Addresses Reviewer 3 (round 2): show, in the supplement, (a) the distributions
of preferred numerosities per range condition with the central tendency
predicted by a retinotopic (foveal-bias) account marked, and (b) the freely
mapped preferred numerosities in both conditions against the predictions of
the efficient-coding model (mu_wide follows the range remapping) and the
retinotopic model (mu_wide = mu_narrow + 7.5, the fixed offset implied by the
response-bar geometry: both bars share 0.25 dva per numerosity unit and only
their centers differ).

Data: model 3 (shift ratio free per voxel, i.e. mu freely mapped in both
conditions), ground-truth stimulus space, NPCr, smoothed, voxels with
cvR2 > 0. Writes PDF + PNG to <bids>/derivatives/figures/.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

bids_folder = Path('/data/ds-neuralpriors')

# Condition colors as in the manuscript (fmri_models_analysis.ipynb COLORS_COND);
# 2D density in Blues as in the manuscript's mu-wide-vs-narrow scatters
COL_NARROW = '#56B4E9'
COL_WIDE = 'C1'
COL_EFFICIENT = '#C0334D'
COL_RETINO = '#3A8F3A'
COL_IDENTITY = '0.35'

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


DVA_PER_UNIT = 0.25  # both bars: 0.25 dva per numerosity unit
RF_CENTERS = [-1.25, 0., 1.25]  # dva
RF_GRAYS = ['0.55', '0.1', '0.55']


def plot_schematic(ax):
    """Two response bars (narrow/wide) over three visuospatial RFs: a voxel
    tuned to a fixed position reads out a numerosity 7.5 higher on the wide bar."""
    from matplotlib.patches import Rectangle

    hw_narrow, hw_wide = 7.5 * DVA_PER_UNIT, 15 * DVA_PER_UNIT  # half-widths in dva
    y_n, y_w, bar_h = 1.35, 1.95, 0.32

    xx = np.linspace(-4.2, 4.2, 500)
    for c, col in zip(RF_CENTERS, RF_GRAYS):
        rf = np.exp(-(xx - c) ** 2 / (2 * 0.5 ** 2))
        ax.fill_between(xx, rf, color=col, alpha=.12, lw=0)
        ax.plot(xx, rf, color=col, lw=1.)
        ax.plot([c, c], [1.02, y_w + bar_h], ls=':', lw=.7, color=col, zorder=0)
    ax.text(-3., 0.65, 'Spatial RFs', fontsize=7, color='0.4', ha='center')
    ax.text(0, -0.32, '+', fontsize=9, color='0.2', ha='center', va='center')
    ax.text(0.35, -0.32, 'Fixation', fontsize=6.5, color='0.4', ha='left', va='center')

    for y0, hw, col, name in [(y_n, hw_narrow, COL_NARROW, 'Narrow'),
                              (y_w, hw_wide, COL_WIDE, 'Wide')]:
        ax.add_patch(Rectangle((-hw, y0), 2 * hw, bar_h, facecolor=col,
                               alpha=.25, edgecolor=col, lw=.8))
        ax.text(-4.35, y0 + bar_h / 2, name, fontsize=7, color=col,
                ha='right', va='center')

    # Numerosities at bar ends and at the RF centers' readout positions;
    # the foveal readouts (the range centers) are bolded — they are the
    # central tendencies marked in panel b
    for x, lab in [(-hw_narrow, '10'), (hw_narrow, '25')]:
        ax.text(x, y_n - 0.11, lab, fontsize=6, ha='center', va='top', color='0.2')
    for x, lab in [(-hw_wide, '10'), (hw_wide, '40')]:
        ax.text(x, y_w + bar_h + 0.08, lab, fontsize=6, ha='center', va='bottom', color='0.2')
    for c in RF_CENTERS:
        weight = 'bold' if c == 0 else 'normal'
        ax.text(c, y_n + bar_h / 2, f'{17.5 + c / DVA_PER_UNIT:g}', fontsize=6,
                ha='center', va='center', color='0.1', fontweight=weight)
        ax.text(c, y_w + bar_h / 2, f'{25 + c / DVA_PER_UNIT:g}', fontsize=6,
                ha='center', va='center', color='0.1', fontweight=weight)
    ax.annotate('', xy=(4.15, y_w + bar_h / 2), xytext=(4.15, y_n + bar_h / 2),
                arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))
    ax.text(4.35, (y_n + y_w + bar_h) / 2, '+7.5', fontsize=7, color='0.2',
            ha='left', va='center')

    ax.set_xlim(-4.4, 5.4)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([-3.75, 0, 3.75])
    ax.set_xticklabels(['−3.75', '0', '3.75'])
    ax.set_xlabel('Horizontal position (dva)')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)


def load_pars(model_label=3):
    pars = pd.read_csv(bids_folder / 'derivatives' / 'encoding_models' /
                       'group_roi-NPCr_desc-groundtruth_parameters.tsv',
                       sep='\t', index_col=[0, 1, 2, 3], header=[0, 1])
    pars = pars.xs(model_label, level='model_label')
    return pars[pars[('cvr2', 'nan')] > 0.0]


def print_stats(pars):
    n, w = pars[('mu', 'narrow')].values, pars[('mu', 'wide')].values
    pred_eff = np.where(n < 10, n, (n - 10) * 2 + 10)
    pred_ret = n + 7.5
    ae_e, ae_r = np.abs(w - pred_eff), np.abs(w - pred_ret)
    subs = pars.index.get_level_values('subject_id')
    per_sub = pd.DataFrame({'eff': ae_e, 'ret': ae_r, 'sub': subs}).groupby('sub').median()
    print(f'n voxels: {len(pars)}, n subjects: {per_sub.shape[0]}')
    print(f'median mu narrow/wide: {np.median(n):.2f} / {np.median(w):.2f}')
    print(f'median abs error: efficient {np.median(ae_e):.2f}, retinotopic {np.median(ae_r):.2f}')
    print(f'voxels better fit by efficient model: {(ae_e < ae_r).mean():.1%}')
    print(f'subjects with lower median error under efficient: '
          f'{(per_sub.eff < per_sub.ret).sum()}/{len(per_sub)}')
    print(stats.wilcoxon(per_sub.eff, per_sub.ret))


def plot_histograms(pars, ax):
    mu = pars['mu'].melt(var_name='range', value_name='mu')
    sns.histplot(mu, x='mu', hue='range', bins=np.arange(0, 41),
                 stat='percent', common_norm=False, element='step', fill=False,
                 palette={'narrow': COL_NARROW, 'wide': COL_WIDE},
                 linewidth=1.2, legend=False, ax=ax)

    # Retinotopic (foveal-bias) prediction: tuning centered on the middle of
    # each response bar, i.e. the middle of each numerosity range. Lines stop
    # short of the top so the shared label sits directly above them.
    ax.vlines([17.5, 25.], 0, 13., colors=[COL_NARROW, 'C1'], lw=1., ls='--')
    ax.text(21.25, 13.8, 'Retinotopic prediction\n(range centers)', fontsize=7,
            ha='center', va='bottom', color='0.3')

    med_narrow = pars[('mu', 'narrow')].median()
    med_wide = pars[('mu', 'wide')].median()
    for x, col in [(med_narrow, COL_NARROW), (med_wide, COL_WIDE)]:
        ax.plot(x, 15.7, marker='v', ms=4.5, color=col, clip_on=False)
    ax.text(9.4, 15.7, 'Medians', fontsize=7, ha='right', va='center', color='0.3')

    ax.text(6.8, 13., 'Narrow', fontsize=8, color=COL_NARROW, ha='right')
    ax.text(19., 6.2, 'Wide', fontsize=8, color='C1', ha='left')

    ax.set_xlim(0, 40)
    ax.set_ylim(0, 16)
    ax.set_xticks([0, 10, 17.5, 25, 40])
    ax.set_xticklabels(['0', '10', '17.5', '25', '40'])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel('Preferred numerosity μ')
    ax.set_ylabel('Percent of voxels')


def plot_model_comparison(pars, ax, fig):
    n, w = pars[('mu', 'narrow')].values, pars[('mu', 'wide')].values
    keep = (n >= 0) & (n <= 40) & (w >= 0) & (w <= 40)
    hb = ax.hexbin(n[keep], w[keep], gridsize=(60 * np.array([np.sqrt(3), 1])).astype(int),
                   linewidths=0., cmap='Blues', mincnt=1, edgecolors='none',
                   norm=mpl.colors.PowerNorm(0.5, vmin=0))

    ax.axline((0, 0), slope=1, color=COL_IDENTITY, ls='--', lw=.7, zorder=100)
    ax.plot([0, 10, 40], [0, 10, 70], c=COL_EFFICIENT, lw=1.8, zorder=101)
    ax.plot([0, 40], [7.5, 47.5], c=COL_RETINO, lw=1.8, ls='--', zorder=101)

    # Labels along the lines (rotations match slopes on these equal-aspect axes)
    ax.text(21.6, 36.5, 'Efficient shift', fontsize=7, color=COL_EFFICIENT,
            ha='center', va='center', rotation=63.43)
    ax.text(30.5, 36., 'Retinotopic (+7.5)', fontsize=7, color=COL_RETINO,
            ha='center', va='center', rotation=45)
    ax.text(37., 34.6, 'Identity', fontsize=6.5, color=COL_IDENTITY,
            ha='center', va='center', rotation=45)

    cbar_ax = ax.inset_axes([0.97, 0.08, .015, .22])
    cb = fig.colorbar(hb, cax=cbar_ax, label='Count', ticks=[1, int(hb.get_array().max())])
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(axis='both', length=0, labelsize=7)
    cb.ax.yaxis.label.set_size(7)
    cb.outline.set_visible(False)

    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal')
    ax.set_xticks([0, 10, 25, 40])
    ax.set_yticks([0, 10, 25, 40])
    ax.set_xlabel('μ (narrow)')
    ax.set_ylabel('μ (wide)')


def main():
    pars = load_pars()
    print_stats(pars)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55),
                             gridspec_kw=dict(width_ratios=[1.1, 1.25, 1]),
                             constrained_layout=True)
    plot_schematic(axes[0])
    plot_histograms(pars, axes[1])
    plot_model_comparison(pars, axes[2], fig)

    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='bottom', ha='right')
    sns.despine(ax=axes[0], offset=5, left=True, trim=True)
    sns.despine(ax=axes[1], offset=5, trim=True)
    sns.despine(ax=axes[2], offset=5)

    stem = bids_folder / 'derivatives' / 'figures' / 'figS_retinotopy_control'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png', dpi=300)
    print('saved', stem)


if __name__ == '__main__':
    main()
