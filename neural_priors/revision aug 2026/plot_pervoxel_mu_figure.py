"""Per-voxel recovery of preferred numerosity (Reviewer 4, below-range nPRFs).

Generative vs. recovered preferred numerosity (narrow condition), one panel
per design, across all simulated voxels of the width-recovery simulations at
the empirically matched noise level (0.8) -- both sampling schemes (pooled,
subject-wise) and both generative sd_wide_scale values pooled within each
design panel. House style matching plot_retinotopy_figure.py /
plot_mu_below_range_figure.py (same SI figure set). Dashed lines mark the
presented range [10, 25]; the dashed diagonal is the identity.

Writes model_recovery_mu_pervoxel.pdf/.png to <bids>/derivatives/figures/.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

bids_folder = Path('/data/ds-neuralpriors')

COL_IN = '#3A8F3A'

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


def load():
    files = list(bids_folder.glob('simulated_recovery_sd/*/noise_0.8/design_*/iteration-*_pervoxel.csv'))
    df = pd.concat([pd.read_csv(f).assign(
        design=f.parent.name.replace('design_', '').replace('_subjectwise', ''))
        for f in files], ignore_index=True)
    df['design'] = df['design'].str.capitalize().map(
        {'Full': 'Full design (10–25 and 10–40)', 'Censored': 'Censored design (10–25 only)'})
    return df


def plot_panel(ax, fig, sub, show_cbar):
    hb = ax.hexbin(sub['gen_mu_narrow'], sub['est_mu_narrow'], gridsize=70,
                   extent=(0, 40, 0, 40), linewidths=0., cmap='Blues',
                   mincnt=1, edgecolors='none', norm=mpl.colors.PowerNorm(0.4, vmin=0))

    ax.add_patch(Rectangle((10, 10), 15, 15, facecolor=COL_IN, alpha=.08, lw=0, zorder=0))
    ax.axline((0, 0), slope=1, color='0.35', ls='--', lw=.7, zorder=100)
    ax.vlines([10, 25], 0, 37, colors='0.4', ls='--', lw=.8, zorder=100)
    ax.text(17.5, 38, 'Presented range', fontsize=7, ha='center', va='bottom', color='0.3')

    if show_cbar:
        cbar_ax = ax.inset_axes([0.96, 0.06, .02, .22])
        cb = fig.colorbar(hb, cax=cbar_ax, label='Voxel count', ticks=[1, int(hb.get_array().max())])
        cb.ax.yaxis.set_label_position('left')
        cb.ax.tick_params(length=0, labelsize=6.5)
        cb.ax.yaxis.label.set_size(6.5)
        cb.outline.set_visible(False)

    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xticks([0, 10, 25, 40])
    ax.set_yticks([0, 10, 25, 40])
    ax.set_aspect('equal')
    ax.set_xlabel('Generative μ (narrow condition)')


def main():
    df = load()
    designs = ['Full design (10–25 and 10–40)', 'Censored design (10–25 only)']

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.7), constrained_layout=True)
    for ax, design in zip(axes, designs):
        plot_panel(ax, fig, df[df['design'] == design], show_cbar=(design == designs[-1]))
        ax.set_title(design, fontsize=9)
    axes[0].set_ylabel('Recovered μ (narrow condition)')

    for ax, letter in zip(axes, 'ab'):
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='bottom', ha='right')

    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=True)

    stem = bids_folder / 'derivatives' / 'figures' / 'model_recovery_mu_pervoxel'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png', dpi=300)
    print('saved', stem.name)


if __name__ == '__main__':
    main()
