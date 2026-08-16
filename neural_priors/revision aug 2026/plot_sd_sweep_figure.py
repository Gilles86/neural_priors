"""Recovery of the width-scaling parameter (sd_wide_scale / r_sigma) across
a dense sweep of generative values (simulate_data_sd_sweep.py), rather than
just the null=1.0 / empirical=1.29 conditions in the production Fig. S9.
Answers: is recovery unbiased across the whole plausible range of r_sigma,
not only at those two points?

10 iterations per generative value (quick diagnostic, not the full
100-iterations-per-condition production analysis); same procedure otherwise
(model 15, noise=0.8 empirically-matched, full design, pooled 250
voxels/iteration).

Writes figS_sd_sweep.pdf/.png to <bids>/derivatives/figures/.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

bids_folder = Path('/data/ds-neuralpriors')

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

COL_DATA = '#3B5BA5'
COL_REGRESSION = '#E07B39'


def load():
    files = list(bids_folder.glob('simulated_recovery_sd_sweep/*/noise_0.8/design_full/iteration-*_results.csv'))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    df = load()
    print(f'n fits: {len(df)}')
    summary = df.groupby('gen_sd_wide_scale')['sd_wide_scale'].agg(['mean', 'std', 'count'])
    print(summary)

    slope, intercept, r, p, se = stats.linregress(df['gen_sd_wide_scale'], df['sd_wide_scale'])
    bias = df['sd_wide_scale'] - df['gen_sd_wide_scale']
    print(f'\nOLS: recovered = {intercept:.3f} + {slope:.3f} x generative, r={r:.4f}, p={p:.2e}')
    print(f'mean bias: {bias.mean():.4f}, sd: {bias.std():.4f}')

    fig, ax = plt.subplots(figsize=(3.6, 3.4), constrained_layout=True)

    lim = (0.5, 2.0)
    ax.axline((0, 0), slope=1, color='0.35', ls='--', lw=.9, zorder=1)
    ax.text(1.85, 1.9, 'Identity', fontsize=7, color='0.35', ha='right', va='bottom')

    x_fit = np.array(lim)
    ax.plot(x_fit, intercept + slope * x_fit, color=COL_REGRESSION, lw=1.3, zorder=2)
    ax.text(1.1, intercept + slope * 1.1 - 0.07, 'Regression', fontsize=7,
           color=COL_REGRESSION, ha='left', va='top')

    ax.scatter(df['gen_sd_wide_scale'], df['sd_wide_scale'], s=14, color=COL_DATA,
              alpha=0.5, edgecolors='none', zorder=3)
    ax.errorbar(summary.index, summary['mean'], yerr=summary['std'], fmt='o', ms=5,
               color=COL_DATA, mec='white', mew=0.6, ecolor=COL_DATA, elinewidth=1.2,
               capsize=2, zorder=4)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_aspect('equal')
    ax.set_xlabel('Generative $r_\\sigma$')
    ax.set_ylabel('Recovered $r_\\sigma$')
    ax.set_title(f'n=10 simulations/value\nr = {r:.3f}, mean bias = {bias.mean():+.3f}', fontsize=8)

    sns.despine(ax=ax, offset=5, trim=True)

    stem = bids_folder / 'derivatives' / 'figures' / 'figS_sd_sweep'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png', dpi=300)
    print('saved', stem)


if __name__ == '__main__':
    main()
