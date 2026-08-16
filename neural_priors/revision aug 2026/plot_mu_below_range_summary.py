"""Reproducible source for the below-range mu_narrow recovery numbers quoted in
notes/response_mu_below_range.md (Reviewer 4, point 4) and the si.tex sentence
on cautious interpretation of preferred numerosities below the presented range.

Pools per-voxel generative-vs-recovered mu_narrow across ALL cells of the
width-recovery simulation at noise=0.8: both sampling schemes (pooled,
subject-wise), both designs (full, censored), both generative sd_wide_scale
values (1.0, 1.287794) -- 800 files, 194,212 simulated voxels total. There is
no mu_wide column in this data: delta_wide was fixed at 2.0 (not estimated
per voxel) in the width-recovery simulation, so only the narrow-condition
preferred numerosity has an independently recovered value here. See
`revision feb 2026/simulate_data.py` / `recovery_bias.ipynb` (Fig. 8) if the
shift-recovery (mu_wide) numbers are needed instead.

Bins are right-closed ((-inf,5], (5,10], (10,25], (25,inf)) via pd.cut,
matching the originally-quoted table exactly.

Writes model_recovery_mu_below_range_summary.pdf/.png to
<bids>/derivatives/figures/.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

bids_folder = Path('/data/ds-neuralpriors')

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

BINS = [-np.inf, 5, 10, 25, np.inf]
LABELS = ['< 5', '5–10', '10–25\n(in range)', '> 25']
COL_IN = '#3A8F3A'
COL_OUT = '#C0334D'


def load():
    files = list(bids_folder.glob('simulated_recovery_sd/*/noise_0.8/design_*/iteration-*_pervoxel.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['bin'] = pd.cut(df['gen_mu_narrow'], bins=BINS, labels=LABELS)
    return df


def compute_table(df):
    rows = []
    for label, sub in df.groupby('bin', observed=True):
        rho, _ = stats.spearmanr(sub['gen_mu_narrow'], sub['est_mu_narrow'])
        err = sub['est_mu_narrow'] - sub['gen_mu_narrow']
        rows.append(dict(bin=label, n=len(sub), rho=rho, mae=err.abs().median(), bias=err.median()))
    table = pd.DataFrame(rows).set_index('bin').loc[LABELS]

    below = df[df['gen_mu_narrow'] < 10]
    inrange = df[(df['gen_mu_narrow'] > 10) & (df['gen_mu_narrow'] <= 25)]
    rho_below, _ = stats.spearmanr(below['gen_mu_narrow'], below['est_mu_narrow'])
    print(table.to_string(float_format=lambda x: f'{x:.2f}'))
    print(f'below-range classified est<10: {(below["est_mu_narrow"] < 10).mean():.1%} (n={len(below)})')
    print(f'in-range classified est in [10,25]: '
          f'{((inrange["est_mu_narrow"] >= 10) & (inrange["est_mu_narrow"] <= 25)).mean():.1%} (n={len(inrange)})')
    print(f'pooled below-range (gen<10) rho: {rho_below:.3f} (n={len(below)})')
    return table


def plot_summary(table, df):
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.4), constrained_layout=True)
    colors = [COL_OUT, COL_OUT, COL_IN, COL_OUT]

    axes[0].bar(LABELS, table['rho'], color=colors)
    axes[0].set_ylabel('Spearman ρ\n(generative vs. recovered)')
    axes[0].set_ylim(-0.2, 1)
    axes[0].axhline(0, color='0.5', lw=0.6)

    axes[1].bar(LABELS, table['mae'], color=colors)
    axes[1].set_ylabel('Median absolute error')

    axes[2].bar(LABELS, table['n'], color=colors)
    axes[2].set_ylabel('n simulated voxels')
    axes[2].set_yscale('log')

    for ax in axes:
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel('Generative preferred numerosity')

    fig.suptitle('Recovery of preferred numerosity (narrow condition), by generative bin', fontsize=9)

    stem = bids_folder / 'derivatives' / 'figures' / 'model_recovery_mu_below_range_summary'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png', dpi=300)
    print('saved', stem.name)


if __name__ == '__main__':
    df = load()
    print(f'total voxels: {len(df)}')
    table = compute_table(df)
    plot_summary(table, df)
