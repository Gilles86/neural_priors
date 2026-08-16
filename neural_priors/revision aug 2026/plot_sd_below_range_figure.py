"""Supplementary figure: recovery of tuning width (sd_narrow) as a function
of whether the voxel's preferred numerosity falls inside or outside the
presented stimulus range -- the reviewer's mu/width identifiability
trade-off (see revision aug 2026/README.md, notes/response_sd_recovery.md),
now broken out the same way as plot_mu_below_range_figure.py.

Panels a/b: generative-vs-recovered sd_narrow, split by design (full,
censored) -- pooling both sampling schemes and both generative sd_wide_scale
values within each design. Panel c: Spearman rho between generative and
recovered sd_narrow, binned by the SAME four generative-mu_narrow bins as
plot_mu_below_range_figure.py (not binned by sd itself) -- this is what
tests the mu/width trade-off directly: is width recovery worse for voxels
whose preferred numerosity lies outside the presented range? No n=
annotations in-panel (see the printed console table instead); no median-AE
panel here (unlike the mu figure) -- the rho panel already carries the
mu/width trade-off story on its own.

Data: width-recovery simulation at noise=0.8 (800 files, 194,212 simulated
voxels total). Overall pooled rho(gen_sd_narrow, est_sd_narrow) = 0.60 --
this is the r(sd)~0.6 figure referenced earlier for the rebuttal (see
recovery_bias_sd.ipynb cell 10, which computes the same correlation at
noise=0.5 rather than the empirically-calibrated noise=0.8 used here).

Writes figS_sd_below_range.pdf/.png to <bids>/derivatives/figures/.
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

MU_BINS = [-np.inf, 5, 10, 25, np.inf]
MU_BIN_LABELS = ['< 5', '5–10', '10–25', '> 25']
BIN_COLORS = [COL_OUT, COL_OUT, COL_IN, COL_OUT]
DESIGN_MAP = {'Full': 'Full design (10–25 and 10–40)', 'Censored': 'Censored design (10–25 only)'}
SD_LIM = 2.5


def load():
    files = list(bids_folder.glob('simulated_recovery_sd/*/noise_0.8/design_*/iteration-*_pervoxel.csv'))
    df = pd.concat([pd.read_csv(f).assign(
        design=f.parent.name.replace('design_', '').replace('_subjectwise', ''))
        for f in files], ignore_index=True)
    df['design'] = df['design'].str.capitalize().map(DESIGN_MAP)
    df['mu_bin'] = pd.cut(df['gen_mu_narrow'], bins=MU_BINS, labels=MU_BIN_LABELS)
    return df


def compute_table(df):
    rows = []
    for label in MU_BIN_LABELS:
        sub = df[df['mu_bin'] == label]
        rho, _ = stats.spearmanr(sub['gen_sd_narrow'], sub['est_sd_narrow'])
        abs_err = (sub['est_sd_narrow'] - sub['gen_sd_narrow']).abs()
        err = sub['est_sd_narrow'] - sub['gen_sd_narrow']
        rows.append(dict(bin=label, n=len(sub), rho=rho,
                         mae_median=abs_err.median(), bias=err.median(),
                         mae_mean=abs_err.mean(), mae_sem=abs_err.std(ddof=1) / np.sqrt(len(sub))))
    table = pd.DataFrame(rows).set_index('bin').loc[MU_BIN_LABELS]
    print(table.to_string(float_format=lambda x: f'{x:.3f}'))
    rho_all, _ = stats.spearmanr(df['gen_sd_narrow'], df['est_sd_narrow'])
    print(f'overall pooled rho (all mu): {rho_all:.3f} (n={len(df)})')
    return table


def compute_range_split(df):
    """Precise Spearman rho for generative-vs-recovered sd_narrow, split by
    whether the voxel's generative mu_narrow falls inside the presented
    stimulus range [10, 25] or outside it (< 10 or > 25) -- the two-way
    split referenced in notes/response_sd_recovery.md's 'r ~ 0.6' sentence."""
    in_range = df[(df['gen_mu_narrow'] >= 10) & (df['gen_mu_narrow'] <= 25)]
    out_range = df[(df['gen_mu_narrow'] < 10) | (df['gen_mu_narrow'] > 25)]

    print()
    for label, sub in [('overall (pooled)', df), ('in-range (10<=mu<=25)', in_range),
                       ('out-of-range (mu<10 or mu>25)', out_range)]:
        rho, p = stats.spearmanr(sub['gen_sd_narrow'], sub['est_sd_narrow'])
        print(f'{label}: rho = {rho:.4f}, p = {p:.3g}, n = {len(sub)}')


def plot_joint(ax, fig, sub, title, show_cbar):
    hb = ax.hexbin(sub['gen_sd_narrow'], sub['est_sd_narrow'], gridsize=60,
                   extent=(0, SD_LIM, 0, SD_LIM), linewidths=0., cmap='Blues',
                   mincnt=1, edgecolors='none', norm=mpl.colors.PowerNorm(0.4, vmin=0))

    ax.axline((0, 0), slope=1, color='0.35', ls='--', lw=.7, zorder=100)

    if show_cbar:
        cbar_ax = ax.inset_axes([0.94, 0.06, .03, .22])
        cb = fig.colorbar(hb, cax=cbar_ax, label='Voxel count', ticks=[1, int(hb.get_array().max())])
        cb.ax.yaxis.set_label_position('left')
        cb.ax.tick_params(length=0, labelsize=6.5)
        cb.ax.yaxis.label.set_size(6.5)
        cb.outline.set_visible(False)

    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(0, SD_LIM)
    ax.set_ylim(0, SD_LIM)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_aspect('equal')
    ax.set_xlabel('Generative σ (narrow,\nlog-numerosity units)')


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
    ax.set_xlabel('Generative μ (narrow,\nboth designs pooled)')
    return headroom


def plot_rho_points(ax, table):
    """Spearman rho between generative and recovered sd_narrow, per
    generative mu_narrow bin (the mu/width trade-off). No error bars: each
    value is a single correlation over pooled voxels, not a mean over
    independent replicates."""
    _bin_point_plot(ax, table, 'rho', 'Spearman ρ\n(generative vs. recovered σ)')
    ax.axhline(0, color='0.7', lw=0.6, zorder=0)
    ax.set_ylim(-0.3, 1.05)


def main():
    df = load()
    print(f'total voxels: {len(df)}')
    table = compute_table(df)
    compute_range_split(df)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9),
                             gridspec_kw=dict(width_ratios=[1., 1., 0.95]),
                             constrained_layout=True)
    plot_joint(axes[0], fig, df[df['design'] == DESIGN_MAP['Full']], 'Full design (10–25 and 10–40)', show_cbar=False)
    plot_joint(axes[1], fig, df[df['design'] == DESIGN_MAP['Censored']], 'Censored design (10–25 only)', show_cbar=True)
    plot_rho_points(axes[2], table)
    axes[0].set_ylabel('Recovered σ (narrow,\nlog-numerosity units)')

    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.2, 1.08, letter, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='bottom', ha='right')

    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=True)

    stem = bids_folder / 'derivatives' / 'figures' / 'figS_sd_below_range'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png', dpi=300)
    print('saved', stem)


if __name__ == '__main__':
    main()
