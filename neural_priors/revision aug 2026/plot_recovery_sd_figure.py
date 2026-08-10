"""Publication figure for the width-scaling (sd_wide_scale) recovery simulation.

Panels:
  a  Recovered shared width-scaling factor, full design (subject-wise sampling)
  b  Same, censored design
  c  Recovery error as a function of the number of simulated voxels

Writes figS_recovery_sd.pdf/.svg to <bids>/derivatives/figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
sns.set_context('paper')

COLOR_NULL = '#7F7F7F'    # generative scale 1 (no widening)
COLOR_WIDE = '#3B5BA5'    # generative scale 1.29 (empirical)

bids_folder = Path('/data/ds-neuralpriors')

files = list(bids_folder.glob('simulated_recovery_sd/sd_scale_*/noise_0.5/design_*_subjectwise/iteration-*_results.csv'))
df = pd.concat([pd.read_csv(f).assign(design=f.parent.name.replace('design_', '').replace('_subjectwise', ''),
                                      gen=float(f.parent.parent.parent.name.split('_')[-1]))
                for f in files], ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.4), constrained_layout=True)

bins = np.linspace(0.85, 1.45, 41)
design_titles = {'full': 'Full design (wide range 10–40)',
                 'censored': 'Censored design (wide range 10–25)'}

for ax, design in zip(axes[:2], ['full', 'censored']):
    d = df[df.design == design]
    for gen, color in [(1.0, COLOR_NULL), (1.287794, COLOR_WIDE)]:
        est = d.loc[d.gen == gen, 'sd_wide_scale']
        ax.hist(est, bins=bins, density=True, color=color, alpha=0.75, edgecolor='none')
        ax.axvline(gen, color=color, ls='--', lw=0.8, zorder=0)
    ax.set_xlabel('Recovered width scaling')
    ax.set_xticks([0.9, 1.0, 1.1, 1.2, 1.29, 1.4])
    ax.set_xticklabels(['0.9', '1', '1.1', '1.2', '1.29', '1.4'])
    ax.text(0.0, 1.02, design_titles[design], transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, color='0.25')

axes[0].set_ylabel('Density')
axes[1].set_ylabel('')
axes[1].set_yticklabels([])
ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
for ax in axes[:2]:
    ax.set_ylim(0, ymax * 1.28)

# Direct labels under the panel heading, in the data's colors
axes[0].text(1.0, ymax * 1.06, 'Generative: 1\n(no widening)', color=COLOR_NULL,
             ha='center', va='top', fontsize=7.5)
axes[0].text(1.29, ymax * 1.06, 'Generative: 1.29\n(empirical)', color=COLOR_WIDE,
             ha='center', va='top', fontsize=7.5)

# Panel c: error vs population size (both designs pooled)
ax = axes[2]
df['error'] = df['sd_wide_scale'] - df['gen']
for gen, color in [(1.0, COLOR_NULL), (1.287794, COLOR_WIDE)]:
    d = df[df.gen == gen]
    ax.scatter(d.n_voxels, d.error, s=7, color=color, alpha=0.4, edgecolors='none')
ax.axhline(0, color='0.7', ls='--', lw=0.6, zorder=0)
ax.set_xlabel('Simulated voxels per subject')
ax.set_ylabel('Recovery error')
ax.set_xticks([0, 250, 500])
ax.set_yticks([-0.1, 0, 0.1])
ax.text(0.02, 1.02, 'One dot per simulated subject', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=8, color='0.25')

for ax, letter in zip(axes, 'abc'):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', ha='right')

sns.despine(fig=fig, offset=5, trim=True)

out = bids_folder / 'derivatives' / 'figures'
fig.savefig(out / 'figS_recovery_sd.pdf')
fig.savefig(out / 'figS_recovery_sd.svg')
print('saved to', out / 'figS_recovery_sd.pdf')
