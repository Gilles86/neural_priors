"""Per-voxel recovery of preferred numerosity (Reviewer 4, below-range nPRFs).

2D histogram of generative vs. recovered preferred numerosity (narrow
condition) across all simulated voxels of the width-recovery simulations at
the empirically matched noise level (0.8). Same seaborn template as the
other recovery figures. Dashed lines mark the lower bound of the presented
range (10); the dotted diagonal is the identity.

Writes model_recovery_mu_pervoxel.pdf/.png to <bids>/derivatives/figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bids_folder = Path('/data/ds-neuralpriors')

files = list(bids_folder.glob('simulated_recovery_sd/*/noise_0.8/design_*/iteration-*_pervoxel.csv'))
df = pd.concat([pd.read_csv(f).assign(design=f.parent.name.replace('design_', '').replace('_subjectwise', ''))
                for f in files], ignore_index=True)
df['design'] = df['design'].str.capitalize().map({'Full': 'Full (10-25 and 10-40)',
                                                  'Censored': 'Censored (All conditions 10-25)'})

g = sns.FacetGrid(df, col='design', margin_titles=True, height=3.2, aspect=1.0)
g.map_dataframe(sns.histplot, x='gen_mu_narrow', y='est_mu_narrow',
                bins=(np.linspace(0, 40, 80), np.linspace(0, 40, 80)), cbar=False)

for ax in g.axes.ravel():
    ax.plot([0, 40], [0, 40], color='k', ls=':', lw=1)
    ax.axvline(10, color='0.4', ls='--', lw=0.8)
    ax.axhline(10, color='0.4', ls='--', lw=0.8)
    ax.set(xlim=(0, 40), ylim=(0, 40))
    ax.set_xticks([0, 10, 25, 40])
    ax.set_yticks([0, 10, 25, 40])

g.set_axis_labels('Generative preferred numerosity', 'Recovered preferred numerosity')
g.set_titles(col_template='{col_name} design')

stem = bids_folder / 'derivatives' / 'figures' / 'model_recovery_mu_pervoxel'
g.savefig(f'{stem}.pdf')
g.savefig(f'{stem}.png', dpi=300)
print('saved', stem.name)
