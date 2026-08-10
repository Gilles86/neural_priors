"""Supplementary figure: retinotopic (bar-position) account vs efficient coding.

Reviewer 4, point (3). Because the response bar spans a fixed visual angle,
purely retinotopic tuning to the bar predicts an *additive* shift of
preferred numerosity between conditions (mu_wide = mu_narrow + 7.5, the
offset between the two range centers), whereas efficient-coding range
adaptation predicts a *multiplicative* rescaling
(mu_wide = 2*(mu_narrow - 10) + 10 above the shared lower bound).

Panel a: joint distribution of preferred numerosities in the narrow vs wide
condition, with both predictions overlaid. Uses the model-3 fits (per-voxel
*free* shift, NPCr, cvR2 > 0), so the wide-condition preferred numerosities
are estimated independently of either prediction (model 15 fixes the shift
at the efficient-coding value and cannot arbitrate).
Panel b: marginal distributions of preferred numerosity per condition; under
retinotopic (bar-position) tuning these should be centered on the range
centers (17.5 and 25, dashed lines), which they are not.

Writes retinotopic_control.pdf/.png to <bids>/derivatives/figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bids_folder = Path('/data/ds-neuralpriors')

pars = pd.read_csv(bids_folder / 'derivatives' / 'encoding_models' / 'group_roi-NPCr_desc-groundtruth_parameters.tsv',
                   sep='\t', index_col=[0, 1, 2, 3], header=[0, 1])
pars = pars.xs(3, level='model_label')
pars = pars[pars[('cvr2', 'nan')] > 0.0]

mu_n, mu_w = pars[('mu', 'narrow')], pars[('mu', 'wide')]

pred_efficient = np.where(mu_n < 10, mu_n, (mu_n - 10) * 2 + 10)
pred_retinotopic = mu_n + 7.5
ae_eff = np.abs(mu_w - pred_efficient)
ae_ret = np.abs(mu_w - pred_retinotopic)
print(f'median abs. error efficient: {ae_eff.median():.2f}, retinotopic: {ae_ret.median():.2f}; '
      f'P(voxel closer to efficient) = {(ae_eff < ae_ret).mean():.3f} (n={len(pars)})')

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

# Panel a: joint distribution + model predictions
ax = axes[0]
h = ax.hist2d(mu_n, mu_w, bins=[np.linspace(0, 40, 100)] * 2, cmap='Blues', norm='log')
ax.plot([0, 10, 25], [0, 10, 40], c='r', label='Efficient coding')
ax.plot([0, 25], [7.5, 32.5], c='g', label='Retinotopic (bar position)')
ax.set_xlim(0, 26)
ax.set_xlabel('Preferred numerosity, narrow condition')
ax.set_ylabel('Preferred numerosity, wide condition')
ax.legend(frameon=False, loc='upper left')

# Panel b: marginal distributions vs range centers
ax = axes[1]
palette = sns.color_palette()
for (label, mu, center), color in zip(
        [('Narrow (10-25)', mu_n, 17.5), ('Wide (10-40)', mu_w, 25.0)], palette):
    sns.histplot(x=mu, bins=np.arange(0, 41), stat='percent', element='step',
                 fill=False, color=color, label=label, ax=ax)
    ax.axvline(center, color=color, ls='--', lw=1)
ax.text(17.5, ax.get_ylim()[1] * 0.97, ' Range centers\n (retinotopic prediction)',
        fontsize=8, color='0.3', va='top')
ax.set_xlabel('Preferred numerosity')
ax.set_ylabel('Percent of voxels')
ax.legend(frameon=False)

for ax, letter in zip(axes, 'ab'):
    ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='right')

sns.despine(fig=fig)

stem = bids_folder / 'derivatives' / 'figures' / 'retinotopic_control'
fig.savefig(f'{stem}.pdf')
fig.savefig(f'{stem}.png', dpi=300)
print('saved', stem.name)
