"""Recovery figures for shift (delta_wide) and width scaling (sd_wide_scale).

Reproduces the exact layout of the original Supplementary Fig. 8
(`revision feb 2026/recovery_bias.ipynb`): seaborn FacetGrid, one column per
design, hue = generative value, density histograms, default palette.

By default renders the pooled-sampling simulations at noise SD 0.8 — the
level at which the simulated single-trial R^2 matches the empirical median
(see README.md) — plus the subject-wise variants. Writes PDF and PNG for
each figure to <bids>/derivatives/figures/.

`make_combined_figure()` merges the pooled and subject-wise sampling variants
into a single figure (row = sampling variant, col = design) and writes it
under the canonical `model_recovery_{param}.pdf`/`.png` names — this is the
version cited as Fig. 8 / Fig. S9.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

bids_folder = Path('/data/ds-neuralpriors')

SPECS = {
    'mu': dict(folder='simulated_recovery_mu', estcol='delta_wide',
               xlabel=r'Estimated $r_\mu$', bins=np.linspace(0, 3., 50),
               legend_labels={'1.00': '1.0', '2.00': '2.0'}, hue_order=['1.0', '2.0']),
    'sd': dict(folder='simulated_recovery_sd', estcol='sd_wide_scale',
               xlabel=r'Estimated $r_\sigma$', bins=np.linspace(0, 2., 50),
               legend_labels={'1.00': '1.0', '1.29': '1.29'}, hue_order=['1.0', '1.29']),
}


def load_results(param, noise, subjectwise):
    spec = SPECS[param]
    files = [f for f in bids_folder.glob(f'{spec["folder"]}/*/noise_{noise}/design_*/iteration-*_results.csv')
             if f.parent.name.endswith('_subjectwise') == subjectwise]

    pars = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pars['design'] = [f.parent.name.replace('design_', '').replace('_subjectwise', '') for f in files]
    pars['generative value'] = [f'{float(f.parent.parent.parent.name.split("_")[-1]):.2f}' for f in files]
    pars['generative value'] = pars['generative value'].map(spec['legend_labels'])
    pars['design'] = pars['design'].str.capitalize().map({'Full': 'Full (10-25 and 10-40)',
                                                          'Censored': 'Censored (All conditions 10-25)'})
    return pars


def _add_generative_lines(g, spec):
    """Dashed vertical line at each generative value, colour-matched to its hue."""
    colors = sns.color_palette(n_colors=len(spec['hue_order']))
    for ax in g.axes.flat:
        for label, color in zip(spec['hue_order'], colors):
            ax.axvline(float(label), color=color, ls='--', lw=1)


def make_figure(param, noise=0.8, subjectwise=False):
    spec = SPECS[param]
    pars = load_results(param, noise, subjectwise)

    g = sns.FacetGrid(pars, col='design', hue='generative value', hue_order=spec['hue_order'],
                       margin_titles=True)
    g.map(sns.histplot, spec['estcol'], stat='density', common_norm=False, bins=spec['bins'])
    _add_generative_lines(g, spec)
    g.add_legend()

    g.set_axis_labels(spec['xlabel'], 'Density')
    g.set_titles(col_template='{col_name} design')

    suffix = '_subjectwise' if subjectwise else ''
    stem = bids_folder / 'derivatives' / 'figures' / f'model_recovery_{param}{suffix}'
    g.savefig(f'{stem}.pdf')
    g.savefig(f'{stem}.png', dpi=300)
    print('saved', stem.name, f'(noise={noise})')


def make_combined_figure(param, noise=0.8):
    spec = SPECS[param]

    pooled = load_results(param, noise, subjectwise=False)
    pooled['Sampling'] = 'Pooled (250 voxels/iteration)'
    subjectwise = load_results(param, noise, subjectwise=True)
    subjectwise['Sampling'] = 'Subject-wise (1 subject/iteration)'
    pars = pd.concat([pooled, subjectwise], ignore_index=True)

    g = sns.FacetGrid(pars, row='Sampling', col='design', hue='generative value',
                       hue_order=spec['hue_order'], margin_titles=True,
                       gridspec_kws=dict(hspace=0.06))
    g.map(sns.histplot, spec['estcol'], stat='density', common_norm=False, bins=spec['bins'])
    _add_generative_lines(g, spec)
    g.add_legend()

    g.set_axis_labels(spec['xlabel'], 'Density')
    g.set_titles(col_template='{col_name} design', row_template='{row_name}')

    stem = bids_folder / 'derivatives' / 'figures' / f'model_recovery_{param}'
    g.savefig(f'{stem}.pdf')
    g.savefig(f'{stem}.png', dpi=300)
    print('saved', stem.name, f'(noise={noise}, combined)')


if __name__ == '__main__':
    for param in ['mu', 'sd']:
        for subjectwise in [False, True]:
            make_figure(param, noise=0.8, subjectwise=subjectwise)
        make_combined_figure(param, noise=0.8)
