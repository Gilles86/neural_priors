"""Recovery figure for the width-scaling parameter (sd_wide_scale), matched to SFig 8.

Produces the same figure layout as `revision feb 2026/recovery_bias.ipynb`
(current Supplementary Fig. 8, recovery of the shift parameter delta_wide):
seaborn FacetGrid, one column per design, hue = generative value, density
histograms.

Two files are written to <bids>/derivatives/figures/:
  model_recovery_sd.pdf              pooled sampling (250 voxels across
                                     subjects per iteration - the exact
                                     procedural analog of SFig 8)
  model_recovery_sd_subjectwise.pdf  subject-wise sampling (one random
                                     subject per iteration, all of their
                                     supra-threshold voxels)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

bids_folder = Path('/data/ds-neuralpriors')


def load_results(subjectwise):
    suffix = '_subjectwise' if subjectwise else ''
    files = [f for f in bids_folder.glob('simulated_recovery_sd/sd_scale_*/noise_0.5/design_*/iteration-*_results.csv')
             if f.parent.name.endswith('_subjectwise') == subjectwise]

    pars = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pars['design'] = [f.parent.name.replace('design_', '').replace('_subjectwise', '') for f in files]
    pars['gen_sd_wide_scale'] = [f'{float(f.parent.parent.parent.name.split("_")[-1]):.2f}' for f in files]
    pars['design'] = pars['design'].str.capitalize().map({'Full': 'Full (10-25 and 10-40)',
                                                          'Censored': 'Censored (All conditions 10-25)'})
    return pars, suffix


def make_figure(subjectwise=False):
    pars, suffix = load_results(subjectwise)

    g = sns.FacetGrid(pars, col='design', hue='gen_sd_wide_scale', margin_titles=True)
    g.map(sns.histplot, 'sd_wide_scale', stat='density', common_norm=False, bins=np.linspace(0, 2., 50))
    g.add_legend()

    g.set_axis_labels('Estimated sd_wide_scale', 'Density')
    g.set_titles(col_template='{col_name} design')

    g.savefig(bids_folder / 'derivatives' / 'figures' / f'model_recovery_sd{suffix}.pdf')
    print('saved', f'model_recovery_sd{suffix}.pdf')


if __name__ == '__main__':
    make_figure(subjectwise=False)
    make_figure(subjectwise=True)
