#!/usr/bin/env python3
"""Average numerosity tuning — the numerosity twin of value_prf's
`average_tuning_curve_ds{1,2}.pdf`, same design and same test.

Every well-fitting voxel is lined up on its own preferred numerosity and
averaged, on the normalised axis x = (numerosity - mu)/sigma, so voxels with
different preferred numerosities and widths can be averaged at all.

As in the value figure, BOTH alignments are shown, because the trap is the same:

  in-sample     mu, sigma and the voxel selection come from the trials being
                averaged -- a peak is then guaranteed, even from pure noise
  out-of-sample they come from the OTHER half of the runs -- the real curve

Reads `numerosity_normalized_binned.tsv`, which `run_analysis.py` already wrote
(per subject, per x-bin), so nothing is refitted here.  The value version reads
a group-level file; this one aggregates across subjects itself.

Run:
  ~/mambaforge/envs/braincoder/bin/python \\
      -m neural_priors.value_comparison.plot_average_tuning_numerosity
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from value_prf.visualize import set_style, panel_letter

OUT_FIG = Path.home() / 'git/value_prf/notes/figures'
OUT_DAT = Path.home() / 'git/value_prf/notes/data'

ROI_C = {'NPCr': '#5D8C3F', 'NPCl': '#8FBF6A'}
MODEL_LABEL = {'prf': 'Single-peak PRF\n4 parameters',
               'wprf': 'Weighted PRF\n11 basis + intercept'}
MODEL_ORDER = ['prf', 'wprf']
ALIGN_TITLE = {'circular': 'IN-SAMPLE', 'heldout': 'OUT-OF-SAMPLE'}
ALIGN_SUB = {
    'circular': 'μ, σ and voxel selection from the SAME trials\n'
                'A peak here is guaranteed — this is the control',
    'heldout': 'μ, σ and voxel selection from the OTHER half of the runs\n'
               'This is the real average tuning curve',
}
ALIGN_C = {'circular': '#8C2E2E', 'heldout': '#1F4E79'}
MIN_FRAC_VOX = 0.25       # hide bins where few voxels contribute (as in value)


def centre_flank(g):
    """Per subject: mean in |x|<0.5 minus mean in 1.5<|x|<3 — the same
    statistic the value figure tests."""
    c = g[np.abs(g.x_norm) < 0.5]['measured_z'].mean()
    f = g[(np.abs(g.x_norm) > 1.5) & (np.abs(g.x_norm) < 3)]['measured_z'].mean()
    return c - f


def main(sample='full', rois=('NPCr', 'NPCl'), out_fig=OUT_FIG,
         out_dat=OUT_DAT):
    set_style()
    d = pd.read_csv(Path(out_dat) / 'numerosity_normalized_binned.tsv', sep='\t')
    d = d[(d['sample'] == sample) & d.roi.isin(rois)]
    # same bin-coverage guard as the value figure: only x-bins where a decent
    # fraction of the selected voxels actually contribute
    d = d[d.frac_voxels >= MIN_FRAC_VOX]
    n_sub = d.subject.nunique()

    models = [m for m in MODEL_ORDER if m in set(d.model)]
    fig, allax = plt.subplots(2, len(models) + 1,
                              figsize=(2.55 * len(models) + 2.3, 4.8),
                              constrained_layout=True, sharex='col',
                              width_ratios=[0.20] + [1] * len(models))
    allax = np.atleast_2d(allax)
    for a in allax[:, 0]:
        a.set_axis_off()
    axes = allax[:, 1:]

    tidy, stat_rows = [], []
    for row, align in enumerate(('circular', 'heldout')):
        for col, mdl in enumerate(models):
            ax = axes[row, col]
            sub = d[(d.alignment == align) & (d.model == mdl)]
            for roi, g in sub.groupby('roi'):
                agg = (g.groupby('x_norm')['measured_z']
                       .agg(m='mean', se='sem', n='size').reset_index()
                       .sort_values('x_norm'))
                c = ROI_C[roi]
                ax.fill_between(agg.x_norm, agg.m - agg.se, agg.m + agg.se,
                                color=c, alpha=0.25, lw=0, zorder=2)
                ax.plot(agg.x_norm, agg.m, color=c, lw=1.8, zorder=3)
                lx = {'NPCr': 1.5, 'NPCl': -1.5}[roi]
                ly = float(np.interp(lx, agg.x_norm, agg.m))
                ax.text(lx + (0.2 if roi == 'NPCr' else -0.2), ly, roi, color=c,
                        fontsize=7, fontweight='bold',
                        ha='left' if roi == 'NPCr' else 'right', va='bottom')
                tidy += [dict(dataset='neural_priors', sample=sample,
                              alignment=align, model=mdl, roi=roi,
                              x_norm=r.x_norm, measured_mean_z=r.m,
                              measured_sem_z=r.se, n_subjects=int(r.n))
                         for r in agg.itertuples()]
            # the same shaded regions that define the statistic
            ax.axvspan(-0.5, 0.5, color='0.55', alpha=0.13, lw=0, zorder=0)
            for lo_x, hi_x in ((-3, -1.5), (1.5, 3)):
                ax.axvspan(lo_x, hi_x, color='0.55', alpha=0.06, lw=0, zorder=0)
            ax.axvline(0, color='0.6', lw=0.7, ls='--', zorder=1)
            ax.axhline(0, color='0.6', lw=0.7, ls='--', zorder=1)
            ax.set(xticks=[-4, -2, 0, 2, 4], xlim=(-4, 4))
            if row == 1:
                ax.set_xlabel('(Numerosity − μ) / σ')
            if col == 0:
                ax.set_ylabel('Average response (z)')
            if row == 0:
                ax.set_title(MODEL_LABEL[mdl], fontsize=8)

            lines = []
            for roi, g in sub.groupby('roi'):
                v = np.array([centre_flank(gg) for _, gg
                              in g.groupby('subject')])
                v = v[np.isfinite(v)]
                if len(v) < 3:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                lines.append(f'{roi}: {v.mean():+.3f} z, t({len(v)-1})={t:.2f}, '
                             f'p={p:.2g}')
                stat_rows.append(dict(dataset='neural_priors', sample=sample,
                                      alignment=align, model=mdl, roi=roi,
                                      centre_minus_flank=float(v.mean()),
                                      sem=float(stats.sem(v)), t=float(t),
                                      p=float(p), n_subjects=len(v)))
            ax.text(0.02, 0.97, '\n'.join(lines), transform=ax.transAxes,
                    fontsize=6.2, color='0.25', ha='left', va='top',
                    linespacing=1.35)
            panel_letter(ax, 'abcd'[row * len(models) + col])

    for r in range(2):
        lo = min(a.get_ylim()[0] for a in axes[r])
        hi = max(a.get_ylim()[1] for a in axes[r])
        for a in axes[r]:
            a.set_ylim(lo, hi)

    amp = {}
    for a in ('circular', 'heldout'):
        s = d[(d.alignment == a) & (d.model == 'prf')]
        amp[a] = s.groupby(['roi', 'x_norm'])['measured_z'].mean().max()
    extra = {'circular': '',
             'heldout': f'\nPeak is ~{amp["circular"] / amp["heldout"]:.0f}× '
                        f'smaller than the row above'}
    for row, align in enumerate(('circular', 'heldout')):
        a = allax[row, 0]
        colr = ALIGN_C[align]
        a.text(0.30, 0.5, ALIGN_TITLE[align], transform=a.transAxes,
               rotation=90, ha='center', va='center', fontsize=12,
               fontweight='bold', color=colr, clip_on=False)
        a.text(0.80, 0.5, ALIGN_SUB[align] + extra[align],
               transform=a.transAxes, rotation=90, ha='center', va='center',
               fontsize=6.3, color=colr, linespacing=1.4, clip_on=False)
        a.plot([0.02, 0.02], [0.02, 0.98], transform=a.transAxes, color=colr,
               lw=2.0, clip_on=False)

    n_tr = {'full': 480, 'matched128': 128}[sample]
    fig.suptitle(f'Average numerosity tuning, each voxel aligned on its own '
                 f'preferred numerosity  ·  {n_sub} subjects, {n_tr} trials',
                 fontsize=10, fontweight='bold')
    fig.text(0.5, -0.05,
             'TEST — for each subject: mean response in the dark shaded centre '
             '(|x| < 0.5σ) minus the mean in the pale shaded flanks '
             '(1.5σ < |x| < 3σ),\nthen a one-sample t-test of those per-subject '
             'differences against zero. Identical statistic, models and '
             'selection to the value figures — only the stimulus differs.',
             ha='center', va='top', fontsize=6.6, color='0.25', linespacing=1.4)
    sns.despine(fig=fig, offset=4, trim=False)
    stem = Path(out_fig) / f'average_tuning_curve_numerosity_{sample}'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', bbox_inches='tight', pad_inches=0.02,
                    dpi=300 if ext == 'png' else None)
    print(f'saved {stem}.pdf / .png')
    pd.DataFrame(tidy).to_csv(
        Path(out_dat) / f'data_average_tuning_numerosity_{sample}.tsv',
        sep='\t', index=False)
    sdf = pd.DataFrame(stat_rows)
    sdf.to_csv(Path(out_dat) /
               f'data_average_tuning_numerosity_stats_{sample}.tsv',
               sep='\t', index=False)
    print(sdf[sdf.alignment == 'heldout'].round(4).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', default='full', choices=['full', 'matched128'])
    a = p.parse_args()
    main(sample=a.sample)
