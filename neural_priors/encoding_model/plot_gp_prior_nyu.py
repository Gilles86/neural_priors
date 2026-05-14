"""NYU vision-science style figures for the GP-prior pRF decoding study.

Three focused figures:

  Fig 1 — headline: decoding Pearson r per (pipeline × range × method ×
          omega), with the paper baseline annotated.
  Fig 2 — supporting: median absolute decoding error in the same layout.
  Fig 3 — within-subject paired Δ contrasts (the "what's driving the gain").

Follows the house style for NYU vision figures: despine + offset, outward
ticks, Helvetica, direct labels, no legend frame, in-panel annotations,
no grid. One physical size, set at figure creation, not rescaled.

Usage:
    python plot_gp_prior_nyu.py /shares/zne.uzh/gdehol/ds-neuralpriors \\
        --out ~/figs_nyu
"""

import argparse
import os
import os.path as op
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PIPELINES = {
    'unsmoothed': 'gp_prior_roi-NPCr',
    'smoothed':   'gp_prior_roi-NPCr.smoothed',
}
METHOD_ORDER = ['classical', 'ml', 'bayes']
METHOD_NICE = {'classical': 'Classical', 'ml': 'ML', 'bayes': 'Bayes'}
OMEGA_ORDER = ['plain', 'distance']
PIPELINE_ORDER = ['unsmoothed', 'smoothed']
PIPELINE_NICE = {'unsmoothed': 'Unsmoothed BOLD',
                 'smoothed':   'Smoothed BOLD'}
RANGES = ['narrow', 'wide']
RANGE_NICE = {'narrow': 'Narrow range', 'wide': 'Wide range'}

# Paper baselines (analyze_decoding.ipynb, model 15, top-100 voxels).
PAPER_R = {'narrow': 0.082, 'wide': 0.136}
PAPER_MAE = {'narrow': 3.90, 'wide': 6.33}

# Muted, hand-picked palette: gray for plain Ω, deep blue for distance Ω.
OMEGA_COLOR = {'plain': '#9C9C9C', 'distance': '#3B5BA5'}


# ----------------------------------------------------------------- rcParams

def apply_nyu_style():
    """House style: Helvetica, despine, outward ticks, editable fonts."""
    mpl.rcParams.update({
        # Typography
        'font.family':     'sans-serif',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue',
                             'TeX Gyre Heros', 'Arial', 'DejaVu Sans'],
        'font.size':       9,
        'axes.labelsize':  10,
        'axes.titlesize':  10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'mathtext.fontset': 'stixsans',

        # Axes
        'axes.linewidth':   0.8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.labelpad':    4,

        # Ticks: outward, short, thin
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,

        # Lines / markers
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,

        # Legend
        'legend.frameon': False,
        'legend.handlelength': 1.5,

        # Vector output, editable text
        'pdf.fonttype': 42,
        'ps.fonttype':  42,
        'svg.fonttype': 'none',

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })
    sns.set_context('paper')

    # Helvetica availability — fall back transparently if not present.
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        wanted = ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros']
        if not any(w in installed for w in wanted):
            warnings.warn(
                f'Helvetica not found; falling back to: '
                f'{mpl.rcParams["font.sans-serif"][3:]}', stacklevel=2)
    except Exception:
        pass


# --------------------------------------------------------------------- load

def _concat_tsv(bids_folder, pattern):
    frames = []
    range_re = re.compile(r'_range-(narrow|wide)_')
    for pipeline, subdir in PIPELINES.items():
        root = Path(bids_folder) / 'derivatives' / 'encoding_models' / subdir
        for path in sorted(root.glob(pattern)):
            subj = path.name.split('_')[0].replace('sub-', '')
            df = pd.read_csv(path, sep='\t')
            df.insert(0, 'subject', subj)
            df.insert(1, 'pipeline', pipeline)
            if 'stim_range' not in df.columns:
                m = range_re.search(path.name)
                df['stim_range'] = m.group(1) if m else 'unknown'
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_decoding(bids_folder):
    df = _concat_tsv(bids_folder, 'sub-*/func/sub-*_desc-decoding.tsv')
    if df.empty:
        return df
    if 'omega' not in df.columns:
        df['omega'] = 'plain'
    return df


# ----------------------------------------------------------------- figures

def _per_subject(decoding, value):
    """Mean-across-folds per (subject, pipeline, range, method, omega)."""
    return (decoding
            .groupby(['subject', 'pipeline', 'stim_range',
                      'method', 'omega'])[value]
            .mean()
            .reset_index())


def _empty_panel(ax, msg):
    # Hide ticks via tick_params (per-axis visibility), NOT set_xticks([]),
    # because sharex=True makes set_xticks modify the locator on every panel.
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=8, color='0.5')
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_headline_r(decoding, out_path):
    """The headline: decoded-vs-true Pearson r per (pipeline × range × method × omega).

    2×2 panel grid (pipelines × ranges), each panel a strip plot of
    per-subject mean r split by method and omega. Horizontal dashed
    line at the paper baseline. Narrow text annotation calls out
    'Paper baseline (model 15)' on the top-left panel only.
    """
    per_subj = _per_subject(decoding, 'r')
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0),
                              sharex=True, sharey='row',
                              constrained_layout=True)
    for r_idx, pipeline in enumerate(PIPELINE_ORDER):
        for c_idx, rng in enumerate(RANGES):
            ax = axes[r_idx, c_idx]
            df = per_subj[(per_subj['pipeline'] == pipeline)
                           & (per_subj['stim_range'] == rng)]
            if df.empty:
                _empty_panel(ax, 'No data yet')
                continue

            sns.stripplot(
                data=df, x='method', y='r', hue='omega',
                order=METHOD_ORDER, hue_order=OMEGA_ORDER,
                palette=OMEGA_COLOR, dodge=True, jitter=0.18,
                size=4, alpha=0.7, edgecolor='none', ax=ax)
            # Median bar per (method, omega).
            sns.pointplot(
                data=df, x='method', y='r', hue='omega',
                order=METHOD_ORDER, hue_order=OMEGA_ORDER,
                palette={'plain': '#5c5c5c', 'distance': '#1f3a73'},
                dodge=0.4, join=False, markers='_', scale=1.5,
                errorbar=None, ax=ax)
            if ax.get_legend():
                ax.get_legend().remove()

            ax.axhline(0, color='0.7', lw=0.6, zorder=0)
            paper = PAPER_R.get(rng)
            if paper is not None:
                ax.axhline(paper, color='0.4', ls='--', lw=0.8, zorder=0)
                if r_idx == 0 and c_idx == 0:
                    ax.annotate(
                        'Paper baseline\n(narrow)',
                        xy=(2.4, paper),
                        xytext=(2.6, paper - 0.10),
                        ha='left', va='top', fontsize=7, color='0.4',
                        arrowprops=dict(arrowstyle='-',
                                         connectionstyle='arc3,rad=0.2',
                                         color='0.4', lw=0.5))

            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels([METHOD_NICE[m] for m in METHOD_ORDER])
            ax.set_xlabel('')
            if c_idx == 0:
                ax.set_ylabel(f'{PIPELINE_NICE[pipeline]}\nMean within-fold r')
            else:
                ax.set_ylabel('')
            if r_idx == 0:
                ax.set_title(RANGE_NICE[rng], pad=4)

    # One direct-label legend in the bottom-right panel.
    legend_ax = axes[-1, -1]
    legend_ax.plot([], [], 'o', color=OMEGA_COLOR['plain'],
                    markersize=4, alpha=0.7, label='Free-form Ω')
    legend_ax.plot([], [], 'o', color=OMEGA_COLOR['distance'],
                    markersize=4, alpha=0.7,
                    label='Distance-modulated Ω')
    legend_ax.legend(loc='lower right', frameon=False, fontsize=7,
                      handletextpad=0.3, borderaxespad=0.2)

    sns.despine(fig=fig, offset=5, trim=True)
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300,
                 bbox_inches='tight', pad_inches=0.02)
    print(f'  wrote {out_path}')
    return fig


def fig_medae(decoding, out_path):
    """Supporting: median absolute decoding error, same layout."""
    per_subj = _per_subject(decoding, 'median_ae')
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0),
                              sharex=True, sharey='row',
                              constrained_layout=True)
    for r_idx, pipeline in enumerate(PIPELINE_ORDER):
        for c_idx, rng in enumerate(RANGES):
            ax = axes[r_idx, c_idx]
            df = per_subj[(per_subj['pipeline'] == pipeline)
                           & (per_subj['stim_range'] == rng)]
            if df.empty:
                _empty_panel(ax, 'No data yet'); continue

            sns.stripplot(
                data=df, x='method', y='median_ae', hue='omega',
                order=METHOD_ORDER, hue_order=OMEGA_ORDER,
                palette=OMEGA_COLOR, dodge=True, jitter=0.18,
                size=4, alpha=0.7, edgecolor='none', ax=ax)
            sns.pointplot(
                data=df, x='method', y='median_ae', hue='omega',
                order=METHOD_ORDER, hue_order=OMEGA_ORDER,
                palette={'plain': '#5c5c5c', 'distance': '#1f3a73'},
                dodge=0.4, join=False, markers='_', scale=1.5,
                errorbar=None, ax=ax)
            if ax.get_legend():
                ax.get_legend().remove()

            paper = PAPER_MAE.get(rng)
            if paper is not None:
                ax.axhline(paper, color='0.4', ls='--', lw=0.8, zorder=0)

            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels([METHOD_NICE[m] for m in METHOD_ORDER])
            ax.set_xlabel('')
            if c_idx == 0:
                ax.set_ylabel(f'{PIPELINE_NICE[pipeline]}\n'
                              f'Median |decoded − true|')
            else:
                ax.set_ylabel('')
            if r_idx == 0:
                ax.set_title(RANGE_NICE[rng], pad=4)

    sns.despine(fig=fig, offset=5, trim=True)
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300,
                 bbox_inches='tight', pad_inches=0.02)
    print(f'  wrote {out_path}')
    return fig


def fig_contrasts(decoding, out_path):
    """Within-subject Δ r contrasts: the four key questions."""
    per_subj = _per_subject(decoding, 'r')
    wide = per_subj.pivot_table(
        index=['subject', 'stim_range'],
        columns=['pipeline', 'method', 'omega'],
        values='r').reset_index()

    cols = set(wide.columns)
    def _v(row, p, m, o):
        if (p, m, o) not in cols:
            return np.nan
        v = row[(p, m, o)]
        return float(v) if np.isscalar(v) else float(np.asarray(v).ravel()[0])

    contrasts = [
        ('Dist Ω − Plain Ω  (Sm., Bayes)',
         lambda r: _v(r, 'smoothed', 'bayes', 'distance')
                    - _v(r, 'smoothed', 'bayes', 'plain')),
        ('Dist Ω − Plain Ω  (Unsm., Bayes)',
         lambda r: _v(r, 'unsmoothed', 'bayes', 'distance')
                    - _v(r, 'unsmoothed', 'bayes', 'plain')),
        ('Bayes − Classical  (Unsm., Dist Ω)',
         lambda r: _v(r, 'unsmoothed', 'bayes', 'distance')
                    - _v(r, 'unsmoothed', 'classical', 'distance')),
        ('Bayes(Unsm.) − Classical(Sm.)  (Dist Ω)',
         lambda r: _v(r, 'unsmoothed', 'bayes', 'distance')
                    - _v(r, 'smoothed', 'classical', 'distance')),
    ]

    rows = []
    for _, row in wide.iterrows():
        subj = row[('subject', '', '')] if ('subject', '', '') in cols else row['subject']
        rng = row[('stim_range', '', '')] if ('stim_range', '', '') in cols else row['stim_range']
        for name, fn in contrasts:
            rows.append(dict(subject=subj, stim_range=rng,
                              contrast=name, delta=fn(row)))
    df = pd.DataFrame(rows).dropna(subset=['delta'])
    if df.empty:
        print(f'  skipping {out_path}: no contrast data')
        return None

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4),
                              sharey=True, constrained_layout=True)
    for ax, rng in zip(axes, RANGES):
        sub = df[df['stim_range'] == rng]
        if sub.empty:
            _empty_panel(ax, f'No {rng} data yet')
            continue
        sns.stripplot(
            data=sub, x='contrast', y='delta',
            order=[c for c, _ in contrasts],
            color='#3B5BA5', size=5, alpha=0.7, edgecolor='none',
            jitter=0.18, ax=ax)
        sns.pointplot(
            data=sub, x='contrast', y='delta',
            order=[c for c, _ in contrasts],
            color='#1f3a73', markers='_', scale=1.5,
            join=False, errorbar=None, ax=ax)
        ax.axhline(0, color='0.5', lw=0.6, ls='--')
        ax.set_title(RANGE_NICE[rng], pad=4)
        ax.set_xlabel('')
        ax.set_ylabel('Δ within-fold r')
        ax.set_xticks(range(len(contrasts)))
        ax.set_xticklabels([c for c, _ in contrasts])
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_horizontalalignment('right')

    sns.despine(fig=fig, offset=5, trim=True)
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300,
                 bbox_inches='tight', pad_inches=0.02)
    print(f'  wrote {out_path}')
    return fig


# ----------------------------------------------------------------------- main

def main(bids_folder, out_dir):
    apply_nyu_style()
    os.makedirs(out_dir, exist_ok=True)

    decoding = load_decoding(bids_folder)
    if decoding.empty:
        print('No decoding TSVs found yet — nothing to plot.')
        return
    n_subj = decoding['subject'].nunique()
    n_rows = len(decoding)
    print(f'Loaded {n_rows} decoding rows from {n_subj} subjects.\n')

    fig_headline_r(decoding, op.join(out_dir, 'fig1_decoding_r.pdf'))
    fig_medae(decoding,       op.join(out_dir, 'fig2_decoding_medae.pdf'))
    fig_contrasts(decoding,   op.join(out_dir, 'fig3_contrasts.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bids_folder')
    parser.add_argument('--out', default='/tmp/gp_prior_figs_nyu')
    args = parser.parse_args()
    main(args.bids_folder, args.out)
