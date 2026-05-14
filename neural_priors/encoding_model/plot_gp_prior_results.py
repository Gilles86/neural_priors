"""Headline figures for the GP-prior pRF fit on NPCr.

Loads the per-subject TSVs that ``fit_gp_prior.py`` writes into
``derivatives/encoding_models/gp_prior_roi-NPCr[.smoothed]/`` and
produces the figures we care about, then writes them as a single
multi-page PDF plus per-figure PNGs.

The result schema is fixed:

  * desc-cvr2.tsv          — long-format cvR² per (fold, voxel, method)
  * desc-decoding.tsv      — per (fold, method, omega) — n_sig, mae,
                             median_ae, mae_log, median_ae_log, r,
                             plus the whole-brain FDR mixture summary
  * desc-decoded_trials.tsv — per (fold, method, omega, trial)
  * desc-hyperpars.tsv     — per (fold, GP-regularised parameter)
  * desc-wholebrain_r2_mixture.tsv — one row per subject

Run from the project root:

    python plot_gp_prior_results.py /data/ds-neuralpriors \\
        --out /tmp/gp_prior_figs
"""

import argparse
import os
import os.path as op
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import pandas as pd
import seaborn as sns


PIPELINES = {                                        # display → derivative subdir
    'unsmoothed': 'gp_prior_roi-NPCr',
    'smoothed':   'gp_prior_roi-NPCr.smoothed',
}
METHOD_ORDER = ['classical', 'ml', 'bayes']
OMEGA_ORDER = ['plain', 'distance']
PIPELINE_ORDER = ['unsmoothed', 'smoothed']
RANGES = ['narrow', 'wide']

METHOD_COLORS = {'classical': '#666', 'ml': '#1f77b4', 'bayes': '#d62728'}
OMEGA_HATCH = {'plain': '', 'distance': '///'}
# Paper baselines (analyze_decoding.ipynb, model 15, top-100 voxels).
PAPER_R = {'narrow': 0.082, 'wide': 0.136}
PAPER_MAE = {'narrow': 3.90, 'wide': 6.33}


def _setup_style():
    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'figure.titlesize': 17,
    })
    sns.set_style('whitegrid')


def _concat_tsv(bids_folder, pattern):
    """Concatenate matching TSVs across both pipelines and stamp with subject+pipeline."""
    frames = []
    for pipeline, subdir in PIPELINES.items():
        root = Path(bids_folder) / 'derivatives' / 'encoding_models' / subdir
        for path in sorted(root.glob(pattern)):
            subj = path.name.split('_')[0].replace('sub-', '')
            df = pd.read_csv(path, sep='\t')
            df.insert(0, 'subject', subj)
            df.insert(1, 'pipeline', pipeline)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------- loading

def load_all(bids_folder):
    cvr2     = _concat_tsv(bids_folder, 'sub-*/func/sub-*_desc-cvr2.tsv')
    decoding = _concat_tsv(bids_folder, 'sub-*/func/sub-*_desc-decoding.tsv')
    trials   = _concat_tsv(bids_folder, 'sub-*/func/sub-*_desc-decoded_trials.tsv')
    hyperp   = _concat_tsv(bids_folder, 'sub-*/func/sub-*_desc-hyperpars.tsv')
    wb_fit   = _concat_tsv(bids_folder, 'sub-*/func/sub-*_desc-wholebrain_r2_mixture.tsv')

    # Backward-compat: old runs didn't have an 'omega' column.
    for df in (decoding, trials):
        if len(df) and 'omega' not in df.columns:
            df['omega'] = 'plain'

    return {'cvr2': cvr2, 'decoding': decoding, 'trials': trials,
            'hyperp': hyperp, 'wb_fit': wb_fit}


# ----------------------------------------------------------------- figures

def fig_decoding_r(decoding):
    """Mean per-fold Pearson r per (pipeline, range, method, omega) — the headline.

    Each subject contributes its mean-across-folds r per cell. Boxes show
    the across-subject distribution. Paper baseline overlaid as a dashed
    horizontal line per range.
    """
    per_subj = (decoding
                .groupby(['subject', 'pipeline', 'stim_range',
                          'method', 'omega'])['r']
                .mean()
                .reset_index())

    g = sns.catplot(
        data=per_subj, x='method', y='r',
        hue='omega', row='pipeline', col='stim_range',
        kind='box', order=METHOD_ORDER, hue_order=OMEGA_ORDER,
        row_order=PIPELINE_ORDER, col_order=RANGES,
        height=3.6, aspect=1.2, palette={'plain': '#aaa', 'distance': '#1f77b4'})
    for (row_val, col_val, _), ax in g.axes_dict.items() if hasattr(g, 'axes_dict') else []:
        pass
    # The newer seaborn doesn't always expose axes_dict; do it the manual way:
    for r_idx, pipeline in enumerate(PIPELINE_ORDER):
        for c_idx, rng in enumerate(RANGES):
            ax = g.axes[r_idx, c_idx]
            paper = PAPER_R.get(rng)
            if paper is not None:
                ax.axhline(paper, color='k', ls='--', lw=1, alpha=0.6,
                            label=f'Paper r = {paper:.2f}' if r_idx == 0 and c_idx == 0 else None)
            ax.axhline(0, color='0.6', lw=0.6)
    g.set_axis_labels('Method', 'Per-subject mean r (decoded vs true)')
    g.set_titles('Pipeline = {row_name} | Range = {col_name}')
    g.fig.suptitle('Decoding Pearson r — head-to-head', y=1.02)
    return g.fig


def fig_decoding_medae(decoding):
    """Same layout but for median absolute error in natural numerosity units."""
    per_subj = (decoding
                .groupby(['subject', 'pipeline', 'stim_range',
                          'method', 'omega'])['median_ae']
                .mean()
                .reset_index())
    g = sns.catplot(
        data=per_subj, x='method', y='median_ae',
        hue='omega', row='pipeline', col='stim_range',
        kind='box', order=METHOD_ORDER, hue_order=OMEGA_ORDER,
        row_order=PIPELINE_ORDER, col_order=RANGES,
        height=3.6, aspect=1.2, palette={'plain': '#aaa', 'distance': '#1f77b4'})
    for r_idx, pipeline in enumerate(PIPELINE_ORDER):
        for c_idx, rng in enumerate(RANGES):
            ax = g.axes[r_idx, c_idx]
            paper = PAPER_MAE.get(rng)
            if paper is not None:
                ax.axhline(paper, color='k', ls='--', lw=1, alpha=0.6)
    g.set_axis_labels('Method', 'Median |decoded − true| numerosity')
    g.set_titles('Pipeline = {row_name} | Range = {col_name}')
    g.fig.suptitle('Decoding median absolute error', y=1.02)
    return g.fig


def fig_decoding_log_mae(decoding):
    per_subj = (decoding
                .groupby(['subject', 'pipeline', 'stim_range',
                          'method', 'omega'])['median_ae_log']
                .mean()
                .reset_index())
    g = sns.catplot(
        data=per_subj, x='method', y='median_ae_log',
        hue='omega', row='pipeline', col='stim_range',
        kind='box', order=METHOD_ORDER, hue_order=OMEGA_ORDER,
        row_order=PIPELINE_ORDER, col_order=RANGES,
        height=3.6, aspect=1.2, palette={'plain': '#aaa', 'distance': '#1f77b4'})
    g.set_axis_labels('Method', 'Median |log(decoded) − log(true)|')
    g.set_titles('Pipeline = {row_name} | Range = {col_name}')
    g.fig.suptitle('Decoding error in log space (Weber-friendly)', y=1.02)
    return g.fig


def fig_cvr2(cvr2):
    """Encoding-model cvR² per subject × pipeline × range × method.

    Voxelwise medians per fold then meaned per subject (within-fold pool
    smooths over the worst voxel-outliers, mean across folds completes
    the per-subject point estimate).
    """
    per_subj = (cvr2
                .groupby(['subject', 'pipeline', 'stim_range',
                          'method'])['cvr2']
                .median()
                .reset_index())
    g = sns.catplot(
        data=per_subj, x='method', y='cvr2',
        row='pipeline', col='stim_range',
        kind='box', order=METHOD_ORDER,
        row_order=PIPELINE_ORDER, col_order=RANGES,
        palette=METHOD_COLORS,
        height=3.6, aspect=1.1)
    for r_idx, pipeline in enumerate(PIPELINE_ORDER):
        for c_idx, rng in enumerate(RANGES):
            g.axes[r_idx, c_idx].axhline(0, color='0.6', lw=0.6)
    g.set_axis_labels('Method', 'Median cvR² across voxels & folds')
    g.set_titles('Pipeline = {row_name} | Range = {col_name}')
    g.fig.suptitle('Encoding-model cvR² (median across voxels per fold, '
                    'mean across folds per subject)', y=1.02)
    return g.fig


def fig_contrasts(decoding):
    """Within-subject paired contrasts on per-fold mean r.

    Three contrasts on per-subject mean r:
      * dist − plain  (within a method, within pipeline)
      * bayes − classical  (within omega, within pipeline)
      * bayes(unsm, dist) − classical(sm, dist)  (the big one)
    """
    per_subj = (decoding
                .groupby(['subject', 'pipeline', 'stim_range',
                          'method', 'omega'])['r']
                .mean()
                .reset_index())
    wide = per_subj.pivot_table(
        index=['subject', 'stim_range'],
        columns=['pipeline', 'method', 'omega'],
        values='r').reset_index()

    def _safe(p, m, o):
        try:
            return wide[(p, m, o)]
        except KeyError:
            return pd.Series(np.nan, index=wide.index)

    rows = []
    for _, r in wide.iterrows():
        s = r['subject']
        rng = r['stim_range']
        get = lambda p, m, o: r.get((p, m, o), np.nan)
        rows.append(dict(subject=s, stim_range=rng,
                          contrast='dist − plain (smoothed/bayes)',
                          delta=get('smoothed', 'bayes', 'distance')
                                 - get('smoothed', 'bayes', 'plain')))
        rows.append(dict(subject=s, stim_range=rng,
                          contrast='dist − plain (unsm/bayes)',
                          delta=get('unsmoothed', 'bayes', 'distance')
                                 - get('unsmoothed', 'bayes', 'plain')))
        rows.append(dict(subject=s, stim_range=rng,
                          contrast='bayes − classical (sm/dist)',
                          delta=get('smoothed', 'bayes', 'distance')
                                 - get('smoothed', 'classical', 'distance')))
        rows.append(dict(subject=s, stim_range=rng,
                          contrast='bayes(unsm,dist) − classical(sm,dist)',
                          delta=get('unsmoothed', 'bayes', 'distance')
                                 - get('smoothed', 'classical', 'distance')))
    df = pd.DataFrame(rows).dropna(subset=['delta'])
    if df.empty:
        return None
    g = sns.catplot(
        data=df, x='stim_range', y='delta', col='contrast',
        kind='strip', col_wrap=2, height=3.6, aspect=1.2,
        col_order=df['contrast'].unique(), palette='tab10')
    for ax in g.axes.flat:
        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
    g.set_axis_labels('Range', 'Δ Pearson r')
    g.fig.suptitle('Within-subject contrasts on decoded-r', y=1.02)
    return g.fig


def fig_voxel_yield(decoding):
    """Number of FDR-significant voxels per fold per (pipeline × method)."""
    per_subj = (decoding
                .groupby(['subject', 'pipeline', 'stim_range',
                          'method'])['n_sig_voxels']
                .mean()
                .reset_index())
    g = sns.catplot(
        data=per_subj, x='method', y='n_sig_voxels',
        row='pipeline', col='stim_range',
        kind='box', order=METHOD_ORDER,
        row_order=PIPELINE_ORDER, col_order=RANGES,
        palette=METHOD_COLORS,
        height=3.6, aspect=1.1)
    g.set_axis_labels('Method', 'Mean # FDR-significant voxels per fold')
    g.set_titles('Pipeline = {row_name} | Range = {col_name}')
    g.fig.suptitle('Voxel yield under whole-brain FDR mixture (α=0.05)',
                    y=1.02)
    return g.fig


def fig_gp_hyperparameters(hyperp):
    """Spatial scale + variance + nugget per (regularised parameter, pipeline, range)."""
    if not len(hyperp):
        return None
    figs = []
    for metric, ylab in [('lengthscale', 'Lengthscale (mm)'),
                          ('variance',    'GP variance'),
                          ('nugget',      'GP nugget')]:
        g = sns.catplot(
            data=hyperp, x='parameter', y=metric,
            row='pipeline', col='stim_range',
            kind='box', row_order=PIPELINE_ORDER, col_order=RANGES,
            height=3.4, aspect=1.2)
        g.set_axis_labels('Parameter', ylab)
        g.set_titles('Pipeline = {row_name} | Range = {col_name}')
        g.fig.suptitle(f'GP {metric} per regularised parameter', y=1.03)
        figs.append(g.fig)
    return figs


def fig_trial_scatter(trials, subject=None, pipeline='unsmoothed',
                       stim_range='narrow'):
    """Decoded vs true per trial for one subject (or the first one available).

    One panel per (method, omega). Useful sanity check that decoder
    actually tracks numerosity rather than guessing the mean.
    """
    if not len(trials):
        return None
    if subject is None:
        subject = trials['subject'].iloc[0]
    df = trials[(trials['subject'] == subject)
                 & (trials['pipeline'] == pipeline)
                 & (trials['stim_range'] == stim_range)]
    if df.empty:
        return None
    g = sns.relplot(
        data=df, x='true', y='decoded', col='method', row='omega',
        kind='scatter', alpha=0.4,
        col_order=METHOD_ORDER, row_order=OMEGA_ORDER,
        palette=METHOD_COLORS, height=3.2)
    for ax in g.axes.flat:
        lo = min(df['true'].min(), df['decoded'].min())
        hi = max(df['true'].max(), df['decoded'].max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4)
    g.fig.suptitle(
        f'sub-{subject} {pipeline} {stim_range}: decoded vs true', y=1.02)
    return g.fig


def fig_wb_mixture_summary(wb_fit):
    """Whole-brain logit-Gauss R² mixture summary per subject."""
    if not len(wb_fit):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    sns.swarmplot(data=wb_fit, x='pipeline', y='r2_threshold',
                   order=PIPELINE_ORDER, ax=axes[0])
    axes[0].set_ylabel('α = 0.05 R² threshold')
    axes[0].set_title('Whole-brain FDR R² threshold')
    sns.swarmplot(data=wb_fit, x='pipeline', y='signal_mean_r2',
                   order=PIPELINE_ORDER, ax=axes[1])
    axes[1].set_ylabel('Mixture signal-component mean R²')
    axes[1].set_title('Whole-brain signal mean')
    sns.swarmplot(data=wb_fit, x='pipeline', y='signal_weight',
                   order=PIPELINE_ORDER, ax=axes[2])
    axes[2].set_ylabel('Mixture signal weight')
    axes[2].set_title('Whole-brain signal mass')
    fig.suptitle('Whole-brain R² mixture summary across subjects', y=1.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- summary tab

def summary_table(decoding):
    """Mean across-subjects of per-subject mean r and medAE per cell."""
    per_subj = (decoding
                .groupby(['subject', 'pipeline', 'stim_range',
                          'method', 'omega'])
                .agg(r=('r', 'mean'),
                     medAE=('median_ae', 'mean'),
                     medAE_log=('median_ae_log', 'mean'),
                     n_sig=('n_sig_voxels', 'mean'))
                .reset_index())
    return (per_subj
            .groupby(['pipeline', 'stim_range', 'method', 'omega'])
            [['r', 'medAE', 'medAE_log', 'n_sig']]
            .mean()
            .round(3))


# --------------------------------------------------------------- main runner

def main(bids_folder, out_dir, subject_for_scatter=None):
    _setup_style()
    os.makedirs(out_dir, exist_ok=True)

    data = load_all(bids_folder)
    decoding = data['decoding']
    if decoding.empty:
        print('No decoding TSVs found — nothing to plot.')
        return
    print(f'Loaded decoding rows: {len(decoding)} '
          f'(subjects={decoding["subject"].nunique()})')

    tab = summary_table(decoding)
    tab.to_csv(op.join(out_dir, 'summary.tsv'), sep='\t')
    print('\n=== Summary (mean across subjects) ===')
    print(tab.to_string())

    pdf_path = op.join(out_dir, 'gp_prior_figures.pdf')
    with pdf_backend.PdfPages(pdf_path) as pdf:
        for name, fn in [
            ('decoding_r',       lambda: fig_decoding_r(decoding)),
            ('decoding_medae',   lambda: fig_decoding_medae(decoding)),
            ('decoding_logmae',  lambda: fig_decoding_log_mae(decoding)),
            ('cvr2',             lambda: fig_cvr2(data['cvr2'])),
            ('contrasts',        lambda: fig_contrasts(decoding)),
            ('voxel_yield',      lambda: fig_voxel_yield(decoding)),
            ('trial_scatter',    lambda: fig_trial_scatter(
                data['trials'], subject=subject_for_scatter)),
            ('wb_mixture',       lambda: fig_wb_mixture_summary(data['wb_fit'])),
        ]:
            try:
                fig = fn()
            except Exception as e:
                print(f'  {name} failed: {e!r}')
                continue
            if fig is None:
                continue
            png_path = op.join(out_dir, f'{name}.png')
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f'  wrote {png_path}')

        # GP hyperparameter set is three figures.
        for i, fig in enumerate(fig_gp_hyperparameters(data['hyperp']) or []):
            png_path = op.join(out_dir, f'gp_hyperpars_{i}.png')
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f'  wrote {png_path}')

    print(f'\nMulti-page PDF: {pdf_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bids_folder', help='BIDS root, e.g. /data/ds-neuralpriors')
    parser.add_argument('--out', default='/tmp/gp_prior_figs',
                        help='Output directory for PDF + PNGs')
    parser.add_argument('--subject', default=None,
                        help='Subject ID for per-trial scatter (defaults to first available)')
    args = parser.parse_args()
    main(args.bids_folder, args.out, subject_for_scatter=args.subject)
