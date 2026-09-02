"""How much stronger is numerosity tuning than value tuning, on a matched analysis?

Reads the source data written by run_analysis.py (numerosity) and value_side.py
(value), plus value_prf's own normalised-axis output, and makes one figure whose
panels carry the comparison without a caption:

  a  Example numerosity voxels (NPC): all 480 single trials, equal-N binned
     means, the weighted-PRF fit, and the envelope of the leave-one-run-out
     refits.
  b  The same for the best value voxels (vmPFC), from value_prf's own selection.
  c  The CIRCULAR control on the normalised axis x = (stimulus - mu)/sigma. Both
     datasets produce a clean Gaussian here from selection alone -- this panel
     exists so panel d can be read at all.
  d  The real test: mu, sigma and the voxel selection come from the OTHER half
     of the runs.
  e  Per-subject centre-minus-flank on the held-out axis -- the effect size.
  f  Per-subject best single-voxel cvR^2 against its own label-permutation null.

Also writes notes/data/data_numerosity_vs_value.tsv: one tidy row per plotted
value, with a `dataset` column, so every number in the figure can be re-plotted.

Run:
  KERAS_BACKEND=tensorflow PYTHONPATH=~/git/neural_priors:~/git/value_prf \
      ~/mambaforge/envs/braincoder/bin/python \
      -m neural_priors.value_comparison.plot_numerosity_vs_value
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RC = {
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4, 'patch.linewidth': 0.5,
    'legend.frameon': False, 'legend.handlelength': 1.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
}

C_NUM = '#8172B2'        # numerosity, NPC
C_NUM_M = '#B7A9D9'      # numerosity, trial-count-matched
C_VMPFC = '#C44E52'      # value, vmPFC
C_V1 = '#3B5BA5'         # value, V1
C_NULL = '#9C9C9C'       # whole-brain control
DATA_C = '0.15'

X_LIM = 4.0
MIN_VOX_FRAC = 0.25
TIDY = []


def tid(**kw):
    TIDY.append(kw)


def panel_letter(ax, letter, x=-0.26, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='bottom', ha='right')


# ── loading ─────────────────────────────────────────────────────────────────

def load(datadir):
    d = Path(datadir)
    D = dict(
        num_stats=pd.read_csv(d / 'numerosity_voxel_stats.tsv', sep='\t'),
        val_stats=pd.read_csv(d / 'value_voxel_stats.tsv', sep='\t'),
        num_binned=pd.read_csv(d / 'numerosity_normalized_binned.tsv', sep='\t'),
        num_ex_trials=pd.read_csv(d / 'numerosity_example_trials.tsv', sep='\t'),
        num_ex_bins=pd.read_csv(d / 'numerosity_example_bins.tsv', sep='\t'),
        num_ex_curves=pd.read_csv(d / 'numerosity_example_curves.tsv', sep='\t'),
        # interior-peaked voxels: the best-fitting ones prefer LOW numerosities
        # and so look monotonic over the sampled range, which is honest but
        # does not show what a numerosity PRF looks like when its peak is
        # inside the range (see plot_bell_examples.py)
        num_bell_trials=pd.read_csv(d / 'numerosity_bell_trials.tsv', sep='\t'),
        num_bell_bins=pd.read_csv(d / 'numerosity_bell_bins.tsv', sep='\t'),
        num_bell_curves=pd.read_csv(d / 'numerosity_bell_curves.tsv', sep='\t'),
        val_norm=pd.concat([pd.read_csv(d / f'data_prf_normalized_ds{ds}.tsv',
                                        sep='\t') for ds in (1, 2)],
                           ignore_index=True),
        val_norm_sub=pd.concat(
            [pd.read_csv(d / f'data_prf_normalized_per_subject_ds{ds}.tsv',
                         sep='\t') for ds in (1, 2)], ignore_index=True),
        val_ex_bins=pd.read_csv(d / 'data_example_voxel_fits_nice.tsv', sep='\t'),
        val_ex_trials=pd.read_csv(d / 'data_example_voxel_trials_nice.tsv',
                                  sep='\t'),
        val_ex_curves=pd.read_csv(d / 'data_example_voxel_modelcurves_nice.tsv',
                                  sep='\t'))
    for name, fn in (('cf_null', 'heldout_centreflank_null.tsv'),
                     ('interior', 'heldout_interior_mu.tsv')):
        D[name] = pd.read_csv(d / fn, sep='\t') if (d / fn).exists() else None
    D['num_stats']['subject'] = D['num_stats']['subject'].map(
        lambda s: f'{int(s):02d}')
    for k in ('num_ex_trials', 'num_ex_bins', 'num_ex_curves'):
        D[k]['subject'] = D[k]['subject'].map(lambda s: f'{int(s):02d}')
    return D


# ── normalised-axis aggregation ─────────────────────────────────────────────

def agg_numerosity(binned, roi, sample, model, alignment):
    g = binned[(binned.roi == roi) & (binned['sample'] == sample)
               & (binned.model == model) & (binned.alignment == alignment)]
    m = g.pivot_table(index='subject', columns='x_norm', values='measured_z')
    p = g.pivot_table(index='subject', columns='x_norm', values='model_z')
    f = g.pivot_table(index='subject', columns='x_norm', values='frac_voxels')
    n = len(m)
    good = np.nanmean(f.values, 0) >= MIN_VOX_FRAC
    return dict(x=m.columns.values.astype(float), good=good, n=n,
                m=np.nanmean(m.values, 0),
                se=np.nanstd(m.values, 0, ddof=1) / np.sqrt(n),
                pm=np.nanmean(p.values, 0), per_subject=m.values)


def agg_value(val_norm, roi, model, alignment):
    """Average the two value datasets' group curves (they are separate
    experiments; the per-subject test is done in panel e)."""
    g = val_norm[(val_norm.roi == roi) & (val_norm.model == model)
                 & (val_norm.alignment == alignment)]
    m = g.pivot_table(index='dataset', columns='x_norm', values='measured_mean_z')
    p = g.pivot_table(index='dataset', columns='x_norm', values='model_mean_z')
    s = g.pivot_table(index='dataset', columns='x_norm', values='measured_sem_z')
    f = g.pivot_table(index='dataset', columns='x_norm',
                      values='frac_voxels_contributing')
    return dict(x=m.columns.values.astype(float),
                good=np.nanmean(f.values, 0) >= MIN_VOX_FRAC,
                m=np.nanmean(m.values, 0),
                se=np.sqrt(np.nanmean(s.values ** 2, 0)),
                pm=np.nanmean(p.values, 0), n=m.shape[0])


def centre_flank(x, y, good):
    cc = np.abs(x) < 0.5
    ff = (np.abs(x) > 1.5) & (np.abs(x) < 3)
    with np.errstate(all='ignore'):
        return float(np.nanmean(y[cc & good]) - np.nanmean(y[ff & good]))


def per_subject_cf(agg):
    x, good = agg['x'], agg['good']
    return np.array([centre_flank(x, row, good) for row in agg['per_subject']])


# ── panels ──────────────────────────────────────────────────────────────────

def example_panel(ax, tr, bn, cu, color, xlabel, xticks, title):
    ax.scatter(tr['x'], tr['y'], s=1.6, lw=0, color='0.74', alpha=0.55,
               zorder=1, rasterized=True)
    ax.fill_between(cu['x'], cu['lo'], cu['hi'], color=color, alpha=0.22, lw=0,
                    zorder=2)
    ax.plot(cu['x'], cu['y'], color=color, lw=1.4, zorder=4)
    ax.errorbar(bn['x'], bn['y'], yerr=bn['sem'], fmt='o', color=DATA_C, ms=2.8,
                lw=0.8, capsize=0, zorder=5)
    lo = np.nanmin(bn['y'] - bn['sem'])
    hi = np.nanmax(bn['y'] + bn['sem'])
    pad = 0.85 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set(xlabel=xlabel, xticks=xticks)
    ax.set_title(title, fontsize=6.3, pad=2)
    ax.tick_params(labelsize=6.5)


def normalised_panel(ax, series):
    for s in series:
        g = s['good']
        if s.get('pm') is not None:
            ax.plot(s['x'][g], s['pm'][g], color=s['color'], lw=0.9, ls=':',
                    zorder=3, alpha=0.9)
        if s.get('se') is not None:
            ax.fill_between(s['x'][g], (s['m'] - s['se'])[g],
                            (s['m'] + s['se'])[g], color=s['color'], alpha=0.20,
                            lw=0, zorder=2)
        ax.plot(s['x'][g], s['m'][g], 'o-', color=s['color'], ms=2.2, lw=1.2,
                zorder=4)
    ax.axvline(0, color='0.7', lw=0.6, ls='--', zorder=0)
    ax.axhline(0, color='0.7', lw=0.6, ls='--', zorder=0)
    ax.set(xlim=(-X_LIM, X_LIM), xticks=[-4, -2, 0, 2, 4],
           xlabel='(Stimulus − μ) / σ')


def strip_summary(ax, groups, ylabel):
    rng = np.random.default_rng(0)
    for i, (lab, col, vals) in enumerate(groups):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=5, lw=0,
                   color=col, alpha=0.40, zorder=2)
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(i, m, yerr=se, fmt='D', color=col, ms=5.5, lw=1.6,
                    capsize=0, zorder=4, markeredgecolor='0.15',
                    markeredgewidth=1.0)
    ax.axhline(0, color='0.6', lw=0.7, ls='--', zorder=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(ylabel)


# ── figure ──────────────────────────────────────────────────────────────────

def make_figure(D, outstem, roi_num='NPCr', n_examples=4,
                bell=True):
    mpl.rcParams.update(RC)
    fig = plt.figure(figsize=(7.25, 9.3))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.15, 1.15],
                          hspace=0.95, wspace=0.45)
    lines = []

    # ── a: example numerosity voxels ────────────────────────────────────────
    if bell:
        ex, exb, exc = (D['num_bell_trials'], D['num_bell_bins'],
                        D['num_bell_curves'])
    else:
        ex, exb, exc = D['num_ex_trials'], D['num_ex_bins'], D['num_ex_curves']
    ns_all = D['num_stats']
    # the SAME statistic used for the value panels: the 95th percentile of the
    # best-voxel cvR^2 under permuted labels, i.e. an ROI-wise (family-wise)
    # threshold.  Comparing a voxel to an uncorrected per-voxel null would
    # flatter numerosity, whose ROIs have ~7x fewer voxels.
    nnull = (ns_all[(ns_all.roi == roi_num) & (ns_all['sample'] == 'full')]
             .set_index('subject')['null_max_p95'])
    if bell:
        # already filtered to interior peaks; spread over preferred numerosity
        keys = (ex[['subject', 'voxel', 'cvr2_loro', 'preferred_n']]
                .drop_duplicates().sort_values('preferred_n')
                .drop_duplicates('subject'))
        keys = keys.iloc[np.linspace(0, len(keys) - 1, n_examples).astype(int)]
    else:
        keys = (ex[['subject', 'voxel', 'cvr2_loro']].drop_duplicates()
                .sort_values('cvr2_loro', ascending=False)
                .drop_duplicates('subject').head(n_examples))
    n_beat_num = int(sum(k.cvr2_loro > nnull.get(k.subject, np.inf)
                         for _, k in keys.iterrows()))
    for j, (_, k) in enumerate(keys.iterrows()):
        ax = fig.add_subplot(gs[0, j])
        s = (ex.subject == k.subject) & (ex.voxel == k.voxel)
        sb = (exb.subject == k.subject) & (exb.voxel == k.voxel)
        sc = (exc.subject == k.subject) & (exc.voxel == k.voxel)
        example_panel(
            ax, dict(x=ex[s].stimulus.values, y=ex[s].measured_beta.values),
            dict(x=exb[sb].stimulus_bin_center.values,
                 y=exb[sb].measured_mean.values,
                 sem=exb[sb].measured_sem.values),
            dict(x=exc[sc].stimulus_grid.values,
                 y=exc[sc].model_prediction.values,
                 lo=exc[sc].model_loro_lo.values,
                 hi=exc[sc].model_loro_hi.values),
            C_NUM, 'Numerosity', [10, 20, 30, 40],
            f'sub-{k.subject} · cvR² = {k.cvr2_loro:.3f}\n'
            f'ROI null p95 = {nnull.get(k.subject, np.nan):.3f}')
        if j == 0:
            ax.set_ylabel('Response (a.u.)')
            panel_letter(ax, 'a')
            ax.text(0.0, 1.36, f'Numerosity — best NPC voxels · '
                    f'{n_beat_num}/{n_examples} beat the ROI-wise '
                    f'permutation null',
                    transform=ax.transAxes, fontsize=7.5, fontweight='bold',
                    color=C_NUM, ha='left', va='bottom')
        for xx, yy, rr in zip(ex[s].stimulus, ex[s].measured_beta, ex[s].run):
            tid(dataset='numerosity', panel='a', roi=roi_num,
                subject=k.subject, voxel=int(k.voxel), series='single_trial',
                x=float(xx), y=float(yy), yerr=np.nan, run=int(rr))
        for xx, yy, ss in zip(exb[sb].stimulus_bin_center,
                              exb[sb].measured_mean, exb[sb].measured_sem):
            tid(dataset='numerosity', panel='a', roi=roi_num,
                subject=k.subject, voxel=int(k.voxel), series='binned_mean',
                x=float(xx), y=float(yy), yerr=float(ss))
        for xx, yy, lo, hi in zip(exc[sc].stimulus_grid,
                                  exc[sc].model_prediction,
                                  exc[sc].model_loro_lo, exc[sc].model_loro_hi):
            tid(dataset='numerosity', panel='a', roi=roi_num,
                subject=k.subject, voxel=int(k.voxel), series='model_curve',
                x=float(xx), y=float(yy), yerr=np.nan, band_lo=float(lo),
                band_hi=float(hi))

    # ── b: example value voxels ─────────────────────────────────────────────
    vex = D['val_ex_trials'][D['val_ex_trials'].roi == 'vmPFC']
    vexb = D['val_ex_bins'][D['val_ex_bins'].roi == 'vmPFC']
    vexc = D['val_ex_curves'][D['val_ex_curves'].roi == 'vmPFC']
    vkeys = (vex[['subject', 'voxel', 'cvr2_loro']].drop_duplicates()
             .sort_values('cvr2_loro', ascending=False)
             .drop_duplicates('subject').head(n_examples))
    vs = D['val_stats']
    vnull = vs[(vs.roi == 'vmpfc') & (vs.value_dataset == 2)] \
        .set_index('subject')['null_max_p95']
    n_beat_val = int(sum(k.cvr2_loro > vnull.get(k.subject, np.inf)
                         for _, k in vkeys.iterrows()))
    ax_b0 = None
    for j, (_, k) in enumerate(vkeys.iterrows()):
        ax = fig.add_subplot(gs[1, j])
        if j == 0:
            ax_b0 = ax
        s = (vex.subject == k.subject) & (vex.voxel == k.voxel)
        sb = (vexb.subject == k.subject) & (vexb.voxel == k.voxel)
        sc = (vexc.subject == k.subject) & (vexc.voxel == k.voxel)
        nl = vnull.get(k.subject, np.nan)
        example_panel(
            ax, dict(x=vex[s].value.values, y=vex[s].measured_beta.values),
            dict(x=vexb[sb].value_bin_center.values,
                 y=vexb[sb].measured_mean.values,
                 sem=vexb[sb].measured_sem.values),
            dict(x=vexc[sc].value_grid.values,
                 y=vexc[sc].model_prediction.values,
                 lo=vexc[sc].model_loro_lo.values,
                 hi=vexc[sc].model_loro_hi.values),
            C_VMPFC, 'Value rating', [0, 450, 900],
            f'sub-{int(k.subject):02d} · cvR² = {k.cvr2_loro:.3f}\n'
            f'ROI null p95 = {nl:.3f}')
        if j == 0:
            ax.set_ylabel('Response (a.u.)')
            panel_letter(ax, 'b')
            ax.text(0.0, 1.36, f'Value — best vmPFC voxels · '
                    f'{n_beat_val}/{n_examples} beat the ROI-wise '
                    f'permutation null',
                    transform=ax.transAxes, fontsize=7.5, fontweight='bold',
                    color=C_VMPFC, ha='left', va='bottom')
        for xx, yy in zip(vex[s].value, vex[s].measured_beta):
            tid(dataset='value', panel='b', roi='vmPFC',
                subject=f'{int(k.subject):05d}', voxel=int(k.voxel),
                series='single_trial', x=float(xx), y=float(yy), yerr=np.nan)
        for xx, yy, ss in zip(vexb[sb].value_bin_center, vexb[sb].measured_mean,
                              vexb[sb].measured_sem):
            tid(dataset='value', panel='b', roi='vmPFC',
                subject=f'{int(k.subject):05d}', voxel=int(k.voxel),
                series='binned_mean', x=float(xx), y=float(yy), yerr=float(ss))
        for xx, yy, lo, hi in zip(vexc[sc].value_grid, vexc[sc].model_prediction,
                                  vexc[sc].model_loro_lo,
                                  vexc[sc].model_loro_hi):
            tid(dataset='value', panel='b', roi='vmPFC',
                subject=f'{int(k.subject):05d}', voxel=int(k.voxel),
                series='model_curve', x=float(xx), y=float(yy), yerr=np.nan,
                band_lo=float(lo), band_hi=float(hi))

    # ── c / d: normalised axis ──────────────────────────────────────────────
    num = {a: agg_numerosity(D['num_binned'], roi_num, 'full', 'prf', a)
           for a in ('circular', 'heldout')}
    numm = {a: agg_numerosity(D['num_binned'], roi_num, 'matched128', 'prf', a)
            for a in ('circular', 'heldout')}
    wb = {a: agg_numerosity(D['num_binned'], 'wholebrain', 'full', 'prf', a)
          for a in ('circular', 'heldout')}
    vvm = {a: agg_value(D['val_norm'], 'vmPFC', 'prf', a)
           for a in ('circular', 'heldout')}
    vv1 = {a: agg_value(D['val_norm'], 'V1', 'prf', a)
           for a in ('circular', 'heldout')}

    for col, al, letter, title in (
            (0, 'circular', 'c',
             'Circular control — fit and shown on the SAME trials'),
            (2, 'heldout', 'd',
             'The real test — μ, σ and voxel selection from the OTHER runs')):
        ax = fig.add_subplot(gs[2, col:col + 2])
        series = [
            dict(x=num[al]['x'], m=num[al]['m'], se=num[al]['se'],
                 good=num[al]['good'], pm=num[al]['pm'], color=C_NUM,
                 label='Numerosity, NPC'),
            dict(x=vvm[al]['x'], m=vvm[al]['m'], se=vvm[al]['se'],
                 good=vvm[al]['good'], pm=vvm[al]['pm'], color=C_VMPFC,
                 label='Value, vmPFC'),
            dict(x=vv1[al]['x'], m=vv1[al]['m'], se=vv1[al]['se'],
                 good=vv1[al]['good'], pm=vv1[al]['pm'], color=C_V1,
                 label='Value, V1'),
        ]
        normalised_panel(ax, series)
        ax.set_ylabel('Response (z)')
        ax.set_title(title, fontsize=7)
        panel_letter(ax, letter, x=-0.13)
        for i, s in enumerate(series):
            ax.text(0.985, 0.96 - 0.115 * i, s['label'], transform=ax.transAxes,
                    color=s['color'], fontsize=6.5, ha='right', va='top')
            for xx, yy, ee, gg in zip(s['x'], s['m'], s['se'], s['good']):
                tid(dataset=s['label'], panel=letter, roi=s['label'],
                    subject='group', series='measured_z', x=float(xx),
                    y=float(yy), yerr=float(ee), shown=bool(gg))
            for xx, yy, gg in zip(s['x'], s['pm'], s['good']):
                tid(dataset=s['label'], panel=letter, roi=s['label'],
                    subject='group', series='model_z', x=float(xx),
                    y=float(yy), yerr=np.nan, shown=bool(gg))
        ax.text(0.015, 0.985, 'Dotted: model prediction', transform=ax.transAxes,
                fontsize=6.0, color='0.45', ha='left', va='top')
        if al == 'circular':
            ax.text(0.015, 0.86,
                    'A peak here proves nothing:\nit is what noise does when\n'
                    '4 parameters are chosen\nper voxel',
                    transform=ax.transAxes, fontsize=6.2, color='0.3',
                    ha='left', va='top')
        else:
            pk = float(np.nanmax(series[0]['m'][series[0]['good']]))
            ax.annotate('Numerosity generalises;\nvalue barely does',
                        xy=(-0.25, pk), xytext=(-3.85, pk * 2.6),
                        fontsize=6.8, color=C_NUM, ha='left', va='center',
                        arrowprops=dict(arrowstyle='-|>',
                                        connectionstyle='angle3,angleA=0,angleB=65',
                                        color=C_NUM, lw=1.1, mutation_scale=8,
                                        shrinkA=3, shrinkB=9,
                                        relpos=(1.0, 0.5)))

    # ── e: per-subject centre-minus-flank (held out) ────────────────────────
    ax = ax_e = fig.add_subplot(gs[3, 0:2])
    vsub = D['val_norm_sub']
    vh = vsub[(vsub.alignment == 'heldout') & (vsub.model == 'prf')]
    groups = [
        ('Numerosity NPC, 480 trials', C_NUM, per_subject_cf(num['heldout'])),
        ('Numerosity NPC, 128 trials', C_NUM_M, per_subject_cf(numm['heldout'])),
        ('Value vmPFC', C_VMPFC, vh[vh.roi == 'vmPFC'].centre_minus_flank.values),
        ('Value V1', C_V1, vh[vh.roi == 'V1'].centre_minus_flank.values),
        ('Numerosity outside NPC', C_NULL, per_subject_cf(wb['heldout'])),
    ]
    strip_summary(ax, groups, 'Centre − flank (z)')
    panel_letter(ax, 'e', x=-0.13)
    ax.set_title('Held-out tuning amplitude, per subject', fontsize=7)
    # the same statistic with the stimulus labels permuted -- the level this
    # test scores when there is nothing to find
    nullmap = {}
    cn = D.get('cf_null')
    if cn is not None:
        nullmap = {
            'Numerosity NPC, 480 trials': cn[(cn.dataset == 'numerosity')
                                             & (cn.roi == 'NPCr')
                                             & (cn['sample'] == 'full')].null_cf_mean,
            'Numerosity NPC, 128 trials': cn[(cn.dataset == 'numerosity')
                                             & (cn.roi == 'NPCr')
                                             & (cn['sample'] == 'matched128')].null_cf_mean,
            'Value vmPFC': cn[(cn.dataset == 'value')
                              & (cn.roi == 'vmpfc')].null_cf_mean,
            'Value V1': cn[(cn.dataset == 'value')
                           & (cn.roi == 'v1')].null_cf_mean,
            'Numerosity outside NPC': cn[(cn.dataset == 'numerosity')
                                         & (cn.roi == 'wholebrain')].null_cf_mean,
        }
        for i, (lab, col, _) in enumerate(groups):
            v = np.asarray(nullmap.get(lab, []), float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            ax.errorbar(i + 0.32, m, yerr=se, fmt='o', mfc='white', ms=4,
                        lw=1.2, capsize=0, color=col, zorder=4)
            for x in v:
                tid(dataset=lab, panel='e', roi=lab, subject='',
                    series='centre_minus_flank_heldout_PERMUTED', x=np.nan,
                    y=float(x), yerr=np.nan)
        ax.text(0.985, 0.97, 'Open circles: stimulus labels permuted',
                transform=ax.transAxes, fontsize=6.2, color='0.35',
                ha='right', va='top')
    for lab, col, v in groups:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        t, p = stats.ttest_1samp(v, 0)
        lines.append(f'{lab.replace(chr(10), " "):28s} centre−flank = '
                     f'{v.mean():+.4f} z  (SEM {v.std(ddof=1)/np.sqrt(len(v)):.4f}), '
                     f't({len(v)-1}) = {t:.2f}, p = {p:.3g}, n = {len(v)}')
        for i, x in enumerate(v):
            tid(dataset=lab.replace('\n', ' '), panel='e',
                roi=lab.replace('\n', ' '), subject=f'{i}',
                series='centre_minus_flank_heldout', x=np.nan, y=float(x),
                yerr=np.nan)
    top = max(np.nanmean(g[2]) for g in groups)
    ax.set_ylim(None, top * 2.0)

    # ── f: single-voxel cvR^2 vs its permutation null ───────────────────────
    ax = fig.add_subplot(gs[3, 2:4])
    ns = D['num_stats']
    pts = [
        ('Numerosity, NPC', C_NUM,
         ns[(ns.roi == roi_num) & (ns['sample'] == 'full')]),
        ('Value, vmPFC', C_VMPFC, vs[vs.roi == 'vmpfc']),
        ('Value, V1', C_V1, vs[vs.roi == 'v1']),
    ]
    for lab, col, g in pts:
        ax.scatter(g.null_max_p95, g.real_max, s=9, lw=0.4, color=col,
                   alpha=0.7, edgecolor='white', zorder=3, label=lab)
        for a, b in zip(g.null_max_p95, g.real_max):
            tid(dataset=lab, panel='f', roi=lab, subject='', series='null_vs_real',
                x=float(a), y=float(b), yerr=np.nan)
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color='0.55', lw=0.8, ls='--', zorder=1)
    ax.set(xlim=lim, ylim=lim, xlabel='Permutation null, 95th pct of max cvR²',
           ylabel='Best real voxel, cvR²')
    ax.set_title('Is any single voxel individually significant?\n'
                 '(above the dashed line = yes)', fontsize=7)
    panel_letter(ax, 'f', x=-0.16)
    for i, (lab, col, g) in enumerate(pts):
        frac = float((g.p_max < 0.05).mean())
        ax.text(0.985, 0.20 - 0.075 * i,
                f'{lab}: {100*frac:.0f}% of subjects p < 0.05',
                transform=ax.transAxes, color=col, fontsize=6.3, ha='right',
                va='top')
        lines.append(f'{lab:20s} best-voxel cvR2 = {g.real_max.mean():.3f}, '
                     f'null p95 = {g.null_max_p95.mean():.3f}, mean exceedance '
                     f'p = {g.p_max.mean():.3f}, {100*frac:.0f}% of subjects '
                     f'p<0.05 (n = {len(g)})')
    ax.text(0.985, 0.985, 'Best voxel = its own\npermutation null',
            transform=ax.transAxes, fontsize=6.0, color='0.45', ha='right',
            va='top', rotation=0)

    # honest note about what panel a actually shows, placed in the gap under
    # row b rather than floating over any data
    interior = D.get('interior')
    if interior is not None:
        gi = interior[(interior.dataset == 'numerosity')]
        gv = interior[(interior.dataset == 'value') & (interior.roi == 'vmPFC')]
        y0 = ax_b0.get_position().y0 - 0.052
        fig.text(0.06, y0,
                 'The best numerosity voxels prefer LOW numerosities (median '
                 f'μ = {gi.median_mu.median():.0f} of 10–40), so over the '
                 'sampled range their tuning looks like a ramp. That alone '
                 'could drive a centre−flank difference,\nso the held-out test '
                 'was repeated using only voxels with an interior μ (≥1 basis '
                 'σ from either end): numerosity keeps '
                 f'{100*gi.cf_interior.mean()/gi.cf_all.mean():.0f}% of its '
                 f'effect ({gi.cf_interior.mean():+.3f} z), value vmPFC '
                 f'{100*gv.cf_interior.mean()/gv.cf_all.mean():.0f}% '
                 f'({gv.cf_interior.mean():+.3f} z).',
                 fontsize=6.3, color='0.3', ha='left', va='top')

    fig.text(0.06, -0.012,
             'Numerosity: 39 subjects, 480 trials, NPC (right). Value: 64 '
             'subjects pooled over two datasets, 128 trials, vmPFC and V1. '
             'Identical encoding models, grids and selection throughout '
             '(value_prf code, imported).\n'
             'a, b  Grey dots: every single trial. Black: equal-N binned mean '
             '± SEM. Coloured band: envelope of the leave-one-run-out refits. '
             '"ROI null p95" is the 95th percentile of the best-voxel cvR² '
             'under permuted stimulus labels.\n'
             'c–e  Error bars/bands are ±1 SEM across subjects. Bins where '
             '<25% of the selected voxels contribute are hidden. '
             'f  Each point is one subject.',
             fontsize=6.0, color='0.35', ha='left', va='top')

    sns.despine(fig=fig, offset=3, trim=False)
    # sns.despine re-sets the ticks and DROPS any text properties set on the
    # tick labels (verified: rotation goes 32 -> 0).  Rotate afterwards.
    plt.setp(ax_e.get_xticklabels(), rotation=30, ha='right', fontsize=6.2)
    fig.savefig(f'{outstem}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{outstem}.pdf', bbox_inches='tight')
    return lines, dict(num=num, numm=numm, wb=wb, vvm=vvm, vv1=vv1)


def main(datadir, outdir, roi_num='NPCr'):
    D = load(datadir)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lines, aggs = make_figure(D, str(Path(outdir) / 'numerosity_vs_value'),
                              roi_num=roi_num)
    tidy = pd.DataFrame(TIDY)
    tidy.to_csv(Path(datadir) / 'data_numerosity_vs_value.tsv', sep='\t',
                index=False)
    print(f'saved {outdir}/numerosity_vs_value.pdf (+ .png)')
    print(f'saved {datadir}/data_numerosity_vs_value.tsv  ({len(tidy)} rows)')
    print('\n=== summary ===')
    for l in lines:
        print(' ', l)
    return tidy, lines


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datadir',
                   default=str(Path.home() / 'git/value_prf/notes/data'))
    p.add_argument('--outdir',
                   default=str(Path.home() / 'git/value_prf/notes/figures'))
    p.add_argument('--roi', default='NPCr')
    a = p.parse_args()
    main(a.datadir, a.outdir, roi_num=a.roi)
