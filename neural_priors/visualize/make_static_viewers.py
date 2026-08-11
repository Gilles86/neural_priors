"""
Build a standalone pycortex WebGL viewer (cortex.webgl.make_static) per
subject for a given encoding model — same datasets and masking as
visualize_subject_model.py (cvR2, preferred numerosity narrow/wide), but
written to a self-contained folder of static HTML that can be hosted on
GitHub Pages. Run from this directory (`from utils import ...`).
"""

import cortex
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
import os.path as op
import argparse


# --- custom μ colormap "laura" — warm orange (low μ) → cool teal (high μ) --
_LAURA_CMAP = LinearSegmentedColormap.from_list(
    'laura', [
        '#ee7a04', '#ea8660', '#e19288', '#d3a0a6', '#c4aeba',
        '#b3bbc7', '#a2c7cf', '#95d3d0', '#8eddca',
    ], N=256,
)
if 'laura' not in mpl.colormaps:
    mpl.colormaps.register(_LAURA_CMAP, name='laura')

# tuning-width (σ, log space) display range and colormap — fixed across
# subjects/models for comparability; covers p5–p95 of models 3/14/31
SD_VMIN, SD_VMAX, SD_CMAP = 0.1, 2.0, 'viridis'


def _vol_to_surf_both_hemis(img, surfinfo):
    from nilearn import surface as nl_surface
    return np.concatenate(
        [np.squeeze(nl_surface.vol_to_surf(img, surfinfo[hemi]['outer'],
                                           inner_mesh=surfinfo[hemi]['inner']))
         for hemi in ['L', 'R']]).astype(np.float32)


def sample_roi_model_surf(sub, model_label=31, smoothed=True, roi='NPCr',
                          frac_thr=0.33,
                          pars=('mu.narrow', 'mu.wide', 'cvr2')):
    """Sample the T1w-space parameter volumes of an ROI-fitted model (e.g.
    model 31, fitted within NPCr only) onto the fsnative surface, mirroring
    surface/sample_prf_to_surface_nilearn.py.

    The parameter volumes are exactly zero outside the ROI, so plain
    vol_to_surf dilutes boundary vertices toward zero (depth-averaging mixes
    in outside-ROI voxels). Correction: sample the binary ROI mask through
    the identical pipeline, giving the in-ROI fraction f per vertex, and
    divide the sampled parameters by f. Vertices with f <= frac_thr are
    marked invalid (cvr2 = -1). Returns dict of L+R vertex arrays, including
    'roi_frac' (the raw sampled mask fraction, for outline drawing)."""

    import nibabel as nb

    smooth_ext = '.smoothed' if smoothed else ''
    surfinfo = sub.get_surf_info()

    def par_fn(par):
        key = (f'model{model_label}' + ('.cv' if par == 'cvr2' else '')
               + smooth_ext)
        return op.join(sub.bids_folder, 'derivatives', 'encoding_models', key,
                       f'sub-{sub.subject_id}', 'func',
                       f'sub-{sub.subject_id}_desc-{par}.optim_space-T1w_pars.nii.gz')

    # ROI support straight from the parameter volume: mu is softplus-positive
    # inside the fitted ROI and exactly 0 outside (masker inverse_transform),
    # so (mu != 0) is the masker's support on the same grid.
    mu_img = nb.load(par_fn('mu.narrow'))
    support = nb.Nifti1Image((mu_img.get_fdata() != 0).astype('float32'),
                             mu_img.affine)
    frac = _vol_to_surf_both_hemis(support, surfinfo)
    valid = frac > frac_thr

    out = {}
    for par in pars:
        samples = _vol_to_surf_both_hemis(par_fn(par), surfinfo)
        out[par] = np.where(valid, samples / np.maximum(frac, 1e-6), 0.)

    out['cvr2'] = np.where(valid, out['cvr2'], -1.)
    out['roi_frac'] = frac
    return out


def get_npcr_vertex_mask(sub):
    """NPCr as a vertex mask on the subject's native surface, obtained the
    same way the original ROI pipeline (surface/get_npc_mask.py) defines it:
    the group fsaverage NPC1 label resampled to fsnative. mri_surf2surf's
    label mapping is nearest-neighbor on the registered spheres, replicated
    here with a KD-tree so no FreeSurfer installation is needed. NPCr is the
    right-hemisphere label; left-hemisphere vertices are all False."""
    import nibabel as nb
    from nilearn import surface as nl_surface
    from scipy.spatial import cKDTree

    fsdir = op.join(sub.bids_folder, 'derivatives', 'fmriprep',
                    'sourcedata', 'freesurfer')
    # NB: the fitted NPCr ROI is the full NPC label (dice 0.98 with the
    # ips_masks volume mask), NOT the NPC1 subregion (dice 0.46)
    label = nl_surface.load_surf_data(
        op.join(sub.bids_folder, 'derivatives', 'surface_masks',
                'desc-NPC_R_space-fsaverage_hemi-rh.label.gii'))
    avg_sphere = nb.freesurfer.read_geometry(
        op.join(fsdir, 'fsaverage', 'surf', 'rh.sphere.reg'))[0]
    subj_sphere = nb.freesurfer.read_geometry(
        op.join(fsdir, f'sub-{sub.subject_id}', 'surf', 'rh.sphere.reg'))[0]
    nearest = cKDTree(avg_sphere).query(subj_sphere, workers=-1)[1]
    rh_mask = label[nearest] > 0.5

    n_lh = len(nb.freesurfer.read_geometry(
        op.join(fsdir, f'sub-{sub.subject_id}', 'surf', 'lh.white'))[0])
    return np.concatenate([np.zeros(n_lh, dtype=bool), rh_mask])


def get_roi_outline(pyc_subject, vertex_mask, dilate=1):
    """Vertex indices of the ROI border: in-mask vertices with at least one
    out-of-mask neighbor on the surface mesh, optionally dilated by
    (dilate - 1) extra neighbor rings for visibility."""
    from scipy import sparse

    rows, cols = [], []
    offset = 0
    for hemi_pts, hemi_polys in [cortex.db.get_surf(pyc_subject, 'wm',
                                                    hemisphere=h)
                                 for h in ['left', 'right']]:
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            rows.append(hemi_polys[:, a] + offset)
            cols.append(hemi_polys[:, b] + offset)
        offset += len(hemi_pts)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    n = offset
    adj = sparse.coo_matrix(
        (np.ones(len(rows) * 2),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n)).tocsr()

    outside = (~vertex_mask).astype(float)
    outline = vertex_mask & (adj.dot(outside) > 0)
    for _ in range(dilate - 1):
        outline = vertex_mask & ((adj.dot(outline.astype(float)) > 0) | outline)
    return np.where(outline)[0]


def paint_outline(vertex_rgb, outline_idx, color=(255, 255, 255)):
    """Overwrite the RGB channels of a pycortex VertexRGB at the given
    vertex indices (used to draw the NPCr contour into every map)."""
    for channel, value in zip(['red', 'green', 'blue'], color):
        getattr(vertex_rgb, channel).data[outline_idx] = value


def make_dataset(subject, model_label, vmin=5, vmax=25, cvr2_thr=0.0,
                 cmap='nipy_spectral', roi_model_labels=(31, 14)):
    sub = Subject(subject_id=subject)
    pars = sub.get_prf_parameters_surf(model_label=model_label, smoothed=True)

    # mask voxels whose preferred μ is outside [vmin, vmax] in EITHER condition
    out_of_range = ((pars['mu.narrow'] < vmin) | (pars['mu.wide'] < vmin)
                    | (pars['mu.narrow'] > vmax) | (pars['mu.wide'] > vmax))
    pars.loc[out_of_range, 'cvr2'] = -1.

    alpha = (pars['cvr2'] > cvr2_thr).values
    pyc_subject = f'neuralpriors.sub-{subject}'

    ds = {}
    m = f'model {model_label}'
    ds[f'mu narrow ({m})'] = get_alpha_vertex(
        pars['mu.narrow'].values, alpha, vmin=vmin, vmax=vmax,
        subject=pyc_subject, cmap=cmap)

    if not np.allclose(pars['mu.narrow'].values, pars['mu.wide'].values):
        ds[f'mu wide ({m})'] = get_alpha_vertex(
            pars['mu.wide'].values, alpha, vmin=vmin, vmax=vmax,
            subject=pyc_subject, cmap=cmap)

    ds[f'cvR2 ({m})'] = get_alpha_vertex(
        pars['cvr2'].values, alpha, vmin=0.0, vmax=.05,
        subject=pyc_subject, cmap='plasma')

    # tuning width (model 3 fits a single σ shared by both conditions)
    from nilearn import surface as nl_surface
    d3 = op.join(sub.bids_folder, 'derivatives', 'encoding_models',
                 f'model{model_label}.whole_brain.smoothed',
                 f'sub-{subject}', 'func')
    sd3 = np.concatenate([nl_surface.load_surf_data(op.join(
        d3, f'sub-{subject}_desc-sd.narrow.optim.nilearn_space-fsnative_hemi-{h}.func.gii'))
        for h in ['L', 'R']])
    ds[f'sd ({m})'] = get_alpha_vertex(
        sd3, alpha, vmin=SD_VMIN, vmax=SD_VMAX,
        subject=pyc_subject, cmap=SD_CMAP)

    # NPCr-fitted models (winning model 31, free-width model 14): shown there
    for rml in (roi_model_labels or []):
        roi_pars = sample_roi_model_surf(
            sub, model_label=rml,
            pars=('mu.narrow', 'mu.wide', 'cvr2', 'sd.narrow', 'sd.wide'))
        roi_out_of_range = ((roi_pars['mu.narrow'] < vmin)
                            | (roi_pars['mu.wide'] < vmin)
                            | (roi_pars['mu.narrow'] > vmax)
                            | (roi_pars['mu.wide'] > vmax))
        roi_pars['cvr2'][roi_out_of_range] = -1.
        roi_alpha = roi_pars['cvr2'] > cvr2_thr

        rm = f'model {rml}, NPCr only'
        ds[f'mu narrow ({rm})'] = get_alpha_vertex(
            roi_pars['mu.narrow'], roi_alpha, vmin=vmin, vmax=vmax,
            subject=pyc_subject, cmap=cmap)
        ds[f'mu wide ({rm})'] = get_alpha_vertex(
            roi_pars['mu.wide'], roi_alpha, vmin=vmin, vmax=vmax,
            subject=pyc_subject, cmap=cmap)
        ds[f'cvR2 ({rm})'] = get_alpha_vertex(
            roi_pars['cvr2'], roi_alpha, vmin=0.0, vmax=.05,
            subject=pyc_subject, cmap='plasma')

        ds[f'sd narrow ({rm})'] = get_alpha_vertex(
            roi_pars['sd.narrow'], roi_alpha, vmin=SD_VMIN, vmax=SD_VMAX,
            subject=pyc_subject, cmap=SD_CMAP)
        if rml == 14:
            # only model 14 fits σ_wide freely; model 31's is a fixed
            # 1.29 × σ_narrow, so its wide map would be redundant
            ds[f'sd wide ({rm})'] = get_alpha_vertex(
                roi_pars['sd.wide'], roi_alpha, vmin=SD_VMIN, vmax=SD_VMAX,
                subject=pyc_subject, cmap=SD_CMAP)

    # annotate NPCr as a white contour on every map, from the original
    # surface-label definition of the ROI (smooth, no voxel round-trip)
    outline = get_roi_outline(pyc_subject, get_npcr_vertex_mask(sub),
                              dilate=2)
    for vertex_rgb in ds.values():
        paint_outline(vertex_rgb, outline)

    return ds


# Attribution bar injected into every generated viewer: fixed, full-width
# along the bottom, with a short explanation and a link to the preprint and
# the subject overview page.
BANNER_TEMPLATE = """
<div id="paper-banner" style="position:fixed;left:0;right:0;bottom:0;z-index:1000;
     background:rgba(255,255,255,.95);border-top:1px solid #aaa;
     padding:12px 20px;display:flex;gap:28px;align-items:center;
     justify-content:space-between;flex-wrap:wrap;
     font:15px/1.5 'Helvetica Neue',Arial,sans-serif;color:#222;">
  <div style="flex:1 1 520px;min-width:340px;">
    <b style="font-size:17px;">Numerosity receptive fields
    &mdash; participant sub-{subject}</b><br>
    Each vertex is colored by its preferred numerosity (&mu;,
    5&ndash;25), its tuning width (&sigma;, log space, 0.1&ndash;2.0),
    both estimated per stimulus-range condition, or by the
    cross-validated variance explained of the encoding model
    (cvR&sup2;&nbsp;&gt;&nbsp;0 shown). The white contour marks NPCr
    (right numerosity parietal cortex). Maps for model&nbsp;31 (the
    paper's winning model, fixed 1.29 width scaling) and model&nbsp;14
    (tuning width free per voxel) only cover NPCr, the region they were
    fitted in.<br>
    <b>Keys:</b> <kbd style="font-weight:bold;">i</kbd>/<kbd style="font-weight:bold;">f</kbd> inflate or flatten the
    surface (here &ldquo;flat&rdquo; means fully inflated, no cut-open
    flatmap), <kbd style="font-weight:bold;">+</kbd>/<kbd style="font-weight:bold;">&minus;</kbd> step through the maps.
  </div>
  <div style="flex:0 0 auto;">
    <img src="{mu_cbar}" alt="Colorbar preferred numerosity"
         style="display:block;height:56px;">
    <img src="{sd_cbar}" alt="Colorbar tuning width"
         style="display:block;height:56px;margin-top:2px;">
    <img src="{cvr2_cbar}" alt="Colorbar cvR2"
         style="display:block;height:56px;margin-top:2px;">
  </div>
  <div style="flex:0 1 auto;text-align:right;">
    <a href="https://doi.org/10.1101/2025.09.25.675916" target="_blank"
       style="color:#1a5276;font-weight:600;">Distributed range adaptation in
    human parietal encoding of numbers</a><br>
    Prat-Carrabin*, de&nbsp;Hollander*, Bedi, Gershman &amp; Ruff
    (bioRxiv, 2025)<br>
    <a href="../../index.html" style="color:#1a5276;">&larr; All participants</a>
  </div>
</div>
</body>"""


# Interactive per-voxel scatter (μ narrow vs μ wide in NPCr), injected as a
# collapsible panel. JSON data is embedded per subject; plain canvas, no
# external libraries (GitHub Pages friendly). Uses .replace() tokens, not
# str.format, because the JS is full of braces.
SCATTER_TEMPLATE = """
<div id="scatter-panel" style="position:fixed;right:12px;bottom:175px;z-index:1001;
     width:426px;background:rgba(255,255,255,.95);border:1px solid #aaa;
     border-radius:8px;font:13px/1.4 'Helvetica Neue',Arial,sans-serif;color:#222;">
  <div style="display:flex;align-items:center;justify-content:space-between;
       padding:8px 12px 2px;">
    <b style="font-size:14px;">Receptive fields in NPCr</b>
    <button id="scatter-toggle" style="border:1px solid #bbb;background:#fff;
        border-radius:4px;cursor:pointer;font-size:13px;line-height:1;
        padding:2px 7px;">&ndash;</button>
  </div>
  <div id="scatter-body" style="padding:2px 10px 10px;position:relative;">
    <div style="padding:0 2px 2px;">
      <label style="cursor:pointer;"><input type="radio" name="scmodel"
          value="m3" checked> Model 3</label>
      <label style="cursor:pointer;margin-left:10px;"><input type="radio"
          name="scmodel" value="m31"> Model 31</label>
      <label style="cursor:pointer;margin-left:10px;"><input type="radio"
          name="scmodel" value="m14"> Model 14</label>
      <span id="scatter-median" style="float:right;color:#555;"></span>
    </div>
    <div style="padding:0 2px 4px;">
      <label style="cursor:pointer;"><input type="radio" name="scpar"
          value="mu" checked> &mu; (preferred numerosity)</label>
      <label style="cursor:pointer;margin-left:10px;"><input type="radio"
          name="scpar" value="sd"> &sigma; (tuning width)</label>
    </div>
    <canvas id="scatter-canvas" style="width:400px;height:400px;display:block;"></canvas>
    <div style="color:#666;font-size:11.5px;padding-top:5px;">
      Each dot is one NPCr voxel with cvR&sup2;&nbsp;&gt;&nbsp;0: its preferred
      numerosity (&mu;) or tuning width (&sigma;) in the narrow (x) vs the
      wide (y) range condition. Dashed line: no change between conditions.
      Values are the actual voxel fits (not surface-resampled).
      Selecting a model or parameter here also switches the brain map
      (and <kbd style="font-weight:bold;">+</kbd>/<kbd
      style="font-weight:bold;">&minus;</kbd> updates this panel).</div>
    <div id="scatter-tip" style="position:absolute;display:none;background:#222;
        color:#fff;padding:3px 7px;border-radius:4px;font-size:11.5px;
        pointer-events:none;white-space:nowrap;z-index:2;"></div>
  </div>
</div>
<script>
(function () {
  var DATA = __SCATTER_JSON__;
  var LO = 5, HI = 25, W = 400, H = 400, ML = 34, MB = 32, MT = 8, MR = 8;
  var canvas = document.getElementById('scatter-canvas');
  var tip = document.getElementById('scatter-tip');
  var medianEl = document.getElementById('scatter-median');
  var dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  var curModel = 'm3', curPar = 'mu', pts = [];
  function sx(v) { return ML + (v - LO) / (HI - LO) * (W - ML - MR); }
  function sy(v) { return H - MB - (v - LO) / (HI - LO) * (H - MT - MB); }
  function fmt(v) { return (HI - LO) >= 10 ? Math.round(v) : v.toFixed(1); }
  function draw(hoverIx) {
    var view = (DATA[curModel] || {})[curPar] || {lo: 0, hi: 1, pts: []};
    LO = view.lo; HI = view.hi; pts = view.pts;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#e3e3e3'; ctx.lineWidth = 1; ctx.fillStyle = '#555';
    ctx.font = '10.5px Helvetica,Arial,sans-serif';
    var ticks = [];
    for (var i = 0; i <= 5; i++) { ticks.push(LO + (HI - LO) * i / 5); }
    ticks.forEach(function (t) {
      ctx.beginPath(); ctx.moveTo(sx(t), MT); ctx.lineTo(sx(t), H - MB); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ML, sy(t)); ctx.lineTo(W - MR, sy(t)); ctx.stroke();
      ctx.textAlign = 'center'; ctx.fillText(fmt(t), sx(t), H - MB + 13);
      ctx.textAlign = 'right'; ctx.fillText(fmt(t), ML - 5, sy(t) + 3.5);
    });
    ctx.strokeStyle = '#999'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(sx(LO), sy(LO)); ctx.lineTo(sx(HI), sy(HI));
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = '#aaa';
    ctx.strokeRect(ML, MT, W - ML - MR, H - MT - MB);
    ctx.fillStyle = '#333'; ctx.textAlign = 'center';
    ctx.font = '11.5px Helvetica,Arial,sans-serif';
    var sym = curPar === 'mu' ? '\\u03bc' : '\\u03c3';
    ctx.fillText(sym + ' narrow', ML + (W - ML - MR) / 2, H - 4);
    ctx.save(); ctx.translate(10, MT + (H - MT - MB) / 2);
    ctx.rotate(-Math.PI / 2); ctx.fillText(sym + ' wide', 0, 0); ctx.restore();
    pts.forEach(function (p, i) {
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(26,82,118,0.55)'; ctx.fill();
      if (i === hoverIx) {
        ctx.lineWidth = 1.5; ctx.strokeStyle = '#111'; ctx.stroke();
      }
    });
    var ds = pts.map(function (p) { return p[1] - p[0]; }).sort(function (a, b) { return a - b; });
    var med = ds.length ? ds[Math.floor(ds.length / 2)] : null;
    medianEl.textContent = med === null ? 'no voxels' :
      'median \\u0394 = ' + (med >= 0 ? '+' : '') +
      med.toFixed((HI - LO) >= 10 ? 1 : 2);
  }
  canvas.addEventListener('mousemove', function (ev) {
    var r = canvas.getBoundingClientRect();
    var mx = ev.clientX - r.left, my = ev.clientY - r.top;
    var best = -1, bd = 100;
    pts.forEach(function (p, i) {
      var d = Math.pow(sx(p[0]) - mx, 2) + Math.pow(sy(p[1]) - my, 2);
      if (d < bd) { bd = d; best = i; }
    });
    draw(best);
    if (best >= 0) {
      var p = pts[best];
      var s = curPar === 'mu' ? '\\u03bc' : '\\u03c3';
      tip.textContent = s + ' ' + p[0].toFixed(2) + ' \\u2192 ' + p[1].toFixed(2)
        + ' (cvR\\u00b2 ' + p[2].toFixed(3) + ')';
      tip.style.left = Math.min(mx + 14, 320) + 'px';
      tip.style.top = (my + 18) + 'px';
      tip.style.display = 'block';
    } else { tip.style.display = 'none'; }
  });
  canvas.addEventListener('mouseleave', function () { tip.style.display = 'none'; draw(-1); });
  // --- two-way sync with the pycortex viewer ---------------------------
  // Dataset names as generated by make_dataset(); model 3 has a single sd
  // map, model 31 has no 'sd wide' (it is 1.29x 'sd narrow' by construction).
  var DSSUFFIX = {m3: ' (model 3)', m31: ' (model 31, NPCr only)',
                  m14: ' (model 14, NPCr only)'};
  function syncBrain() {
    if (!window.viewer || !viewer.setData || !viewer.dataviews) return;
    var act = (viewer.active && viewer.active.name) || '';
    var prefix;
    if (curPar === 'mu') {
      prefix = /^mu wide/.test(act) ? 'mu wide' : 'mu narrow';
    } else if (curModel === 'm3') {
      prefix = 'sd';
    } else {
      prefix = (curModel === 'm14' && /^sd wide/.test(act)) ? 'sd wide'
                                                            : 'sd narrow';
    }
    var name = prefix + DSSUFFIX[curModel];
    if (!viewer.dataviews[name] || name === act) return;
    try { viewer.setData(name); } catch (e) {}
  }
  function setRadio(group, value) {
    var el = document.querySelector(
      'input[name="' + group + '"][value="' + value + '"]');
    if (el) el.checked = true;
  }
  function hookViewer() {
    if (!window.viewer || !viewer.addEventListener) return false;
    viewer.addEventListener('setData', function (evt) {
      // follow +/- (or dataset-list) switches; setting .checked does not
      // fire 'change', so this cannot loop back into syncBrain()
      var name = evt.name || '';
      var m = /model (\\d+)/.exec(name);
      var changed = false;
      if (m && DATA['m' + m[1]] && 'm' + m[1] !== curModel) {
        curModel = 'm' + m[1]; setRadio('scmodel', curModel); changed = true;
      }
      var par = /^mu /.test(name) ? 'mu' : (/^sd/.test(name) ? 'sd' : null);
      if (par && par !== curPar) {
        curPar = par; setRadio('scpar', par); changed = true;
      }
      if (changed) draw(-1);
    });
    return true;
  }
  var hookTimer = setInterval(function () {
    if (hookViewer()) clearInterval(hookTimer);
  }, 400);
  Array.prototype.forEach.call(document.getElementsByName('scmodel'), function (el) {
    el.addEventListener('change', function () { curModel = el.value; draw(-1); syncBrain(); });
  });
  Array.prototype.forEach.call(document.getElementsByName('scpar'), function (el) {
    el.addEventListener('change', function () { curPar = el.value; draw(-1); syncBrain(); });
  });
  var body = document.getElementById('scatter-body');
  var btn = document.getElementById('scatter-toggle');
  btn.addEventListener('click', function () {
    var open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    btn.innerHTML = open ? '+' : '&ndash;';
  });
  draw(-1);
})();
</script>"""


def get_scatter_json(subject, vmin=5, vmax=25):
    """Per-voxel narrow-vs-wide values in NPCr for models 3 and 31, for both
    preferred numerosity (mu) and tuning width (sd). Voxels are selected like
    the surface maps (cvr2 > 0, mu within [vmin, vmax]); sd axes are data
    driven (99th percentile). Note these are the actual ROI voxel fits from
    the extracted-parameter TSVs, not surface-resampled values."""
    import json
    sub = Subject(subject_id=subject)
    out = {}
    for tag, ml in [('m3', 3), ('m31', 31), ('m14', 14)]:
        try:
            p = sub.get_prf_parameters_volume(model_label=ml, roi='NPCr')
            mn = p[('mu', 'narrow')].values
            mw = p[('mu', 'wide')].values
            cv = p[('cvr2', 'nan')].values
            sel = ((cv > 0) & (mn >= vmin) & (mn <= vmax)
                   & (mw >= vmin) & (mw <= vmax))
            entry = {'mu': {'lo': vmin, 'hi': vmax,
                            'pts': np.stack([mn[sel], mw[sel], cv[sel]],
                                            1).round(3).tolist()}}
            sn = p[('sd', 'narrow')].values
            sw = p[('sd', 'wide')].values
            hi = float(np.ceil(np.percentile(
                np.concatenate([sn[sel], sw[sel]]), 99) * 2)) / 2 if sel.any() else 1.
            sd_sel = sel & (sn <= hi) & (sw <= hi)
            entry['sd'] = {'lo': 0.0, 'hi': hi,
                           'pts': np.stack([sn[sd_sel], sw[sd_sel], cv[sd_sel]],
                                           1).round(3).tolist()}
            out[tag] = entry
        except Exception as e:
            print(f'scatter data failed for sub-{subject} model {ml}: {e}')
            out[tag] = {'mu': {'lo': vmin, 'hi': vmax, 'pts': []},
                        'sd': {'lo': 0, 'hi': 1, 'pts': []}}
    return json.dumps(out)


def make_colorbar_datauri(cmap, vmin, vmax, label, ticks, ticklabels):
    """Slim horizontal colorbar rendered to a base64 data URI (the
    alpha-blended VertexRGB maps bypass pycortex's own colorbar)."""
    import io
    import base64
    import matplotlib.pyplot as plt

    # NB fonts are in points, so at dpi=300 the title is ~45px tall — the
    # fixed figure must leave real headroom or it gets clipped. A fixed
    # canvas (no bbox_inches='tight') keeps all three PNGs the same size
    # with the bar strip at the same position, so they align when stacked.
    with mpl.rc_context({'font.family': ['Helvetica Neue', 'Helvetica',
                                         'Arial', 'DejaVu Sans']}):
        fig = plt.figure(figsize=(3.4, 0.82), dpi=300)
        ax = fig.add_axes([0.06, 0.42, 0.88, 0.26])
        norm = mpl.colors.Normalize(vmin, vmax)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.colormaps[cmap],
                                       norm=norm, orientation='horizontal')
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels)
        cb.outline.set_linewidth(0.4)
        ax.tick_params(labelsize=9.5, length=2.5, pad=2.5)
        ax.set_title(label, fontsize=11, pad=4)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def inject_banner(outpath, subject, model_label, cmap='nipy_spectral',
                  vmin=5, vmax=25):
    fn = op.join(outpath, 'index.html')
    with open(fn) as f:
        html = f.read()
    # idempotent: strip a previously injected bar so the banner can be
    # refreshed without regenerating the (expensive) pycortex bundle
    html = html.split('<div id="paper-banner"')[0]
    mu_cbar = make_colorbar_datauri(
        cmap, vmin, vmax, 'Preferred numerosity (μ)',
        ticks=[5, 10, 15, 20, 25], ticklabels=['5', '10', '15', '20', '25'])
    cvr2_cbar = make_colorbar_datauri(
        'plasma', 0, .05, 'cvR²',
        ticks=[0, .025, .05], ticklabels=['0', '0.025', '≥0.05'])
    sd_cbar = make_colorbar_datauri(
        SD_CMAP, SD_VMIN, SD_VMAX, 'Tuning width (σ, log space)',
        ticks=[0.1, 0.5, 1.0, 1.5, 2.0],
        ticklabels=['≤0.1', '0.5', '1.0', '1.5', '≥2.0'])
    banner = BANNER_TEMPLATE.format(subject=subject, model_label=model_label,
                                    mu_cbar=mu_cbar, cvr2_cbar=cvr2_cbar,
                                    sd_cbar=sd_cbar)
    scatter = SCATTER_TEMPLATE.replace('__SCATTER_JSON__',
                                       get_scatter_json(subject, vmin, vmax))
    banner = banner.replace('</body>', scatter + '\n</body>')
    if '</body>' in html:
        html = html.replace('</body>', banner)
    else:
        # pycortex's static template leaves <body> unclosed — append instead
        html += banner
    with open(fn, 'w') as f:
        f.write(html)


def main(subject, model_label, out_dir, vmin=5, vmax=25, cvr2_thr=0.0,
         cmap='nipy_spectral', reinject_only=False):
    outpath = op.join(out_dir, f'sub-{subject}')
    if not reinject_only:
        ds = make_dataset(subject, model_label, vmin, vmax, cvr2_thr, cmap)
        cortex.webgl.make_static(outpath, ds)
    inject_banner(outpath, subject, model_label, cmap=cmap,
                  vmin=vmin, vmax=vmax)
    print(f'Wrote {outpath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write standalone pycortex viewer(s) for hosting.')
    parser.add_argument('subject', type=str, nargs='?', default=None,
                        help="Subject ID; omit or pass 'all' for all subjects")
    parser.add_argument('--model_label', type=int, default=3)
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Viewers are written to <out_dir>/sub-<id>/')
    parser.add_argument('--vmin', type=float, default=5.0)
    parser.add_argument('--vmax', type=float, default=25.0)
    parser.add_argument('--cvr2_thr', type=float, default=0.0)
    parser.add_argument('--cmap', type=str, default='nipy_spectral',
                        help="Colormap for μ (default: 'nipy_spectral', as in the paper)")
    parser.add_argument('--reinject_only', action='store_true',
                        help='Only refresh the attribution bar in existing '
                             'viewers; skip the pycortex rebuild')

    args = parser.parse_args()

    if args.subject is None or args.subject == 'all':
        subjects = get_all_subject_ids()
    else:
        subjects = [args.subject]

    for subject in subjects:
        try:
            main(subject, args.model_label, args.out_dir, args.vmin,
                 args.vmax, args.cvr2_thr, args.cmap,
                 reinject_only=args.reinject_only)
        except Exception as e:
            print(f'FAILED sub-{subject}: {e}')
