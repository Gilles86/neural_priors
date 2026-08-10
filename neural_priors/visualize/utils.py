import numpy as np
import cortex
import matplotlib as mpl
from matplotlib import colors, cm


def get_alpha_vertex(data, alpha, cmap='nipy_spectral', vmin=np.log(5), vmax=np.log(80), standard_space=False, subject='fsaverage'):

    data = np.clip((data - vmin) / (vmax - vmin), 0., .99)
    data[alpha < 0.01] = 0
    # Use the registry lookup so user-registered cmaps (e.g. 'compression')
    # are found in addition to the built-ins. `mpl.colormaps[name]` works for
    # both, while the older `getattr(cm, name)` only sees built-ins.
    cmap_obj = cmap if hasattr(cmap, '__call__') else mpl.colormaps[cmap]
    red, green, blue = cmap_obj(data,)[:, :3].T

    # Get curvature
    curv = cortex.db.get_surfinfo(subject)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map.
    curv.data = np.sign(curv.data.data) * .25
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray'
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data]).astype(np.float32)

    vx_rgb = (np.vstack([red.data, green.data, blue.data]) * 255.).astype(np.float32)

    display_data = vx_rgb * alpha[np.newaxis, :] + curv_rgb * (1.-alpha[np.newaxis, :])

    return cortex.VertexRGB(*display_data.astype(np.uint8), subject)