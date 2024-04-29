import numpy as np
import cortex
from matplotlib import colors, cm


def get_alpha_vertex(data, alpha, cmap='nipy_spectral', vmin=np.log(5), vmax=np.log(80), standard_space=False, subject='fsaverage'):

    data = np.clip((data - vmin) / (vmax - vmin), 0., .99)
    data[alpha < 0.01] = 0
    red, green, blue = getattr(cm, cmap)(data,)[:, :3].T

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