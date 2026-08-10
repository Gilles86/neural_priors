import cortex
import numpy as np
import matplotlib.pyplot as plt
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import pandas as pd

subject = '39'

model_label = 5

vmin = 5
vmax = 25
cvr2_thr = 0.0

sub = Subject(subject_id=subject)

pars = sub.get_prf_parameters_surf2(model_label=model_label, smoothed=True)

# pars[pars == 0.0] = np.nan

# Set all rows where mu.narrow and mu.wide are both < 5.0 to NaN
# pars.loc[(pars['mu.narrow'] < 10.0) | (pars['mu.wide'] < 10.0), 'mu.narrow'] = np.nan
# pars.loc[(pars['mu.narrow'] < 10.0) | (pars['mu.wide'] < 10.0), 'mu.wide'] = np.nan
pars.loc[(pars['mu.narrow'] < vmin) | (pars['mu.wide'] < vmin), 'cvr2'] = -1.

# Where cvr2 is 0.0, set it to nan
# pars.loc[pars['cvr2'] == 0.0] = np.nan

# pars.loc[(pars['mu.narrow'] < 5.0) & pars['mu.wide'] < 5.0] = np.nan

ds = {}


ds[f'cvr2'] = cortex.Vertex(pars['cvr2'].values, subject=f'neuralpriors.sub-{subject}', cmap='plasma')


ds[f'cvr2_thr'] = get_alpha_vertex(pars['cvr2'].values, (pars['cvr2'] > cvr2_thr).values, vmin=vmin, vmax=.05, subject=f'neuralpriors.sub-{subject}', cmap='plasma')

ds[f'mu.narrow'] = get_alpha_vertex(pars['mu.narrow'].values, (pars['cvr2'] > cvr2_thr).values, vmin=vmin, vmax=25, subject=f'neuralpriors.sub-{subject}', cmap='nipy_spectral')
ds[f'mu.wide'] = get_alpha_vertex(pars['mu.wide'].values, (pars['cvr2'] > cvr2_thr).values, vmin=vmin, vmax=25, subject=f'neuralpriors.sub-{subject}', cmap='nipy_spectral')


# ds[f'model{model_label}_cvr2'] = get_alpha_vertex(d['cvr2'].values, (d['cvr2'] > 0.0).values, vmin=0.0, vmax=.05, subject='fsaverage', cmap='plasma')

cortex.webgl.show(ds)


# df = []

# keys = []
# for sub, model_label in product(subjects, range(1, 9)):
#     try:
#         pars = sub.get_prf_parameters_surf(model_label=model_label, smoothed=True, gaussian=True, space='fsaverage')
#         df.append(pars)
#         keys.append((sub.subject_id, model_label))
#     except Exception as e:
#         print(f'Failed for {sub.subject_id} model {model_label}: {e}')

# df = pd.concat(df, keys=keys, names=['subject_id', 'model_label'])


# for model_label, d in df.groupby('model_label'):

#     d = d.groupby(['hemi', 'vertex']).mean()
#     ds[f'model{model_label}_cvr2'] = get_alpha_vertex(d['cvr2'].values, (d['cvr2'] > 0.0).values, vmin=0.0, vmax=.05, subject='fsaverage', cmap='plasma')

# cortex.webgl.show(ds)

# vmin, vmax = 5, 25
# x = np.linspace(0, 1, 101, True)

# # Width is 80 x 1, so 
# im = plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
#         extent=[vmin, vmax, 0, 1], aspect=1.*(vmax-vmin) / 20.,
#         origin='lower')
# print(im.get_extent())
# plt.yticks([])
# plt.tight_layout()

# ns = np.array([5, 10, 15, 20, 25])
# ns = ns[ns <= vmax]
# plt.xticks(ns, fontsize=24)
# plt.show()