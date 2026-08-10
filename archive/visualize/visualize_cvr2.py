import cortex
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import pandas as pd

subject = '04'

subjects = [Subject(subject_id=subject_id) for subject_id in get_all_subject_ids()]

ds = {}


df = []

keys = []
for sub, model_label in product(subjects, range(1, 9)):
    try:
        pars = sub.get_prf_parameters_surf(model_label=model_label, smoothed=True, gaussian=True, space='fsaverage')
        df.append(pars)
        keys.append((sub.subject_id, model_label))
    except Exception as e:
        print(f'Failed for {sub.subject_id} model {model_label}: {e}')

df = pd.concat(df, keys=keys, names=['subject_id', 'model_label'])


for model_label, d in df.groupby('model_label'):

    d = d.groupby(['hemi', 'vertex']).mean()
    ds[f'model{model_label}_cvr2'] = get_alpha_vertex(d['cvr2'].values, (d['cvr2'] > 0.0).values, vmin=0.0, vmax=.05, subject='fsaverage', cmap='plasma')

cortex.webgl.show(ds)