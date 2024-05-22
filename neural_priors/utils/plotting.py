import seaborn as sns
from braincoder.utils import get_rsq
from neural_priors.utils.data import Subject
import pandas as pd

def plot_prf_predictions(subject, voxels=None, session=None, smoothed=True, roi='NPCr', range_n='both',
                         bids_folder='/data/ds-neuralpriors', n_voxels=None, col_wrap=4,
                         bins=None,
                         sharey=False):

    if (type(subject) is str) or (type(subject) is int):
        subject = Subject(subject, bids_folder)

    if range_n == 'both':
        ranges = ['wide', 'narrow']
    else:
        ranges = [range_n]
    
    pred = [subject.get_prf_predictions(session, smoothed=smoothed, roi=roi, range_n=range_n, return_image=False)
            for range_n in ranges]
    pred = pd.concat(pred, keys=ranges, names=['range'])

    behavior = subject.get_behavioral_data()


    bold = pd.DataFrame(subject.get_single_trial_estimates(session=session, smoothed=smoothed, roi=roi), index=behavior.index)
    bold.set_index(behavior['range'], append=True, inplace=True)
    bold.set_index(behavior['n'], append=True, inplace=True)


    if range_n is not None:
        if range_n != 'both':
            bold = bold.xs(range_n, 0, 'range', drop_level=False) 

    bold.columns.name = 'voxel'
    pred.columns.name = 'voxel'

    result = bold.stack().to_frame('data').join(pred.stack().to_frame('prediction'))

    result = result[~result.isnull().any(axis=1)]

    if voxels is None:
        if n_voxels is None:
            n_voxels = 12
        
        r2 = get_rsq(result['data'].unstack('voxel'), result['prediction'].unstack('voxel'))

        voxels = r2[r2 != 1.0].sort_values(ascending=False).index[:n_voxels]

    if len(voxels) > 50:
        raise ValueError('Too many voxels to plot')

    result = result.unstack('voxel').loc[:, (slice(None), voxels)].stack('voxel')

    hue = 'range' if range_n == 'both' else None
    g = sns.FacetGrid(result.reset_index(), col='voxel', col_wrap=col_wrap, sharey=sharey, hue=hue)

    g.map(sns.lineplot, 'n', 'data', errorbar='se')
    g.map(sns.lineplot, 'n', 'prediction', lw=2.)

    return g