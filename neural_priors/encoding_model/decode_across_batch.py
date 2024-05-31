from neural_priors.utils.data import get_all_subject_ids
from itertools import product
from decode_across import main

# for subject_id, smoothed, range_n in product(get_all_subject_ids()[:1], [False, True], ['narrow', 'wide']):
for subject_id, smoothed, range_n in product(['24'], [False, True], ['narrow', 'wide']):
    main(subject_id, range_n, smoothed, '/data/ds-neuralpriors', False)