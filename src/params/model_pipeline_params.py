import numpy as np

SEED = 2022
_RNG = np.random.default_rng(SEED)


def get_random_seed():
    return _RNG.integers(low=1, high=99999)


TEST_SPLIT = 0.2

# TODO Standardize these search spaces so numerical variables are of the form
# (start, end, prior, paramter_space_type).
#

CV_FOLDS = 4
RUN_PARALLEL = False
NUM_THRESHOLDS = None

LOAD_CHECKPOINTS=False
