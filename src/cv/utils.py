from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import OptimizeResult
from skopt.space import Categorical


def fix_checkpoint_x_iters(x_iters, search_space):
    """This gets around a bug in scikit-optimize when loading a checkpoint.

    When scikit-optimize validates that all the prior x_iters (prior tested hyperparams)
    are within the search space, it will check whether two scikit-learn estimators or transformers
    in the overall pipeline are the same. This will fail because, although the two objects
    are the same type, they will be different instances. This gets around the issue
    by replacing all scikit-learn objects with the one used in the search space.
    """
    new_x_iters = []
    for x_iter in x_iters:
        new_x_iter = []
        for param, dim in zip(x_iter, search_space):
            if isinstance(param, BaseEstimator) or isinstance(param, TransformerMixin):
                assert isinstance(dim, Categorical)
                matching_param = [
                    val for val in dim.categories if type(val) == type(param)
                ]
                assert len(matching_param) > 0
                new_x_iter.append(matching_param[0])
            else:
                new_x_iter.append(param)
        new_x_iters.append(new_x_iter)
    return new_x_iters


def inspect_optimizer_result(res):
    breakpoint()
