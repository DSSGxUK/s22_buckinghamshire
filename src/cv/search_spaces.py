from typing import Dict, List, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from lightgbm.sklearn import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE

from skopt.space import Real, Categorical, Integer, check_dimension, Dimension

from ..constants import Aggregations

from ..params import (
    get_random_seed,  # I don't like having a dependency on params. Params should really only be used by DVC
)

def search_space_as_list(search_space: Dict[str, Union[Dimension, Tuple, List]]):
    # check_dimension will do any conversion necessary if the value is not
    # already an skopt dimension.
    search_space = {k: check_dimension(v) for k, v in search_space.items()}
    space_as_list = []
    for name, dim in search_space.items():
        dim.name = name
        space_as_list.append(dim)
    return space_as_list


# Aggregation methods are only use when the multi-upn dataset is selected
AGGREGATION_METHODS = [Aggregations.mean, Aggregations.min, Aggregations.max]

# All search spaces are functions so that we don't unnecessarily create a bunch of models.
# Signature of a search space is a function that takes no arguments and returns a search space and number of iterations
# There is some coupling with src/cv/processing.py because all the parameter names need to be prepended with 'estimator'
# to match the param of the PandasEstimatorWrapper
# Space0 is for testing the script out
LR0 = lambda: (
    {
        "estimator__model": Categorical([LogisticRegression()]),
        "estimator__model__penalty": Categorical(["elasticnet"]),
        "estimator__model__l1_ratio": Real(0.0, 1.0, "uniform"),
        "estimator__model__C": Real(1e-6, 1e2, "log-uniform"),
        "estimator__model__fit_intercept": Integer(0, 1, "uniform"),
        "estimator__model__solver": Categorical(["saga"]),
        "estimator__model__max_iter": Categorical([1]),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical([0, 1e-4, 1e-2, 1, 2, 4]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    3,
)

LR1 = lambda: (
    {
        "estimator__model": Categorical([LogisticRegression()]),
        "estimator__model__penalty": Categorical(["elasticnet"]),
        "estimator__model__l1_ratio": Real(0.0, 1.0, "uniform"),
        "estimator__model__C": Real(1e-6, 1e2, "log-uniform"),
        "estimator__model__fit_intercept": Integer(0, 1, "uniform"),
        "estimator__model__solver": Categorical(["saga"]),
        "estimator__model__max_iter": Integer(50, 5000, "log-uniform"),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical([0, 1e-4, 1e-2, 1, 2, 4]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

LR2 = lambda: (
    {
        "estimator__model": Categorical([LogisticRegression()]),
        "estimator__model__penalty": Categorical(["elasticnet"]),
        "estimator__model__l1_ratio": Real(0.0, 1.0, "uniform"),
        "estimator__model__C": Real(1e-6, 1e2, "log-uniform"),
        "estimator__model__fit_intercept": Integer(0, 1, "uniform"),
        "estimator__model__solver": Categorical(["saga"]),
        "estimator__model__max_iter": Integer(50, 5000, "log-uniform"),
        "estimator__oversampling": Categorical([SMOTE()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__k_neighbors": Integer(1, 25, "log-uniform"),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

SVC1 = lambda: (
    {
        "estimator__model": Categorical([SVC()]),
        "estimator__model__C": Real(1e-5, 1e5, "log-uniform"),
        "estimator__model__gamma": Real(1e-6, 1e1, "log-uniform"),
        "estimator__model__kernel": Categorical(["rbf", "poly", "sigmoid"]),
        "estimator__model__degree": Integer(1, 8, "uniform"),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical(
            [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4]
        ),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)
SVC2 = lambda: (
    {
        "estimator__model": Categorical([SVC()]),
        "estimator__model__C": Real(1e-5, 1e5, "log-uniform"),
        "estimator__model__gamma": Real(1e-6, 1e1, "log-uniform"),
        "estimator__model__kernel": Categorical(["rbf", "poly", "sigmoid"]),
        "estimator__model__degree": Integer(1, 9, "log-uniform"),
        "estimator__oversampling": Categorical([SMOTE()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__k_neighbors": Integer(1, 25, "log-uniform"),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

LINEAR_SVC1 = lambda: (
    {
        "estimator__model": Categorical([LinearSVC()]),
        "estimator__model__penalty": Categorical(["l1", "l2"]),
        "estimator__model__loss": Categorical(
            ["squared_hinge"]
        ),  # peanlty = 'l1' and loss = 'hinge' is not supported
        "estimator__model__dual": Categorical([False]),
        "estimator__model__C": Real(1e-5, 1e5, "log-uniform"),
        "estimator__model__max_iter": Integer(50, 5000, "log-uniform"),
        "estimator__oversampling": Categorical([SMOTE()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__k_neighbors": Integer(1, 25, "log-uniform"),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)
LINEAR_SVC2 = lambda: (
    {
        "estimator__model": Categorical([LinearSVC()]),
        "estimator__model__penalty": Categorical(["l2"]),
        "estimator__model__loss": Categorical(
            ["hinge", "squared_hinge"]
        ),  # peanlty = 'l1' and loss = 'hinge' is not supported
        "estimator__model__dual": Categorical([True]),
        "estimator__model__C": Real(1e-5, 1e5, "log-uniform"),
        "estimator__model__max_iter": (50, 5000, "log-uniform"),
        "estimator__oversampling": Categorical([SMOTE()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__k_neighbors": (1, 25, "log-uniform"),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)
LINEAR_SVC3 = lambda: (
    {
        "estimator__model": Categorical([LinearSVC()]),
        "estimator__model__penalty": Categorical(["l1", "l2"]),
        "estimator__model__loss": Categorical(
            ["squared_hinge"]
        ),  # peanlty = 'l1' and loss = 'hinge' is not supported
        "estimator__model__dual": Categorical([False]),
        "estimator__model__C": Real(1e-5, 1e5, "log-uniform"),
        "estimator__model__max_iter": (50, 5000, "log-uniform"),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical(
            [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4]
        ),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)
LINEAR_SVC4 = lambda: (
    {
        "estimator__model": Categorical([LinearSVC()]),
        "estimator__model__penalty": Categorical(["l2"]),
        "estimator__model__loss": Categorical(
            ["hinge", "squared_hinge"]
        ),  # peanlty = 'l1' and loss = 'hinge' is not supported
        "estimator__model__dual": Categorical([True]),
        "estimator__model__C": Real(1e-5, 1e5, "log-uniform"),
        "estimator__model__max_iter": (50, 5000, "log-uniform"),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical(
            [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4]
        ),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

KNN1 = lambda: (
    {
        "estimator__model": Categorical([KNeighborsClassifier()]),
        "estimator__model__n_neighbors": (1, 31, "uniform"),
        "estimator__model__metric": Categorical(["euclidean", "manhattan", "cosine"]),
        "estimator__model__weights": Categorical(["uniform", "distance"]),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical([0, 1e-4, 1e-2, 1, 2, 4]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

KNN2 = lambda: (
    {
        "estimator__model": Categorical([KNeighborsClassifier()]),
        "estimator__model__n_neighbors": (1, 31, "uniform"),
        "estimator__model__metric": Categorical(["euclidean", "manhattan", "cosine"]),
        "estimator__model__weights": Categorical(["uniform", "distance"]),
        "estimator__oversampling": Categorical([SMOTE()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__k_neighbors": Integer(1, 25, "log-uniform"),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

LGBM0 = lambda: (
    {
        "estimator__model": Categorical([LGBMClassifier()]),
        "estimator__model__boosting_type": Categorical(["gbdt"]),
        "estimator__model__num_leaves": Integer(1, int(31**2), "log-uniform"),  # 31
        "estimator__model__max_depth": Integer(
            -21, 21, "uniform"
        ),  # default  is -1, <= 0 means no depth constraint
        "estimator__model__learning_rate": Real(1e-4, 1e2, "log-uniform"),  # 0.1
        "estimator__model__min_child_samples": Integer(
            1, int(20**2), "log-uniform"
        ),  # 20
        "estimator__model__is_unbalance": Categorical(
            [False, True]
        ),  # I think this takes into account unbalanced data?
        "estimator__model__objective": Categorical(["binary"]),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical([0, 1e-4, 1e-2, 1, 2, 4]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([None, StandardScaler()]),
    },
    2,
)
LGBM0_1 = lambda: (
    {
        "estimator__model": Categorical([LGBMClassifier()]),
        "estimator__model__boosting_type": Categorical(["gbdt"]),
        "estimator__model__num_leaves": Integer(1, int(31**2), "log-uniform"),  # 31
        "estimator__model__max_depth": Integer(
            -21, 21, "uniform"
        ),  # default  is -1, <= 0 means no depth constraint
        "estimator__model__learning_rate": Real(1e-4, 1e2, "log-uniform"),  # 0.1
        "estimator__model__min_child_samples": Integer(
            1, int(20**2), "log-uniform"
        ),  # 20
        "estimator__model__is_unbalance": Categorical(
            [False, True]
        ),  # I think this takes into account unbalanced data?
        "estimator__model__objective": Categorical(["binary"]),
        "estimator__oversampling": Categorical([None]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([None, StandardScaler()]),
    },
    2,
)

LGBM1 = lambda: (
    {
        "estimator__model": Categorical([LGBMClassifier()]),
        "estimator__model__boosting_type": Categorical(["gbdt", "dart", "goss"]),
        "estimator__model__num_leaves": Integer(2, 31**2, "log-uniform"),  # 31
        "estimator__model__max_depth": (
            -21,
            21,
            "uniform",
        ),  # default  is -1, <= 0 means no depth constraint
        "estimator__model__learning_rate": Real(1e-4, 1e2, "log-uniform"),  # 0.1
        "estimator__model__n_estimators": Integer(5, 100**2, "log-uniform"),  # 100
        "estimator__model__min_child_samples": (2, 20**2, "log-uniform"),  # 20
        "estimator__model__is_unbalance": Categorical(
            [False, True]
        ),  # I think this takes into account unbalanced data?
        "estimator__model__objective": Categorical(["binary"]),
        "estimator__model__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": Real(
            0.1, 1.0, "uniform"
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical([0, 1e-4, 1e-2, 1, 2, 4]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    80,
)
LGBM2 = lambda: (
    {
        "estimator__model": Categorical([LGBMClassifier()]),
        "estimator__model__boosting_type": Categorical(["gbdt", "dart", "goss"]),
        "estimator__model__num_leaves": (2, 31**2, "log-uniform"),  # 31
        "estimator__model__max_depth": (
            -21,
            21,
            "uniform",
        ),  # default  is -1, <= 0 means no depth constraint
        "estimator__model__learning_rate": (1e-4, 1e2, "log-uniform"),  # 0.1
        "estimator__model__n_estimators": (5, 100**2, "log-uniform"),  # 100
        "estimator__model__min_child_samples": (2, 20**2, "log-uniform"),  # 20
        "estimator__model__is_unbalance": Categorical(
            [False, True]
        ),  # I think this takes into account unbalanced data?
        "estimator__model__objective": Categorical(["binary"]),
        "estimator__model__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling": Categorical([None]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

LGBM3 = lambda: (
    {
        "estimator__model": Categorical([LGBMClassifier()]),
        "estimator__model__boosting_type": Categorical(["gbdt"]),
        "estimator__model__num_leaves": (2, 31**2, "log-uniform"),  # 31
        "estimator__model__max_depth": (
            -21,
            21,
            "uniform",
        ),  # default  is -1, <= 0 means no depth constraint
        "estimator__model__learning_rate": (1e-4, 1e2, "log-uniform"),  # 0.1
        "estimator__model__n_estimators": (5, 100**2, "log-uniform"),  # 100
        "estimator__model__min_child_samples": (2, 20**2, "log-uniform"),  # 20
        "estimator__model__is_unbalance": Categorical(
            [False, True]
        ),  # I think this takes into account unbalanced data?
        "estimator__model__objective": Categorical(["binary"]),
        "estimator__oversampling": Categorical([RandomOverSampler()]),
        "estimator__oversampling__sampling_strategy": (
            0.1,
            1.0,
            "uniform",
        ),  # Needs to have a lower bound bigger than % of neets and upper bound of 1.
        "estimator__oversampling__random_state": Categorical([get_random_seed()]),
        "estimator__oversampling__shrinkage": Categorical([0, 1e-4, 1e-2, 1, 2, 4]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)
LGBM4 = lambda: (
    {
        "estimator__model": Categorical([LGBMClassifier()]),
        "estimator__model__boosting_type": Categorical(["gbdt"]),
        "estimator__model__num_leaves": Integer(2, 31**2, "log-uniform"),
        "estimator__model__max_depth": Integer(
            -21, 21, "uniform", "integer"
        ),  # default  is -1, <= 0 means onstraint
        "estimator__model__learning_rate": Real(1e-4, 1e2, "log-uniform"),  # 0.1
        "estimator__model__n_estimators": Integer(5, 100**2, "log-uniform"),
        "estimator__model__min_child_samples": Integer(2, 20**2, "log-uniform"),
        "estimator__model__is_unbalance": Categorical(
            [False, True]
        ),  # I think this takes into account unbalanced data?
        "estimator__model__objective": Categorical(["binary"]),
        "estimator__oversampling": Categorical([None]),
        "estimator__imputation": Categorical([None]),
        "estimator__scaler": Categorical([StandardScaler(), None]),
    },
    50,
)

# This dictionary provides the table of all search spaces. We use these ids to determine which search space to run
SEARCH_SPACES = {
    "lr0": LR0,
    "lr1": LR1,
    "lr2": LR2,
    "svc1": SVC1,
    "svc2": SVC2,
    "linear_svc1": LINEAR_SVC1,
    "linear_svc2": LINEAR_SVC2,
    "linear_svc3": LINEAR_SVC3,
    "linear_svc4": LINEAR_SVC4,
    "knn1": KNN1,
    "knn2": KNN2,
    "lgbm0": LGBM0,
    "lgbm0_1": LGBM0_1,
    "lgbm1": LGBM1,
    "lgbm2": LGBM2,
    "lgbm3": LGBM3,
    "lgbm4": LGBM4,
    "lgbm1_basic": LGBM1,
    "lgbm2_basic": LGBM2
}
