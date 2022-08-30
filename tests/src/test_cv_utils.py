import pytest

from functools import partial
import pandas as pd
import numpy as np

from sklearn.metrics import (
    make_scorer,
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
)
from lightgbm.sklearn import LGBMClassifier
from skopt.space import Categorical, Real, Integer

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTEN

from src import cv
from src.constants import Models, ParameterSpaceTypes, OversamplingMethods, Scalers


def test_build_pipeline_basic():
    sample_pipeline_steps = [
        ("scaler", Scalers.none),
        ("model", Models.linear_svc),
    ]
    fake_linear_svc = "fake_linear_svc"
    mock_model_constructors = {
        Scalers.none: lambda: None,
        Models.linear_svc: lambda: fake_linear_svc,
    }
    expected_pipeline = Pipeline(
        [("scaler", None), ("model", "fake_linear_svc")],
    )

    actual_pipeline = cv.build_pipeline(
        sample_pipeline_steps, model_constructors=mock_model_constructors
    )

    assert isinstance(actual_pipeline, type(expected_pipeline))
    assert actual_pipeline.steps == expected_pipeline.steps


def test_build_pipeline_errors():
    wrong_steps1 = "hello"
    wrong_steps2 = ("hello", "goodbye")
    wrong_steps3 = ["hello"]
    wrong_steps4 = [("hello", "goodbye"), "hello"]

    with pytest.raises(ValueError) as e_info:
        cv.build_pipeline(steps=wrong_steps1)
    with pytest.raises(ValueError) as e_info:
        cv.build_pipeline(steps=wrong_steps2)
    with pytest.raises(ValueError) as e_info:
        cv.build_pipeline(steps=wrong_steps3)
    with pytest.raises(ValueError) as e_info:
        cv.build_pipeline(steps=wrong_steps4)


def test_build_search_space_basic():
    sample_search_space = [
        (
            {
                "model": [Models.logistic_regression],
                "model__penalty": ["elasticnet"],
                "model__l1_ratio": (0.0, 1.0, "uniform", ParameterSpaceTypes.real),
                "model__fit_intercept": (0, 1, "uniform", ParameterSpaceTypes.integer),
                "model__max_iter": [1],
                "oversampling": [OversamplingMethods.smote],
                "oversampling__k_neighbors": (1, 4),
            },
            3,
        ),
        (
            {
                "model": [Models.logistic_regression],
                "model__penalty": ["elasticnet"],
                "model__l1_ratio": (0.0, 1.0, "uniform", ParameterSpaceTypes.real),
                "model__fit_intercept": (0, 1, "uniform", ParameterSpaceTypes.integer),
                "model__max_iter": [10],
                "oversampling": [OversamplingMethods.none],
            },
            5,
        ),
    ]

    fake_logistic_regression = "fake_logistic_regression"
    fake_smote = "fake_smote"
    mock_constructors = {
        Models.logistic_regression: lambda: fake_logistic_regression,
        OversamplingMethods.smote: lambda: fake_smote,
        OversamplingMethods.none: lambda: None,
    }

    expected_built_search_space = [
        (
            {
                "model": Categorical([fake_logistic_regression]),
                "model__penalty": Categorical(["elasticnet"]),
                "model__l1_ratio": Real(0.0, 1.0, prior="uniform"),
                "model__fit_intercept": Integer(0, 1, prior="uniform"),
                "model__max_iter": Categorical([1]),
                "oversampling": Categorical([fake_smote]),
                "oversampling__k_neighbors": (1, 4),
            },
            3,
        ),
        (
            {
                "model": Categorical([fake_logistic_regression]),
                "model__penalty": Categorical(["elasticnet"]),
                "model__l1_ratio": Real(0.0, 1.0, prior="uniform"),
                "model__fit_intercept": Integer(0, 1, prior="uniform"),
                "model__max_iter": Categorical([10]),
                "oversampling": Categorical([None]),
            },
            5,
        ),
    ]
    pipeline_steps = [
        ("model", Models.logistic_regression),
        ("oversampling", OversamplingMethods.none),
    ]

    actual_search_space = cv.build_search_space(
        sample_search_space,
        pipeline_steps,
        model_constructors=mock_constructors,
        parameter_space_types=cv.PARAMETER_SPACE_TYPES,
    )
    assert expected_built_search_space == actual_search_space


def test_build_search_space_errors():
    not_list_space = {"hello": ["goodbye"]}
    not_tup_subspace = [
        {"hello": ["goodbye"]},
    ]
    not_dict_param_space = [
        ("hello", 50),
    ]
    not_int_num_iter = [
        ({"hello": ["goodbye"]}, 50.0),
    ]
    not_valid_search_range = [
        ({"hello": "goodbye"}, 50.0),
    ]
    pipeline_steps = [
        ("model", Models.logistic_regression),
        ("oversampling", OversamplingMethods.none),
    ]

    with pytest.raises(ValueError) as e_info:
        cv.build_search_space(not_list_space, pipeline_steps)
    with pytest.raises(ValueError) as e_info:
        cv.build_search_space(not_tup_subspace, pipeline_steps)
    with pytest.raises(ValueError) as e_info:
        cv.build_search_space(not_dict_param_space, pipeline_steps)
    with pytest.raises(ValueError) as e_info:
        cv.build_search_space(not_int_num_iter, pipeline_steps)
    with pytest.raises(ValueError) as e_info:
        cv.build_search_space(not_valid_search_range, pipeline_steps)


def test_get_pipeline_params():
    pipeline = Pipeline(
        [("scaler", None), ("oversampling", SMOTEN()), ("model", LGBMClassifier())]
    )
    pipeline.set_params(model__objective="binary", oversampling__k_neighbors=10)
    actual_params = cv.get_pipeline_params(pipeline)
    expected_params = {
        "model__objective": "binary",
        "oversampling__k_neighbors": 10,
        "scaler": None,
        "oversampling": "SMOTEN",
        "model": "LGBMClassifier",
        "steps": ["scaler", "oversampling", "model"],
    }

    assert actual_params.items() >= expected_params.items()


def test_threshold_scorer_init():
    thresholded_metrics = {
        "f1_binary": partial(f1_score, average="binary"),
        "f2_binary": partial(fbeta_score, beta=2, average="binary"),
        "recall_binary": partial(recall_score, average="binary"),
        "precision_binary": partial(precision_score, average="binary"),
        "matthews_corrcoef": matthews_corrcoef,
        "accuracy": accuracy_score,
    }
    nonthresholded_metrics = {
        "roc_auc_macro": partial(roc_auc_score, average="macro"),
    }
    watch_score = "f2_binary"
    ts = cv.ThresholdingScorer(
        thresholded_metrics, nonthresholded_metrics, watch_score=watch_score
    )

    assert ts.thresholded_scoring_functions == thresholded_metrics
    assert ts.nonthresholded_scoring_functions == nonthresholded_metrics
    assert ts.watch_score == watch_score

    assert not ts.metrics_record


def test_threshold_scorer_bad_watch_score():
    thresholded_metrics = {
        "f1_binary": partial(f1_score, average="binary"),
        "f2_binary": partial(fbeta_score, beta=2, average="binary"),
        "recall_binary": partial(recall_score, average="binary"),
        "precision_binary": partial(precision_score, average="binary"),
        "matthews_corrcoef": matthews_corrcoef,
        "accuracy": accuracy_score,
    }
    nonthresholded_metrics = {
        "roc_auc_macro": partial(roc_auc_score, average="macro"),
    }
    watch_score = "bad"
    with pytest.raises(ValueError):
        ts = cv.ThresholdingScorer(
            thresholded_metrics, nonthresholded_metrics, watch_score=watch_score
        )


def test_threshold_scorer_get_names():
    thresholded_metrics = {
        "f1_binary": partial(f1_score, average="binary"),
        "f2_binary": partial(fbeta_score, beta=2, average="binary"),
        "recall_binary": partial(recall_score, average="binary"),
        "precision_binary": partial(precision_score, average="binary"),
        "matthews_corrcoef": matthews_corrcoef,
        "accuracy": accuracy_score,
    }
    nonthresholded_metrics = {
        "roc_auc_macro": partial(roc_auc_score, average="macro"),
    }
    watch_score = "f2_binary"
    ts = cv.ThresholdingScorer(
        thresholded_metrics, nonthresholded_metrics, watch_score=watch_score
    )

    assert ts.get_score_name("a_score") == "a_score_score"
    assert ts.get_threshold_name("a_score") == "a_score_threshold"


def test_threshold_scorer_get_threshold_range_no_filter_between_auto_buffer():
    thresholded_metrics = {
        "f1_binary": partial(f1_score, average="binary"),
        "f2_binary": partial(fbeta_score, beta=2, average="binary"),
        "recall_binary": partial(recall_score, average="binary"),
        "precision_binary": partial(precision_score, average="binary"),
        "matthews_corrcoef": matthews_corrcoef,
        "accuracy": accuracy_score,
    }
    nonthresholded_metrics = {
        "roc_auc_macro": partial(roc_auc_score, average="macro"),
    }
    watch_score = "f2_binary"
    ts = cv.ThresholdingScorer(
        thresholded_metrics, nonthresholded_metrics, watch_score=watch_score
    )

    y_score = np.array(
        [
            -0.0,
            -1.9,
            1.4,
            0.0,
            2.3,
            -0.8,
            0.6,
            -0.0,
            -0.4,
            2.6,
            -0.0,
            -1.9,
            1.4,
            0.9,
            -1.0,
            -0.8,
            0.6,
            -0.0,
            -0.4,
            2.6,
        ]
    )

    r = ts.get_threshold_range(
        y_score, num_thresholds=None, method="between", boundary_buffer="auto"
    )
    expected = np.array(
        [-2.35, -1.45, -0.9, -0.6, -0.2, 0.3, 0.75, 1.15, 1.85, 2.45, 2.75]
    )

    np.testing.assert_almost_equal(r, expected, decimal=9)


def test_threshold_scorer_get_threshold_range_no_filter_weighted_between_auto_buffer():
    thresholded_metrics = {
        "f1_binary": partial(f1_score, average="binary"),
        "f2_binary": partial(fbeta_score, beta=2, average="binary"),
        "recall_binary": partial(recall_score, average="binary"),
        "precision_binary": partial(precision_score, average="binary"),
        "matthews_corrcoef": matthews_corrcoef,
        "accuracy": accuracy_score,
    }
    nonthresholded_metrics = {
        "roc_auc_macro": partial(roc_auc_score, average="macro"),
    }
    watch_score = "f2_binary"
    ts = cv.ThresholdingScorer(
        thresholded_metrics, nonthresholded_metrics, watch_score=watch_score
    )

    y_score = np.array(
        [
            -0.0,
            -1.9,
            1.4,
            0.0,
            2.3,
            -0.8,
            0.6,
            -0.0,
            -0.4,
            2.6,
            -0.0,
            -1.9,
            1.4,
            0.9,
            -1.0,
            -0.8,
            0.6,
            -0.0,
            -0.4,
            2.6,
        ]
    )

    r = ts.get_threshold_range(
        y_score, num_thresholds=None, method="weighted_between", boundary_buffer="auto"
    )
    [-1.9, -1.0, -0.8, -0.4, -0.0, 0.6, 0.9, 1.4, 2.3, 2.6]
    expected = np.array(
        [
            -2.2,
            -1.6,  # -1.9, -1
            (-1 * 1 - 0.8 * 2) / 3,  # -.8, -1
            -0.6,  # -.8, -.4
            -0.4 * 2 / 7,  # -.4, 0
            0.6 * 2 / 7,  # .6, 0
            (0.6 * 2 + 0.9 * 1) / 3,
            (1.4 * 2 + 0.9 * 1) / 3,
            (1.4 * 2 + 2.3 * 1) / 3,
            (2.6 * 2 + 2.3 * 1) / 3,
            (2.6 - (2.6 * 2 + 2.3 * 1) / 3) + 2.6,
        ]
    )

    np.testing.assert_almost_equal(r, expected, decimal=9)


def test_threshold_scoring_decision_function():
    def all_or_nothing(y, y_pred):
        return np.all(y == y_pred).astype(int)

    thresholded_metrics = {
        "f1_binary": partial(f1_score, average="binary"),
        "f2_binary": partial(fbeta_score, beta=2, average="binary"),
        "accuracy": accuracy_score,
    }
    nonthresholded_metrics = {}
    watch_score = "f2_binary"
    ts = cv.ThresholdingScorer(
        thresholded_metrics, nonthresholded_metrics, watch_score=watch_score
    )

    class Estimator:
        def decision_function(self, X):
            return np.array(
                [
                    -0.0,
                    -1.9,
                    1.4,
                    0.0,
                    2.3,
                    -0.8,
                    0.6,
                    -0.0,
                    -0.4,
                    2.6,
                    -0.0,
                    -1.9,
                    1.4,
                    0.9,
                    -1.0,
                    -0.8,
                    0.6,
                    -0.0,
                    -0.4,
                    2.6,
                ]
            )

    rng = ...  # TODO

    y_true = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1])
