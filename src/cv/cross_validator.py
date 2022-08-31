from datetime import datetime
import os
import math
from joblib import Parallel, delayed
import numpy as np
from typing import Any, Callable, Dict, List

import pandas as pd
from sklearn import clone
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
)
from sklearn.base import BaseEstimator

from .. import log_utils as l
from .. import error_utils as eu
from .. import file_utils as f


class ThresholdingScorer:
    def __init__(
        self,
        thresholded_scoring_functions: Dict[str, Callable],
        num_thresholds: int = None,
        logger=l.PrintLogger,
    ):
        # Scoring functions should be "greater is better" type of scores
        self.thresholded_scoring_functions = thresholded_scoring_functions
        self.num_thresholds = num_thresholds
        logger.info(
            f"Threshold scoring {list(self.thresholded_scoring_functions.keys())} with {'all' if self.num_thresholds is None else self.num_thresholds} thresholds."
        )

    @staticmethod
    def get_threshold_name(name: str):
        return f"{name}__threshold"

    @staticmethod
    def get_threshold_range(
        y_score,
        num_thresholds,
        method="between",
        boundary_buffer="auto",
        logger=l.PrintLogger,
    ):

        if num_thresholds is not None:
            if not isinstance(num_thresholds, int):
                logger.error(
                    f"num_thresholds must be an integer, received {num_thresholds} instead"
                )
                raise ValueError()

            step_size = math.ceil(y_score.shape[0] / num_thresholds)
            logger.debug(
                f"For num_thresholds {num_thresholds} and y_score length {y_score.shape[0]}, using step_size {step_size}"
            )
            y_score = np.sort(y_score)[::step_size]

        decision_boundaries, counts = np.unique(y_score, return_counts=True)
        if decision_boundaries.shape[0] == 1:
            # This means there's just one unique value. We'll just use that as our threshold
            return decision_boundaries

        logger.debug(f"Using method {method} for calculation of thresholds")
        if method == "between":
            inner_thresholds = (decision_boundaries[:-1] + decision_boundaries[1:]) / 2
        elif method == "weighted_between":
            weighted_decision_boundaries = decision_boundaries * counts
            inner_thresholds = (
                weighted_decision_boundaries[:-1] + weighted_decision_boundaries[1:]
            ) / (counts[:-1] + counts[1:])
        else:
            eu.error_with_logging(
                f'Unknown threshold range method {method}. Must be in {["between", "weighted_between"]}.',
                logger,
                ValueError,
            )

        if boundary_buffer == "auto":
            logger.debug(f"Automatically computing boundary thresholds")
            left_bd = 2 * decision_boundaries[0] - inner_thresholds[0]
            right_bd = 2 * decision_boundaries[-1] - inner_thresholds[-1]
        else:
            logger.debug(
                f"Using buffer {boundary_buffer} for computing boundary thresholds"
            )
            boundary_buffer = float(boundary_buffer)
            left_bd = decision_boundaries[0] - boundary_buffer
            right_bd = decision_boundaries[-1] - boundary_buffer

        thresholds = np.concatenate(
            [
                [left_bd],
                inner_thresholds,
                [right_bd],
            ]
        )

        return thresholds

    def __call__(self, y, y_score, logger=l.PrintLogger):
        threshold_range = self.get_threshold_range(
            y_score,
            num_thresholds=self.num_thresholds,
            method="between",
            boundary_buffer="auto",
            logger=logger,
        )
        y_scores_all = (y_score[:, None] >= threshold_range[None, :]).astype(int)

        score_metrics = {}
        threshold_metrics = {}
        for name, scorer in self.thresholded_scoring_functions.items():
            best_score = -float("inf")
            best_threshold = None
            for i in range(threshold_range.shape[0]):
                score = scorer(y, y_scores_all[:, i])
                if score > best_score:
                    best_score = score
                    best_threshold = threshold_range[i]

            # change from numpy type to python type
            score_metrics[name] = float(best_score)
            threshold_metrics[self.get_threshold_name(name)] = float(best_threshold)

        return score_metrics, threshold_metrics


class MultiMetricScorer:
    def __init__(
        self, scoring_functions: Dict[str, Callable], logger=l.PrintLogger
    ) -> None:
        self.scoring_functions = scoring_functions

    def __call__(self, y, y_score, logger=l.PrintLogger):
        scorer_metrics = {}
        for name, scorer in self.scoring_functions.items():
            scorer_metrics[name] = float(scorer(y, y_score))

        return (
            scorer_metrics,
            {},
        )  # We have no other metrics that we want to keep track of, just scores


class UnionScorer:
    def __init__(self, *scorers) -> None:
        """A scorer that combines multiple scorers into one.

        Parameters
        ----------
        *scorers: tuple of Scorer
            Scorers to be combined.
        """
        self.scorers = scorers

    def __call__(self, y, y_score):
        scorer_results = []
        other_results = []
        for scorer in self.scorers:
            score_metrics, other_metrics = scorer(y, y_score)
            scorer_results.append(score_metrics)
            other_results.append(other_metrics)

        combined_scorer_metrics = {
            name: val
            for metric_dict in scorer_results
            for name, val in metric_dict.items()
        }
        combined_other_metrics = {
            name: val
            for metric_dict in other_results
            for name, val in metric_dict.items()
        }

        return combined_scorer_metrics, combined_other_metrics


# class ResultsTracker:

#     def __init__(self) -> None:
#         self.params = []
#         self.scoring_metrics = []
#         self.other_metrics = []

#     def add(self, params, scoring_metrics, other_metrics):
#         self.params.append(params)


def fit_and_score(
    estimator,
    Xtr_index: pd.Index,
    ytr: pd.Series,
    Xval_index: pd.Index,
    yval: pd.Series,
    scorer,
):
    """
    This is a grouped fit and score.

    Xtr and ytr are assumed to be pandas objects. They should have the same indexes,
    but
    """
    estimator.fit(Xtr_index, ytr)
    try:
        y_score = estimator.decision_function(Xval_index)
    except AttributeError:
        y_score = estimator.predict_proba(Xval_index)

    return scorer(yval, y_score)


def cv_iterator_pd_wrapper(y_series: pd.Series):
    """This a cross-validation splitter for a binary classification task

    The wrapper takes as input the index y_series, the labels for the data. Its
    index is what will be split, and the labels will be used for stratification

    In our case y_series is always the series where each row is a student and the
    value is a binary 0/1 for whether the outcome (e.g. NEET) occured.
    """

    def pd_cv_iterator(cv_object: BaseCrossValidator):
        # We'll run the cv_iterator over the index of y_series
        # and use y_series's actual values for stratification

        for train, val in cv_object.split(X=y_series.index, y=y_series.values):
            train_index = y_series.index[train]
            val_index = y_series.index[val]
            yield (
                (train_index, y_series.loc[train_index]),
                (val_index, y_series.loc[val_index]),
            )

    return pd_cv_iterator


def pd_score_wrapper(scorer):
    def pd_scorer(y: pd.Series, y_score: pd.Series):
        # reindex y_score to match y
        y_score = y_score.loc[y.index]

        y_score_np, y_np = y_score.values, y.values
        return scorer(y_np, y_score_np)

    return pd_scorer


class ResultsTracker:
    def __init__(self) -> None:
        self.params = []
        self.score_all_fold_metrics = []
        self.other_all_fold_metrics = []

    def record(
        self,
        params,
        all_fold_scores: List[Dict[str, float]],
        all_fold_other: List[Dict[str, Any]],
    ):
        self.params.append(params)
        self.score_all_fold_metrics.append(all_fold_scores)
        self.other_all_fold_metrics.append(all_fold_other)

    def to_df(self):
        if len(self.params) == 0:
            # We haven't recorded anything, so just return an empty dataframe
            return pd.DataFrame()

        assert (
            len(self.params)
            == len(self.other_all_fold_metrics)
            == len(self.score_all_fold_metrics)
        )

        serialized_params_df = pd.DataFrame(
            [{k: str(v) for k, v in param.items()} for param in self.params]
        )
        split_score_dfs = [
            pd.DataFrame(split_score_metrics)
            for split_score_metrics in zip(*self.score_all_fold_metrics)
        ]
        split_other_dfs = [
            pd.DataFrame(split_other_metrics)
            for split_other_metrics in zip(*self.other_all_fold_metrics)
        ]

        split_score_groups = pd.concat(split_score_dfs, axis=0).groupby(level=0)
        mean_score_df = split_score_groups.mean()
        max_score_df = split_score_groups.max()
        min_score_df = split_score_groups.min()

        score_df = pd.concat(
            [
                df.rename(lambda col: f"{col}_split{i}", axis=1)
                for i, df in enumerate(split_score_dfs)
            ]
            + [
                mean_score_df.rename(lambda col: f"{col}_mean", axis=1),
                max_score_df.rename(lambda col: f"{col}_max", axis=1),
                min_score_df.rename(lambda col: f"{col}_min", axis=1),
            ],
            axis=1,
        )

        # Using the min and max scores, we'll create dfs for the best and worst other metrics
        # This isn't a good way to do this, it assumes that two shapes are the same.
        # Probably a better way to do it is to take advantage of the inherent hierarchy:
        # each score has a score value and other metrics associated with it. A natural
        # way to do this is using multi-index columns.
        assert all(
            split_other_dfs[0].columns.equals(split_other_df.columns)
            for split_other_df in split_other_dfs[1:]
        )
        split_other_score_name_cols = split_other_dfs[0].columns.map(
            lambda s: s.split("__")[0]
        )
        split_corresponding_score_dfs = [
            split_score_df.loc[:, split_other_score_name_cols]
            for split_score_df in split_score_dfs
        ]

        max_corresponding_score_df = max_score_df.loc[:, split_other_score_name_cols]
        curr = split_other_dfs[0]
        for i in range(1, len(split_other_dfs)):
            curr = curr.mask(
                split_corresponding_score_dfs[i] == max_corresponding_score_df,
                other=split_other_dfs[i],
            )
        best_other_dfs = curr

        min_corresponding_score_df = min_score_df.loc[:, split_other_score_name_cols]
        curr = split_other_dfs[0]
        for i in range(1, len(split_other_dfs)):
            curr = curr.mask(
                split_corresponding_score_dfs[i] == min_corresponding_score_df,
                other=split_other_dfs[i],
            )
        worst_other_dfs = curr

        mean_other_dfs = pd.concat(split_other_dfs, axis=0).groupby(level=0).mean()

        other_df = pd.concat(
            [
                df.rename(lambda col: f"{col}_split{i}", axis=1)
                for i, df in enumerate(split_other_dfs)
            ]
            + [
                mean_other_dfs.rename(lambda col: f"{col}_mean", axis=1),
                best_other_dfs.rename(lambda col: f"{col}_best", axis=1),
                worst_other_dfs.rename(lambda col: f"{col}_worst", axis=1),
            ],
            axis=1,
        )

        return pd.concat([serialized_params_df, score_df, other_df], axis=1)


class ObjectiveFunction:
    def __init__(
        self,
        base_model: BaseEstimator,
        y_series,
        scorer,
        watch_score: str,
        n_folds=5,
        random_state=1,
        n_jobs=-1,
        results_csv_fp=None,
        append_to_old_results=False,
        maximize=True,
    ) -> None:
        self.base_model = clone(base_model)
        self.y_series = y_series
        self.base_scorer = scorer
        self.watch_score = watch_score
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.results_csv_fp = results_csv_fp
        self.maximize = maximize
        self.append_to_old_results = append_to_old_results

        if not self.append_to_old_results and self.results_csv_fp is not None and os.path.exists(self.results_csv_fp):
            # Delete the old metrics file
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            os.rename(self.results_csv_fp, f.tag_file(self.results_csv_fp, backup=timestamp))

        # We will call the wrapper on the base_cv_object everytime we need an iterator that spits out the different training and validation sets.
        # Otherwise, the iterator would not restart.
        self.base_cv_object = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )
        self.pd_cv_wrapper = cv_iterator_pd_wrapper(y_series=self.y_series)

        self.scorer = pd_score_wrapper(self.base_scorer)

        self.results_tracker = ResultsTracker()

    def __call__(self, **params) -> float:
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch="2*n_jobs")
        self.base_model.set_params(**params)
        with parallel:
            out = parallel(
                delayed(fit_and_score)(
                    estimator=clone(self.base_model),
                    Xtr_index=Xtr_index,
                    ytr=ytr,
                    Xval_index=Xval_index,
                    yval=yval,
                    scorer=self.scorer,
                )
                for (Xtr_index, ytr), (Xval_index, yval) in self.pd_cv_wrapper(
                    self.base_cv_object
                )
            )

        score_all_fold_metrics, other_all_fold_metrics = zip(*out)
        self.results_tracker.record(
            params, score_all_fold_metrics, other_all_fold_metrics
        )

        # A bit hacky. We should really be building the df incrementally. It's not a bottleneck however,
        # so recreating it over and over doesn't really slow things down that much.
        if self.results_csv_fp is not None:
            self.results_tracker.to_df().iloc[-1:].to_csv(
                self.results_csv_fp,
                index=False,
                mode="a",
                header=not os.path.exists(self.results_csv_fp),
            )

        mean_score = np.mean(
            [
                score_metrics[self.watch_score]
                for score_metrics in score_all_fold_metrics
            ]
        )
        print(f"Score: {mean_score}")
        if self.maximize:
            return -mean_score
        else:
            return mean_score
