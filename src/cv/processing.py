from dataclasses import asdict
from multiprocessing.sharedctypes import Value
from textwrap import indent
from typing import Any
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.base import TransformerMixin, BaseEstimator

from ..constants import (
    Aggregations,
    UPN
)

from ..aggregation_utils import build_agg

class DataJoinerTransformerFunc:
    def __init__(self, data_df: pd.DataFrame) -> None:
        self.data_df = data_df
    
    def __call__(self, X_index: pd.Index, y_series=None) -> Any:
        """
        Parameters
        ----------
        X_index: pd.Index
            This is the index of items we want to extract from the
            data_df. In our case, it is the students we want to extract.
            This method will then extract all entries corresponding to that
            student from the data_df
        y_series: pd.Series, optional
            These are the labels for each index item. In our case, these are
            the outcomes for each student we are trying to predict.
        """
        if len(X_index.names) != 1:
            raise ValueError(f"X_index should be only a single index. Found index with multiple hierarchis {X_index.names}.")
        index_name = X_index.names[0]
        if index_name not in self.data_df.index.names:
            raise ValueError(f"X_index should have the same name as one of the indexes of data_df. {index_name} was not in {self.data_df.index.names}.")

        new_X = self.data_df.loc[X_index]
        if y_series is not None:
            if index_name not in y_series.index.names:
                raise ValueError(f"X_index should have the same name as one of the indexes of y_series. {index_name} was not in {y_series.index.names}.")

            new_y = y_series.loc[new_X.index.get_level_values(index_name)]
            return new_X, new_y
        return new_X

class AggregatorTransformerFunc(BaseEstimator):

    def __init__(self, aggregation_method: str = Aggregations.mean, aggregation_index: Any = 0) -> None:
        """
        A post-processor that aggregates using the index of its input as group keys.

        Parameters
        ----------
        aggregation_method: str, optional
            One of the methods from `Aggregations`. It specifies how to aggregate. By default,
            it is 'mean'.
        aggregation_index : Any, optional
            The name of the index column of the input data to aggregate over. This is
            passed to the `level` parameter of a pandas `groupby`. By default, it is 0,
            so it will aggregate over the first index of the provided dataframe.
        """
        self.aggregation_method = aggregation_method
        self.aggregation_index = aggregation_index
        self._validate_params()

        self._aggregation = build_agg(self.aggregation_method)
    
    def _validate_params(self):
        allowed_agg_methods = asdict(Aggregations).values()
        if self.aggregation_method not in allowed_agg_methods:
            raise ValueError(f"The aggregation method must be one of {set(allowed_agg_methods)}. You passed '{self.aggregation_method}'.")
        if self.aggregation_method == Aggregations.categorical_max:
            raise ValueError(f"The '{Aggregations.categorical_max}' aggregation is not valid in the {type(self).__name__}. Please provide a different aggregation method.")

    def __call__(self, data):
        """Aggregate the data accoring to the aggregation_method.

        Parameters
        ----------
        data: pd.DataFrame, pd.Series
            The data to aggregate
        
        Returns
        -------
        pd.Series, or a scalar
            The data aggregated over the aggregation_index according to the aggregation_method.
        """
        return data.groupby(level=self.aggregation_index).agg(self._aggregation)

def identity_preprocessor(X, labels=None):
    if labels is None:
        return X
    else:
        return X, labels

def identity_postprocessor(X):
    return X

class PandasEstimatorWrapper(TransformerMixin, BaseEstimator):
    """
    This wrapper is designed to allow wrapping a scikit-learn
    estimator to handle pandas dataframe inputs. It will pass 
    scikit-learn a numpy array, and store the pandas index to 
    the side. During any of the predict or transform methods, it 
    will put the index back on the output of the estimator.

    It also allows for the use of preprocessing and postprocessing
    functions. For example, you might want to use an aggregator
    to aggregate the outputs of a model over the same values
    of an index (e.g. different years for the same student).

    Currently only supports binary classification.
    """

    def __init__(self, estimator, preprocessor=identity_preprocessor, postprocessor=identity_postprocessor) -> None:
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def fit(self, X_index: pd.Index, y: pd.Series):
        X, y = self.preprocessor(X_index, y_series=y)
        X_np, y_np = X.values, y.values
        self.estimator.fit(X_np, y_np)

        return self
    
    def transform(self, X_index):
        X = self.preprocessor(X_index)
        X_np = X.values
        Xt_np = self.estimator.transform(X_np)
        Xt = pd.DataFrame(Xt_np, index=X.index)
        Xt = self.postprocessor(Xt)
        return Xt

    def predict(self, X_index):
        try:
            return (self.decision_function(X_index) > 0).astype(int)
        except AttributeError: 
            return (self.predict_proba(X_index) > 0.5).astype(int)

    @staticmethod
    def check_binary_prediction(y_pred_np: np.ndarray):
        if y_pred_np.ndim == 2:
            if y_pred_np.shape[1] == 1:
                y_pred_np = y_pred_np[:, 0]
            elif y_pred_np.shape[1] == 2:
                y_pred_np = y_pred_np[:, 1]
            else:
                raise ValueError(f"If y_pred is 2 dimensional it should have either 1 or 2 columns. Got shape {y_pred_np.shape} instead. Note that multi-label classification is not supported.")
        elif y_pred_np.ndim != 1:
            raise ValueError(f"y_pred should be either 1 or 2 dimensional. Got shape {y_pred_np.shape} instead.")
        
        return y_pred_np

    def decision_function(self, X_index):
        X = self.preprocessor(X_index)
        X_np = X.values
        y_pred_np =  self.estimator.decision_function(X_np)
        # We've assumed binary classification, so that we can safely only return the output for one class
        y_pred_np = self.check_binary_prediction(y_pred_np)
        y_pred = pd.Series(y_pred_np, index=X.index)
        y_pred = self.postprocessor(y_pred)
        return y_pred
    
    def predict_proba(self, X_index):
        X = self.preprocessor(X_index)
        X_np = X.values
        y_pred_np =  self.estimator.predict_proba(X_np)
        # We've assumed binary classification, so that we can safely only return the output for one class
        y_pred_np = self.check_binary_prediction(y_pred_np)
        y_pred = pd.Series(y_pred_np, index=X.index)
        y_pred = self.postprocessor(y_pred)
        return y_pred
