"""Scikit-learn doesn't implement post-processors for pipelines, so we do it ourselves.

TODO: Docs
"""
from dataclasses import asdict
from typing import Any
from sklearn.base import BaseEstimator, ClassifierMixin

from ..constants import (
    Aggregations,
    ThresholdType
)

from ..aggregation_utils import build_agg

class AggregatingPostprocessor(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator: BaseEstimator, aggregation_method: str, aggregation_index: Any = 0):
        
        super().__init__()
        self.estimator = estimator
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
        
        if not isinstance(self.estimator, BaseEstimator):
            raise TypeError(f"The provided input to the estimator argument was not an estimator. You passed {self.estimator}.")
        
        if not hasattr(self.estimator, "decision_function") and not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"The provided estimator must have either a 'decision_function' method or 'predict_proba' method. You provided {self.estimator}")
    
    def fit(self, X, y=None):
        """Fit the inner estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y: array-like, shape (n_samples,)
            Target features.

        Returns
        -------
        self : object
            AggregatingPostprocessor class instance.
        
        TODO: Tests
        """
        # Any input validation is put off to the underlying estimator
        self.estimator.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X using the underlying estimator.

        Assumes
        -------
        The outputs of `self.decision_function` and/or `self.predict_proba` are 1-dimensional.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions. The
            estimator will be used to create continuous valued predictions,
            then the predictions will be aggregated using its index.
            The continuous predictions will then be thresholded by 0.5
            to determine the output prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_individuals,)
            Vector containing the class labels for each unique individual in
            the input data.

        TODO: Tests
        """
        try:
            return (self.decision_function(X) > 0).astype(int)
        except AttributeError:
            return (self.predict_proba(X) > 0.5).astype(int)
    
    def decision_function(self, X): 
        """
        TODO: Docs
        TODO: Tests
        """
        if not hasattr(self.estimator, "decision_function"):
            raise AttributeError(f"You are calling 'decision_function' when the base estimator does not implement this method. Please try 'predict_proba' instead if you would like continuous outputs or ensure your estimator {self.estimator} implements the desired behavior.")
        
        y_sample_scores = self.estimator.decision_function(X)
        y_individual_scores = y_sample_scores.groupby(level=self.aggregation_index).agg(self._aggregation)

        return y_individual_scores
    
    def predict_proba(self, X): 
        """
        TODO: Docs
        TODO: Tests
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"You are calling 'predict_proba' when the base estimator does not implement this method. Please try 'predict_proba' instead if you would like continuous outputs or ensure your estimator {self.estimator} implements the desired behavior.")
        y_sample_scores = self.estimator.predict_proba(X)
        y_individual_scores = y_sample_scores.groupby(level=self.aggregation_index).agg(self._aggregation)

        return y_individual_scores

class ThresholdPostprocessor(ClassifierMixin, BaseEstimator):
    """
    Barebones model that will threshold another scikit-learn model. Only for prediction. 
    If we need more methods, we can add them here.
    """
    def __init__(self, estimator: BaseEstimator, threshold: float, threshold_type: str):
        """Thresholds the output of the inner estimator by the provided threshold.

        Parameters
        ----------
        estimator : BaseEstimator, implements `predict` and one of `predict_proba` or `decision_function`
            The estimator to fit and aggregate predictions from.

        threshold: float
            Above this threshold, the model will predict a 1; below, a 0

        threshold_type: str
            The method of the inner estimator the above threshold applies to. Should be one
            of the values from `ThresholdType`
        
        """
        self.estimator = estimator
        self.threshold = threshold
        self.threshold_type = threshold_type
    
    def decision_function(self, *args, **kwargs):
        """
        TODO: Docs
        TODO: Tests
        """
        return self.estimator.decision_function(*args, **kwargs)
    
    def predict_proba(self, *args, **kwargs):
        """
        TODO: Docs
        TODO: Tests
        """
        return self.estimator.predict_proba(*args, **kwargs)
    
    def predict(self, X, **predict_params):
        """
        TODO: Docs
        TODO: Tests
        """
        if self.threshold_type == ThresholdType.decision_function:
            return (self.estimator.decision_function(X, **predict_params) >= self.threshold).astype(int)
        elif self.threshold_type == ThresholdType.predict_proba:
            return (self.estimator.predict_proba(X, **predict_params) >= self.threshold).astype(int)
        else:
            raise ValueError(f'Unknown threshold type {self.threshold_type}')
    
    def fit(self, X, y=None):
        """Fit the inner estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y: array-like, shape (n_samples,)
            Target features.

        Returns
        -------
        self : object
            ThresholdPostprocessor class instance.
        
        TODO: Tests
        """
        # Any input validation is put off to the underlying estimator
        self.estimator.fit(X, y)
        return self