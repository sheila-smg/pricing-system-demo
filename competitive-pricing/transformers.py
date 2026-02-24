from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from xgboost import XGBRegressor


class FeatureSelectorValues(BaseEstimator, TransformerMixin):
    """Select a subset of columns and return as a numpy array."""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = [c for c in self.feature_names if c in X.columns]
        return X[cols].values


class NeverSeenToNanEncoder(BaseEstimator, TransformerMixin):
    """
    Replace any category value not seen during fit with NaN.
    Prevents target-encoder or OHE from being surprised by unseen categories
    at prediction time — a common source of silent errors in production.
    """

    def __init__(self, col=None):
        self.col = col
        self.mapping = None

    def fit(self, X, y=None):
        self.mapping = set(X[self.col].dropna().unique())
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.loc[~X[self.col].isin(self.mapping), self.col] = np.nan
        X[self.col] = X[self.col].astype("O")
        return X[self.col].values.reshape(-1, 1)


class MonotonicQuantileXGBRegressor(BaseEstimator, TransformerMixin):
    """
    Wraps XGBRegressor to predict multiple quantiles simultaneously.

    XGBoost's `reg:quantileerror` objective accepts a vector of quantile
    levels (`quantile_alpha`) and outputs one prediction column per quantile.
    After prediction, columns are sorted to enforce monotonicity across
    quantile levels — ensuring q10 <= q20 <= ... <= q90.
    """

    def __init__(self, estimator=XGBRegressor, **kwargs):
        self.estimator = estimator(**kwargs)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.is_fitted_ = True  # required for sklearn >= 1.4 check_is_fitted
        return self

    def predict(self, X):
        preds = self.estimator.predict(X)
        preds.sort(axis=1)
        return preds

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **parameters):
        self.estimator.set_params(**parameters)
        return self
