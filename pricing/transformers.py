"""
Custom sklearn transformers for insurance pricing pipelines.

These transformers handle feature-specific preprocessing steps that require
domain knowledge, such as computing age from date of birth or deriving vehicle
age from model year. They follow the sklearn Estimator API and are designed to
be composed within sklearn Pipeline and FeatureUnion objects.
"""

from datetime import date

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelectorValues(BaseEstimator, TransformerMixin):
    """
    Select a subset of columns from a DataFrame and return a numpy array.

    Used within FeatureUnion sub-pipelines to route specific features to
    their appropriate preprocessing steps before recombining.

    Parameters
    ----------
    feature_names : list of str
        Column names to select from the input DataFrame.
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        available = [c for c in self.feature_names if c in X.columns]
        return X[available].values


class ComputeAgeFromDOB(BaseEstimator, TransformerMixin):
    """
    Compute policyholder age (in years) from a date-of-birth column.

    feature_1 stores date of birth as a string (YYYY-MM-DD format).
    Missing DOBs are propagated as NaN for downstream imputation.

    Feature treatment rationale:
        Age is one of the most significant rating factors in motor insurance.
        - Younger policyholders tend to exhibit higher claim frequency due to
          limited driving experience (risk increases sharply below ~25).
        - The relationship is non-linear (often U-shaped), with elevated risk
          at both extremes of the age distribution.
        - Computing age at prediction time (vs. storing it statically) ensures
          consistency as the portfolio ages between training and scoring.

    Parameters
    ----------
    dob_column : str
        Name of the date-of-birth column in the input DataFrame.
    reference_date : date or None
        Date from which to measure age. Defaults to today.
    """

    def __init__(self, dob_column="feature_1", reference_date=None):
        self.dob_column = dob_column
        self.reference_date = reference_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ref = pd.Timestamp(self.reference_date or date.today())
        dob = pd.to_datetime(X[self.dob_column], errors="coerce")
        age = (ref - dob).dt.days / 365.25
        return age.values.reshape(-1, 1)


class ComputeVehicleAge(BaseEstimator, TransformerMixin):
    """
    Compute vehicle age (in years) from the vehicle model year column.

    feature_2 stores the vehicle's model year as an integer (e.g., 2018).
    Vehicle age is computed as the difference from the reference year.

    Feature treatment rationale:
        Vehicle age has a dual and opposing effect on insurance risk:
        - Older vehicles tend toward higher claim *frequency* due to mechanical
          degradation and the absence of modern active safety systems (ABS,
          ESC, automatic emergency braking).
        - Claim *severity* often decreases with vehicle age, since repair and
          replacement costs are lower for depreciated vehicles.
        Computing vehicle age at prediction time prevents data drift when the
        same model year cohort is scored across successive underwriting years.

    Parameters
    ----------
    year_column : str
        Name of the model-year column in the input DataFrame.
    reference_year : int or None
        Year from which to measure vehicle age. Defaults to current year.
    """

    def __init__(self, year_column="feature_2", reference_year=None):
        self.year_column = year_column
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ref_year = self.reference_year or date.today().year
        vehicle_age = ref_year - X[self.year_column]
        return vehicle_age.values.reshape(-1, 1)


class FrozenTransformer(BaseEstimator, TransformerMixin):
    """
    Wrap a pre-fitted estimator as a sklearn Transformer.

    The wrapped estimator is never retrained — it is "frozen". During
    transform(), it generates predictions from the input data, which
    become a new feature column fed into the downstream model.

    This enables a transfer-learning / model-stacking pattern within a
    standard sklearn Pipeline + FeatureUnion:

        FeatureUnion([
            ("base_model_predictions", Pipeline([
                ("frozen", FrozenTransformer(pretrained_model)),
                ("imputer", SimpleImputer()),
            ])),
            ("age",      age_pipeline),
            ("region",   region_pipeline),
            ...
        ])

    The base model provides a prior estimate of risk derived from a
    reference dataset (e.g., industry-wide or regulatory data). The
    downstream model then learns a residual correction using proprietary
    features not available in the reference dataset.

    Parameters
    ----------
    fitted_estimator : object with a .predict() method
        A pre-fitted sklearn-compatible estimator or pipeline.
    """

    def __init__(self, fitted_estimator):
        self.fitted_estimator = fitted_estimator

    def __sklearn_clone__(self):
        # Cloning returns the same instance — the estimator must not be retrained
        return self

    def fit(self, X, y=None):
        # No-op: the estimator is frozen
        return self

    def transform(self, X):
        predictions = self.fitted_estimator.predict(X)
        return predictions.reshape(-1, 1)
