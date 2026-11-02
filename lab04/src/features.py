"""Preprocessing transformers for Lab 4. They live in their own module so that
results/preprocessor.joblib can be loaded anywhere this module is importable.

build_preprocessor():  '?' -> NaN -> [numeric: median impute -> clip to train 1st/99th pct]
                                    [categorical: mode impute]
                        -> optional derived features (capital_net, hours_per_age)
Every statistic (median, mode, percentiles) is learned in fit(), i.e. from train only.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def marker_to_nan(X, marker="?"):
    return X.replace(marker, np.nan)


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip every column to [p_low, p_high] percentiles learned in fit()."""

    def __init__(self, percentiles=(1, 99)):
        self.percentiles = percentiles

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.lower_, self.upper_ = np.percentile(X, self.percentiles, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """capital_net = capital-gain - capital-loss; hours_per_age = hours-per-week / (age - 16).
    Stateless; age >= 17 by schema, so the denominator is at least 1."""

    NAMES = ["capital_net", "hours_per_age"]

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        X["capital_net"] = X["capital-gain"] - X["capital-loss"]
        X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + self.NAMES, dtype=object)


def build_preprocessor(numeric, categorical, cfg, derived=True):
    c = cfg["cleaning"]
    columns = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy=c["numeric_impute"])),
                          ("clip", PercentileClipper(tuple(c["clip_percentiles"])))]), numeric),
        ("cat", SimpleImputer(strategy=c["categorical_impute"]), categorical),
    ], verbose_feature_names_out=False)
    steps = [("missing", FunctionTransformer(marker_to_nan, kw_args={"marker": cfg["missing_marker"]},
                                             feature_names_out="one-to-one")),
             ("columns", columns)]
    if derived:
        steps.append(("derived", DerivedFeatures()))
    return Pipeline(steps).set_output(transform="pandas")


def fitted_stats(pre):
    """Medians, modes and clip bounds learned by a fitted preprocessor, one row per column."""
    ct = pre.named_steps["columns"]
    num_pipe, num_cols = ct.named_transformers_["num"], ct.transformers_[0][2]
    cat_imp, cat_cols = ct.named_transformers_["cat"], ct.transformers_[1][2]
    clip = num_pipe.named_steps["clip"]
    rows = [{"column": col, "kind": "numeric", "impute": num_pipe.named_steps["impute"].statistics_[i],
             "clip_low": clip.lower_[i], "clip_high": clip.upper_[i]} for i, col in enumerate(num_cols)]
    rows += [{"column": col, "kind": "categorical", "impute": cat_imp.statistics_[i]}
             for i, col in enumerate(cat_cols)]
    return pd.DataFrame(rows)
