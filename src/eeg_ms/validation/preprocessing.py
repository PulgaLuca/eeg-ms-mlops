"""Pipeline di preprocessing appresa soltanto sui training fold."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ml_pipeline(
    estimator: BaseEstimator,
    *,
    scale_features: bool,
) -> Pipeline:
    scaler: Any = (
        StandardScaler()
        if scale_features
        else "passthrough"
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                scaler,
            ),
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=10,
                ),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )