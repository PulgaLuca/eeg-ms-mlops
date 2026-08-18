"""XGBoost con sample weight calcolati nel training corrente."""

from typing import Any

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class BalancedXGBClassifier(XGBClassifier):
    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        **kwargs: Any,
    ):
        if sample_weight is None:
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y,
            )

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            **kwargs,
        )