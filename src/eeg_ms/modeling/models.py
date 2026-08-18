"""Definizione centralizzata dei modelli e degli spazi di ricerca."""

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from eeg_ms.modeling.xgboost_balanced import (
    BalancedXGBClassifier,
)


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]
    scale_features: bool


def get_model_specifications(
    random_seed: int,
) -> list[ModelSpecification]:
    """Restituisce baseline e modelli da confrontare."""

    return [
        ModelSpecification(
            name="dummy",
            estimator=DummyClassifier(
                strategy="prior",
            ),
            parameter_grid={
                "selector__k": ["all"],
            },
            scale_features=False,
        ),

        ModelSpecification(
            name="logistic_elastic_net",
            estimator=LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=20_000,
                random_state=random_seed,
            ),
            parameter_grid={
                "selector__k": [5, 10, 20],
                "model__C": [0.1, 1.0, 10.0],
                "model__l1_ratio": [0.25, 0.5, 0.75],
            },
            scale_features=True,
        ),

        ModelSpecification(
            name="lda",
            estimator=LinearDiscriminantAnalysis(
                solver="lsqr",
            ),
            parameter_grid={
                "selector__k": [5, 10, 20],
                "model__shrinkage": [
                    "auto",
                    0.1,
                    0.5,
                    0.9,
                ],
            },
            scale_features=True,
        ),

        ModelSpecification(
            name="knn",
            estimator=KNeighborsClassifier(),
            parameter_grid={
                "selector__k": [5, 10, 20],
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": [
                    "uniform",
                    "distance",
                ],
                "model__p": [1, 2],
            },
            scale_features=True,
        ),

        # ModelSpecification(
        #     name="random_forest",
        #     estimator=RandomForestClassifier(
        #         n_estimators=500,
        #         class_weight="balanced",
        #         random_state=random_seed,
        #         n_jobs=1,
        #     ),
        #     parameter_grid={
        #         "selector__k": [10, 20, "all"],
        #         "model__max_depth": [2, 4, None],
        #         "model__min_samples_leaf": [1, 2, 4],
        #         "model__max_features": [
        #             "sqrt",
        #             0.5,
        #         ],
        #     },
        #     scale_features=False,
        # ),

        ModelSpecification(
            name="xgboost",
            estimator=BalancedXGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_seed,
                n_jobs=1,
            ),
            parameter_grid={
                "selector__k": [10, 20, "all"],
                "model__n_estimators": [100, 300],
                "model__max_depth": [1, 2],
                "model__learning_rate": [0.03, 0.1],
                "model__min_child_weight": [1, 3],
            },
            scale_features=False,
        ),
    ]