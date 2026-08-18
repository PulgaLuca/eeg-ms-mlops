"""Esecuzione generica della nested repeated cross-validation."""

from collections.abc import Mapping, Sequence
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from eeg_ms.validation.config import ValidationConfig
from eeg_ms.validation.preprocessing import build_ml_pipeline
from eeg_ms.validation.splitting import (
    iter_outer_splits,
    make_inner_cv,
)


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_predicted = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan

    return {
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(
            y_true,
            y_probability,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_predicted,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_predicted,
            zero_division=0,
        ),
    }


def run_nested_cv(
    subject_features: pd.DataFrame,
    subjects: pd.DataFrame,
    assignments: pd.DataFrame,
    estimator: BaseEstimator,
    parameter_grid: Mapping[str, Sequence],
    config: ValidationConfig,
    *,
    model_name: str,
    scale_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue tuning inner e valutazione outer.

    subject_features deve avere una sola riga per subject_id.
    """

    if subject_features["subject_id"].duplicated().any():
        raise ValueError(
            "subject_features deve contenere una riga per soggetto."
        )

    feature_columns = [
        column
        for column in subject_features.columns
        if column not in {
            "subject_id",
            "group",
            "target",
            "visit",
        }
    ]

    prediction_rows: list[dict] = []
    result_rows: list[dict] = []
    selection_rows: list[dict] = []
    
    for (
        repeat,
        fold,
        train_subjects,
        test_subjects,
    ) in iter_outer_splits(
        subjects=subjects,
        assignments=assignments,
        config=config,
    ):
        train_ids = set(train_subjects["subject_id"])
        test_ids = set(test_subjects["subject_id"])

        train_frame = subject_features.loc[
            subject_features["subject_id"].isin(train_ids)
        ].copy()

        test_frame = subject_features.loc[
            subject_features["subject_id"].isin(test_ids)
        ].copy()

        X_train = train_frame[feature_columns]
        y_train = train_frame["target"].to_numpy()

        X_test = test_frame[feature_columns]
        y_test = test_frame["target"].to_numpy()

        inner_cv = make_inner_cv(
            training_targets=pd.Series(y_train),
            config=config,
            repeat=repeat,
            fold=fold,
        )

        pipeline = build_ml_pipeline(
            estimator=estimator,
            scale_features=scale_features,
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=config.primary_metric,
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        # Imputer, scaler, selector e modello vedono solo X_train.
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        imputer = best_pipeline.named_steps["imputer"]
        selector = best_pipeline.named_steps["selector"]
        fitted_model = best_pipeline.named_steps["model"]

        expanded_feature_names = np.asarray(
            imputer.get_feature_names_out(feature_columns)
        )

        selected_mask = selector.get_support()

        selected_names = expanded_feature_names[
            selected_mask
        ]

        if hasattr(fitted_model, "coef_"):
            raw_importance = np.asarray(
                fitted_model.coef_
            )

            if raw_importance.ndim == 2:
                raw_importance = raw_importance[0]

        elif hasattr(fitted_model, "feature_importances_"):
            raw_importance = np.asarray(
                fitted_model.feature_importances_
            )

        else:
            raw_importance = np.full(
                len(selected_names),
                np.nan,
            )

        for feature_name, importance in zip(
            selected_names,
            raw_importance,
            strict=True,
        ):
            selection_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "feature": feature_name,
                    "importance": float(importance),
                    "absolute_importance": float(abs(importance)),
                }
            )

        probabilities = search.predict_proba(X_test)[:, 1]

        metrics = calculate_binary_metrics(
            y_true=y_test,
            y_probability=probabilities,
        )

        result_rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
                **metrics,
                "best_inner_score": search.best_score_,
                "best_parameters": json.dumps(
                    search.best_params_,
                    sort_keys=True,
                ),
            }
        )

        for subject_id, target, probability in zip(
            test_frame["subject_id"],
            y_test,
            probabilities,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "subject_id": subject_id,
                    "target": int(target),
                    "probability_ms": float(probability),
                }
            )

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selection_rows),
    )