"""Confronto dei modelli mediante nested repeated CV."""

import pandas as pd

from eeg_ms.config import (
    METRICS_DIR,
    OUTER_FOLDS_FILE,
    PREDICTIONS_DIR,
    SUBJECT_FEATURES_ROI,
    SUBJECTS_DATA,
    SELECTED_FEATURES_FILE
)
from eeg_ms.modeling.models import (
    get_model_specifications,
)
from eeg_ms.validation.config import (
    load_validation_config,
)
from eeg_ms.validation.nested_cv import run_nested_cv


def main() -> None:
    config = load_validation_config()

    subject_features = pd.read_parquet(
        SUBJECT_FEATURES_ROI
    )

    subjects = pd.read_csv(
        SUBJECTS_DATA
    )

    assignments = pd.read_csv(
        OUTER_FOLDS_FILE
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_selected_features: list[pd.DataFrame] = []

    specifications = get_model_specifications(
        random_seed=config.random_seed
    )

    for specification in specifications:
        print(f"Avvio modello: {specification.name}")

        metrics, predictions, selected_features = run_nested_cv(
            subject_features=subject_features,
            subjects=subjects,
            assignments=assignments,
            estimator=specification.estimator,
            parameter_grid=specification.parameter_grid,
            config=config,
            model_name=specification.name,
            scale_features=specification.scale_features,
        )

        all_selected_features.append(selected_features)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics.to_csv(
        METRICS_DIR / "nested_cv_fold_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_DIR / "nested_cv_predictions.csv",
        index=False,
    )

    metric_columns = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]

    summary = (
        metrics
        .groupby("model")[metric_columns]
        .agg(["mean", "std", "median"])
        .sort_values(
            ("balanced_accuracy", "mean"),
            ascending=False,
        )
    )

    summary.to_csv(
        METRICS_DIR / "model_comparison_summary.csv"
    )

    selected_features = pd.concat(
        all_selected_features,
        ignore_index=True,
    )

    SELECTED_FEATURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_features.to_csv(
        SELECTED_FEATURES_FILE,
        index=False,
    )

    print("\nConfronto modelli:")
    print(summary.to_string())


if __name__ == "__main__":
    main()