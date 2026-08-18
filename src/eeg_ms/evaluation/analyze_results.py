"""Analisi controllata dei risultati nested CV."""

import pandas as pd

from eeg_ms.config import (
    EVALUATION_FIGURES,
    EVALUATION_TABLES,
    FOLD_METRICS_FILE,
    OOF_PREDICTIONS_FILE,
    SELECTED_FEATURES_FILE,
)
from eeg_ms.evaluation.bootstrap import (
    bootstrap_confidence_intervals,
)
from eeg_ms.evaluation.metrics import (
    aggregate_oof_predictions,
    build_error_analysis,
    calculate_aggregated_oof_metrics,
    summarize_fold_metrics,
)
from eeg_ms.evaluation.plots import (
    plot_bootstrap_intervals,
    plot_confusion_matrices,
    plot_fold_metric_distributions,
    plot_oof_precision_recall_curves,
    plot_oof_roc_curves,
    plot_subject_probabilities,
)
from eeg_ms.modeling.feature_stability import (
    calculate_feature_stability,
)
from eeg_ms.validation.config import (
    load_validation_config,
)


def main() -> None:
    config = load_validation_config()

    fold_metrics = pd.read_csv(
        FOLD_METRICS_FILE
    )

    predictions = pd.read_csv(
        OOF_PREDICTIONS_FILE
    )

    EVALUATION_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated = aggregate_oof_predictions(
        predictions,
        expected_repeats=config.outer_n_repeats,
    )

    fold_summary = summarize_fold_metrics(
        fold_metrics
    )

    oof_metrics = calculate_aggregated_oof_metrics(
        aggregated
    )

    error_analysis = build_error_analysis(
        aggregated
    )

    intervals = bootstrap_confidence_intervals(
        aggregated_predictions=aggregated,
        n_bootstrap=5_000,
        confidence_level=0.95,
        random_seed=config.random_seed,
    )

    fold_summary.to_csv(
        EVALUATION_TABLES / "fold_metric_summary.csv",
        index=False,
    )

    oof_metrics.to_csv(
        EVALUATION_TABLES / "aggregated_oof_metrics.csv",
        index=False,
    )

    aggregated.to_csv(
        EVALUATION_TABLES / "subject_oof_predictions.csv",
        index=False,
    )

    error_analysis.to_csv(
        EVALUATION_TABLES / "subject_error_analysis.csv",
        index=False,
    )

    intervals.to_csv(
        EVALUATION_TABLES
        / "bootstrap_confidence_intervals.csv",
        index=False,
    )

    if SELECTED_FEATURES_FILE.exists():
        selected_features = pd.read_csv(
            SELECTED_FEATURES_FILE
        )

        stability = calculate_feature_stability(
            selected_features=selected_features,
            fold_metrics=fold_metrics,
        )

        stability.to_csv(
            EVALUATION_TABLES / "feature_stability.csv",
            index=False,
        )

    plot_fold_metric_distributions(
        fold_metrics,
        EVALUATION_FIGURES
        / "fold_metric_distributions.png",
    )

    plot_oof_roc_curves(
        aggregated,
        EVALUATION_FIGURES / "oof_roc_curves.png",
    )

    plot_oof_precision_recall_curves(
        aggregated,
        EVALUATION_FIGURES
        / "oof_precision_recall_curves.png",
    )

    plot_confusion_matrices(
        aggregated,
        EVALUATION_FIGURES / "confusion_matrices.png",
    )

    plot_subject_probabilities(
        aggregated,
        EVALUATION_FIGURES / "subject_probabilities.png",
    )

    plot_bootstrap_intervals(
        intervals,
        EVALUATION_FIGURES
        / "bootstrap_balanced_accuracy.png",
        metric="balanced_accuracy",
    )

    print("\nMetriche OOF aggregate:")
    print(
        oof_metrics
        .sort_values(
            "balanced_accuracy",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()