"""Metriche e aggregazione delle predizioni out-of-fold.
Interpretazione
probability_std elevata: previsione instabile rispetto al training set;
probabilità vicina a 0.5: classificazione incerta;
false_negative_ms: soggetto MS classificato HC;
false_positive_hc: soggetto HC classificato MS;
Brier score basso: probabilità più vicine ai target osservati.

La soglia 0.5 è provvisoria. Non dobbiamo scegliere una soglia diversa massimizzando sensibilità o specificità sulle predizioni 
outer aggregate. Una soglia ottimizzata deve essere stimata dentro l’inner CV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


METRIC_COLUMNS = [
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
]


def calculate_binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcola metriche probabilistiche e threshold-based."""

    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else np.nan
    )

    return {
        "roc_auc": roc_auc_score(
            targets,
            probabilities,
        ),
        "average_precision": average_precision_score(
            targets,
            probabilities,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            targets,
            predictions,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(
            targets,
            predictions,
            zero_division=0,
        ),
        "brier_score": brier_score_loss(
            targets,
            probabilities,
        ),
    }


def aggregate_oof_predictions(
    predictions: pd.DataFrame,
    *,
    expected_repeats: int,
) -> pd.DataFrame:
    """
    Riduce le predizioni ripetute a una riga per soggetto e modello.
    """

    required_columns = {
        "model",
        "repeat",
        "fold",
        "subject_id",
        "target",
        "probability_ms",
    }

    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Colonne predizioni mancanti: {sorted(missing)}"
        )

    duplicated = predictions.duplicated(
        ["model", "repeat", "subject_id"]
    )

    if duplicated.any():
        raise ValueError(
            "Un soggetto presenta più predizioni nella stessa "
            "ripetizione."
        )

    target_consistency = (
        predictions
        .groupby(["model", "subject_id"])["target"]
        .nunique()
    )

    if (target_consistency != 1).any():
        raise ValueError(
            "Target incoerenti fra le ripetizioni."
        )

    aggregated = (
        predictions
        .groupby(
            ["model", "subject_id", "target"],
            as_index=False,
        )
        .agg(
            probability_ms=(
                "probability_ms",
                "mean",
            ),
            probability_std=(
                "probability_ms",
                "std",
            ),
            probability_min=(
                "probability_ms",
                "min",
            ),
            probability_max=(
                "probability_ms",
                "max",
            ),
            n_predictions=(
                "probability_ms",
                "size",
            ),
        )
    )

    invalid_counts = (
        aggregated["n_predictions"] != expected_repeats
    )

    if invalid_counts.any():
        examples = aggregated.loc[
            invalid_counts,
            ["model", "subject_id", "n_predictions"],
        ]

        raise ValueError(
            "Numero inatteso di predizioni OOF:\n"
            f"{examples.to_string(index=False)}"
        )

    aggregated["predicted_target"] = (
        aggregated["probability_ms"] >= 0.5
    ).astype(int)

    aggregated["correct"] = (
        aggregated["predicted_target"]
        == aggregated["target"]
    )

    aggregated["distance_from_threshold"] = (
        aggregated["probability_ms"] - 0.5
    ).abs()

    return aggregated


def summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Riassume la variabilità osservata negli outer fold."""

    available_metrics = [
        column
        for column in METRIC_COLUMNS
        if column in fold_metrics.columns
    ]

    summary = (
        fold_metrics
        .groupby("model")[available_metrics]
        .agg(["mean", "std", "median", "min", "max"])
    )

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    return summary.reset_index()


def calculate_aggregated_oof_metrics(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    """Calcola le metriche sulle 32 predizioni aggregate."""

    rows: list[dict] = []

    for model, model_frame in aggregated.groupby("model"):
        metrics = calculate_binary_metrics(
            targets=model_frame["target"].to_numpy(),
            probabilities=(
                model_frame["probability_ms"].to_numpy()
            ),
        )

        rows.append(
            {
                "model": model,
                "n_subjects": model_frame["subject_id"].nunique(),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def build_error_analysis(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    """Costruisce una tabella per l'ispezione degli errori."""

    result = aggregated.copy()

    result["error_type"] = np.select(
        [
            (result["target"] == 1)
            & (result["predicted_target"] == 0),

            (result["target"] == 0)
            & (result["predicted_target"] == 1),
        ],
        [
            "false_negative_ms",
            "false_positive_hc",
        ],
        default="correct",
    )

    return result.sort_values(
        [
            "model",
            "correct",
            "distance_from_threshold",
        ],
        ascending=[True, True, True],
    )