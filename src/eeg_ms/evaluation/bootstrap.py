"""Intervalli di confidenza bootstrap a livello soggetto."""

import numpy as np
import pandas as pd

from eeg_ms.evaluation.metrics import (
    calculate_binary_metrics,
)


def stratified_bootstrap_indices(
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ricampiona separatamente HC e MS.

    Questo mantiene entrambe le classi in ogni campione bootstrap.
    """

    negative_indices = np.flatnonzero(targets == 0)
    positive_indices = np.flatnonzero(targets == 1)

    sampled_negative = rng.choice(
        negative_indices,
        size=len(negative_indices),
        replace=True,
    )

    sampled_positive = rng.choice(
        positive_indices,
        size=len(positive_indices),
        replace=True,
    )

    return np.concatenate(
        [sampled_negative, sampled_positive]
    )


def bootstrap_confidence_intervals(
    aggregated_predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 5_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Calcola intervalli bootstrap per ogni modello."""

    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    for model, model_frame in (aggregated_predictions.groupby("model")):
        model_frame = model_frame.reset_index(drop=True)

        targets = model_frame["target"].to_numpy()
        probabilities = (
            model_frame["probability_ms"].to_numpy()
        )

        point_metrics = calculate_binary_metrics(
            targets,
            probabilities,
        )

        bootstrap_values = {
            metric: []
            for metric in point_metrics
        }

        for _ in range(n_bootstrap):
            indices = stratified_bootstrap_indices(targets, rng,)

            sample_metrics = calculate_binary_metrics(
                targets=targets[indices],
                probabilities=probabilities[indices],
            )

            for metric, value in sample_metrics.items():
                bootstrap_values[metric].append(value)

        for metric, values in bootstrap_values.items():
            values_array = np.asarray(values, dtype=float)

            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": point_metrics[metric],
                    "ci_lower": np.nanquantile(
                        values_array,
                        lower_quantile,
                    ),
                    "ci_upper": np.nanquantile(
                        values_array,
                        upper_quantile,
                    ),
                    "n_bootstrap": n_bootstrap,
                    "confidence_level": confidence_level,
                }
            )

    return pd.DataFrame(rows)