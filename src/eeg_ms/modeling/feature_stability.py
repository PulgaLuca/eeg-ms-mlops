import pandas as pd

def calculate_feature_stability(
    selected_features: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    n_fits = (
        fold_metrics
        .groupby("model")
        .size()
        .rename("n_outer_fits")
        .reset_index()
    )

    stability = (
        selected_features
        .groupby(["model", "feature"])
        .agg(
            n_selected=("feature", "size"),
            mean_absolute_importance=(
                "absolute_importance",
                "mean",
            ),
            median_absolute_importance=(
                "absolute_importance",
                "median",
            ),
        )
        .reset_index()
        .merge(
            n_fits,
            on="model",
            how="left",
        )
    )

    stability["selection_frequency"] = (
        stability["n_selected"]
        / stability["n_outer_fits"]
    )

    return stability.sort_values(
        [
            "model",
            "selection_frequency",
            "mean_absolute_importance",
        ],
        ascending=[True, False, False],
    )