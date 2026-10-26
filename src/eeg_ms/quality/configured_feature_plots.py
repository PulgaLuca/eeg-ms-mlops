"""Run-specific plots describing the feature matrix selected by the user."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eeg_ms.quality.plots import configure_style


METADATA_COLUMNS = {"subject_id", "group", "target", "visit", "window"}
FEATURE_NAME_PARTS = [
    "feature_family",
    "measure",
    "band",
    "spatial_level",
    "location",
    "statistic",
    "condition_view",
]


def _feature_catalog(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in feature_matrix if column not in METADATA_COLUMNS]
    records = []
    for column in feature_columns:
        parts = str(column).split("__")
        if len(parts) == len(FEATURE_NAME_PARTS):
            records.append({"feature_name": column, **dict(zip(FEATURE_NAME_PARTS, parts, strict=True))})
    if not records:
        raise ValueError("Non riesco a ricavare i descrittori dai nomi delle feature selezionate.")
    return pd.DataFrame(records)


def plot_configured_feature_coverage(
    feature_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot how many selected ML columns come from each family/measure/location scale."""
    configure_style()
    catalog = _feature_catalog(feature_matrix)
    catalog["feature_set"] = (
        catalog["feature_family"] + " · " + catalog["measure"] + " · " + catalog["spatial_level"]
    )
    counts = catalog["feature_set"].value_counts().sort_values()
    figure_height = max(4.5, 0.38 * len(counts))
    figure, axis = plt.subplots(figsize=(11, figure_height))
    sns.barplot(x=counts.values, y=counts.index, color="#4c78a8", ax=axis)
    axis.set(
        title="Feature incluse nel run per famiglia, misura e livello spaziale",
        xlabel="Numero di colonne della matrice ML",
        ylabel="",
    )
    for container in axis.containers:
        axis.bar_label(container, padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_configured_feature_examples(
    feature_matrix: pd.DataFrame,
    output_path: Path,
    *,
    max_features: int = 8,
) -> list[str]:
    """Show deterministic examples from the selected features at subject level.

    In epoch mode, window rows are averaged within subject for this descriptive plot;
    they are never treated as independent people in the visualization.
    """
    configure_style()
    catalog = _feature_catalog(feature_matrix)
    strata = ["feature_family", "measure", "band", "spatial_level"]
    representatives = catalog.sort_values(strata + ["location", "statistic", "condition_view"])
    representatives = representatives.drop_duplicates(strata).head(max_features)
    selected_names = representatives["feature_name"].tolist()

    values = feature_matrix[["subject_id", "group", *selected_names]].copy()
    if "window" in feature_matrix.columns:
        values = values.groupby(["subject_id", "group"], as_index=False, observed=True)[selected_names].mean()
    long = values.melt(
        id_vars=["subject_id", "group"],
        value_vars=selected_names,
        var_name="feature_name",
        value_name="value",
    )

    n_columns = 2
    n_rows = math.ceil(len(selected_names) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(13, 3.7 * n_rows),
        squeeze=False,
    )
    for axis, feature_name in zip(axes.flat, selected_names, strict=False):
        subset = long.loc[long["feature_name"] == feature_name]
        sns.boxplot(
            data=subset,
            x="group",
            y="value",
            order=["hc", "ms"],
            showfliers=False,
            color="#c7d8ea",
            ax=axis,
        )
        sns.stripplot(
            data=subset,
            x="group",
            y="value",
            order=["hc", "ms"],
            hue="group",
            hue_order=["hc", "ms"],
            dodge=False,
            jitter=0.16,
            alpha=0.75,
            size=4,
            legend=False,
            ax=axis,
        )
        axis.set_title(feature_name.replace("__", " · "), fontsize=8, wrap=True)
        axis.set_xlabel("Gruppo")
        axis.set_ylabel("Valore per soggetto")

    for axis in axes.flat[len(selected_names):]:
        axis.set_visible(False)

    level_note = "media delle epoche per soggetto" if "window" in feature_matrix.columns else "valori subject-level"
    figure.suptitle(
        f"Esempi descrittivi delle feature selezionate · {level_note} · "
        "non sono le feature selezionate dal modello nei singoli fold",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return selected_names


def create_configured_feature_plots(
    feature_matrix: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    """Create and return paths of the run-specific selected-feature plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = figures_dir / "configured_feature_coverage.png"
    examples_path = figures_dir / "configured_feature_examples.png"
    plot_configured_feature_coverage(feature_matrix, coverage_path)
    examples = plot_configured_feature_examples(feature_matrix, examples_path)
    return [coverage_path.name, examples_path.name, *examples]
