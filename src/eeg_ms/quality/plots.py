"""Grafici riproducibili per EDA e controllo qualità."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def configure_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
    )


def plot_group_balance(
    subjects: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()

    figure, axis = plt.subplots(figsize=(5, 4))

    sns.countplot(
        data=subjects,
        x="group",
        order=["hc", "ms"],
        ax=axis,
    )

    axis.set(
        title="Numero di soggetti per gruppo",
        xlabel="Gruppo",
        ylabel="Soggetti",
    )

    for container in axis.containers:
        axis.bar_label(container)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_subject_distributions(
    subject_summary: pd.DataFrame,
    *,
    measure: str,
    band: str,
    spatial_level: str,
    condition: str,
    output_path: Path,
) -> None:
    """Ogni punto mostrato rappresenta un soggetto."""

    configure_style()

    subset = subject_summary.loc[
        (subject_summary["measure"] == measure)
        & (subject_summary["band"] == band)
        & (subject_summary["spatial_level"] == spatial_level)
        & (subject_summary["condition"] == condition)
    ]

    figure, axis = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=subset,
        x="location",
        y="window_median",
        hue="group",
        hue_order=["hc", "ms"],
        showfliers=False,
        ax=axis,
    )

    sns.stripplot(
        data=subset,
        x="location",
        y="window_median",
        hue="group",
        hue_order=["hc", "ms"],
        dodge=True,
        alpha=0.65,
        size=3,
        legend=False,
        ax=axis,
    )

    axis.set(
        title=(
            f"{measure} · {band} · {spatial_level} · {condition}"
        ),
        xlabel="Canale/ROI",
        ylabel="Mediana sulle 18 finestre",
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_condition_differences(
    paired: pd.DataFrame,
    *,
    measure: str,
    band: str,
    spatial_level: str,
    output_path: Path,
) -> None:
    """Distribuzione delle differenze appaiate CE-OE."""

    configure_style()

    subset = paired.loc[
        (paired["measure"] == measure)
        & (paired["band"] == band)
        & (paired["spatial_level"] == spatial_level)
    ]

    figure, axis = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=subset,
        x="location",
        y="CE_minus_OE",
        hue="group",
        hue_order=["hc", "ms"],
        showfliers=False,
        ax=axis,
    )

    sns.stripplot(
        data=subset,
        x="location",
        y="CE_minus_OE",
        hue="group",
        hue_order=["hc", "ms"],
        dodge=True,
        alpha=0.65,
        size=3,
        legend=False,
        ax=axis,
    )

    axis.axhline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
    )

    axis.set(
        title=f"Differenza appaiata CE−OE · {measure} · {band}",
        xlabel="Canale/ROI",
        ylabel="CE − OE",
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_correlation_heatmap(
    correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()

    figure, axis = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        correlations,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=axis,
    )

    axis.set_title(
        "Correlazioni di Spearman tra feature subject-level"
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)