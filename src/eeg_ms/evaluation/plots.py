"""Visualizzazioni per l'analisi controllata dei modelli."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def configure_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
    )


def plot_fold_metric_distributions(
    fold_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()

    metrics = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]

    metrics = [
        metric
        for metric in metrics
        if metric in fold_metrics.columns
    ]

    long_frame = fold_metrics.melt(
        id_vars=["model", "repeat", "fold"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

    chart = sns.catplot(
        data=long_frame,
        x="model",
        y="score",
        col="metric",
        col_wrap=3,
        kind="box",
        sharey=True,
        height=3.5,
        aspect=1.2,
    )

    for axis in chart.axes.flat:
        axis.tick_params(
            axis="x",
            rotation=45,
        )
        axis.set_ylim(0, 1)

    chart.set_axis_labels(
        "Modello",
        "Score outer fold",
    )

    chart.figure.tight_layout()
    chart.figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(chart.figure)


def plot_oof_roc_curves(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()

    figure, axis = plt.subplots(figsize=(7, 6))

    for model, frame in aggregated.groupby("model"):
        fpr, tpr, _ = roc_curve(
            frame["target"],
            frame["probability_ms"],
        )

        auc = roc_auc_score(
            frame["target"],
            frame["probability_ms"],
        )

        axis.plot(
            fpr,
            tpr,
            label=f"{model} · AUC={auc:.2f}",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="grey",
        label="Casuale",
    )

    axis.set(
        title="Curve ROC sulle predizioni OOF aggregate",
        xlabel="1 − Specificità",
        ylabel="Sensibilità",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    axis.legend(
        loc="lower right",
        fontsize="small",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_oof_precision_recall_curves(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()

    figure, axis = plt.subplots(figsize=(7, 6))

    for model, frame in aggregated.groupby("model"):
        precision, recall, _ = precision_recall_curve(
            frame["target"],
            frame["probability_ms"],
        )

        average_precision = average_precision_score(
            frame["target"],
            frame["probability_ms"],
        )

        axis.plot(
            recall,
            precision,
            label=(
                f"{model} · AP={average_precision:.2f}"
            ),
        )

    prevalence = aggregated["target"].mean()

    axis.axhline(
        prevalence,
        linestyle="--",
        color="grey",
        label=f"Prevalenza MS={prevalence:.2f}",
    )

    axis.set(
        title=(
            "Precision-recall sulle predizioni OOF aggregate"
        ),
        xlabel="Recall / sensibilità",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    axis.legend(
        loc="lower left",
        fontsize="small",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrices(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()

    models = sorted(aggregated["model"].unique())
    n_columns = 3
    n_rows = math.ceil(len(models) / n_columns)

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4 * n_columns, 3.5 * n_rows),
        squeeze=False,
    )

    for axis, model in zip(
        axes.flat,
        models,
        strict=False,
    ):
        frame = aggregated.loc[
            aggregated["model"] == model
        ]

        predicted = (
            frame["probability_ms"] >= 0.5
        ).astype(int)

        matrix = confusion_matrix(
            frame["target"],
            predicted,
            labels=[0, 1],
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            xticklabels=["HC", "MS"],
            yticklabels=["HC", "MS"],
            ax=axis,
        )

        axis.set(
            title=model,
            xlabel="Predetto",
            ylabel="Reale",
        )

    for axis in axes.flat[len(models):]:
        axis.set_visible(False)

    figure.suptitle(
        "Confusion matrix soggetto-livello · soglia 0.5"
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_subject_probabilities(
    aggregated: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()

    models = sorted(aggregated["model"].unique())
    n_columns = 2
    n_rows = math.ceil(len(models) / n_columns)

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(8 * n_columns, 4 * n_rows),
        squeeze=False,
    )

    for axis, model in zip(
        axes.flat,
        models,
        strict=False,
    ):
        frame = (
            aggregated.loc[
                aggregated["model"] == model
            ]
            .sort_values("probability_ms")
            .reset_index(drop=True)
        )

        colors = frame["target"].map(
            {
                0: "tab:blue",
                1: "tab:orange",
            }
        )

        axis.errorbar(
            x=np.arange(len(frame)),
            y=frame["probability_ms"],
            yerr=frame["probability_std"].fillna(0),
            fmt="none",
            ecolor="grey",
            alpha=0.5,
            capsize=2,
        )

        axis.scatter(
            np.arange(len(frame)),
            frame["probability_ms"],
            c=colors,
            s=30,
        )

        axis.axhline(
            0.5,
            color="black",
            linestyle="--",
            linewidth=1,
        )

        axis.set(
            title=model,
            xlabel="Soggetti ordinati",
            ylabel="Probabilità MS",
            ylim=(-0.05, 1.05),
        )

    for axis in axes.flat[len(models):]:
        axis.set_visible(False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_bootstrap_intervals(
    intervals: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "balanced_accuracy",
) -> None:
    configure_style()

    subset = (
        intervals.loc[
            intervals["metric"] == metric
        ]
        .sort_values("estimate")
        .reset_index(drop=True)
    )

    lower_error = (
        subset["estimate"] - subset["ci_lower"]
    )

    upper_error = (
        subset["ci_upper"] - subset["estimate"]
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.errorbar(
        x=subset["estimate"],
        y=subset["model"],
        xerr=np.vstack(
            [lower_error, upper_error]
        ),
        fmt="o",
        capsize=4,
    )

    axis.axvline(
        0.5,
        linestyle="--",
        color="grey",
        label="Prestazione casuale",
    )

    axis.set(
        title=(
            f"{metric}: stima e intervallo bootstrap al 95%"
        ),
        xlabel=metric,
        ylabel="Modello",
        xlim=(0, 1),
    )

    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)