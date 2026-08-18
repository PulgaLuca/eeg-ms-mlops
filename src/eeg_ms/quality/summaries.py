"""Statistiche descrittive e controlli di qualità subject-level."""

import numpy as np
import pandas as pd


FEATURE_KEYS = [
    "feature_family",
    "measure",
    "band",
    "spatial_level",
    "location",
]


def subject_level_summary(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Riassume le 18 finestre per soggetto.

    La mediana è robusta rispetto a finestre anomale.
    """

    group_columns = [
        "subject_id",
        "group",
        "target",
        "condition",
        *FEATURE_KEYS,
    ]

    return (
        canonical
        .groupby(
            group_columns,
            observed=True,
            dropna=False,
        )
        .agg(
            window_mean=("value", "mean"),
            window_median=("value", "median"),
            window_std=("value", "std"),
            window_q25=("value", lambda x: x.quantile(0.25)),
            window_q75=("value", lambda x: x.quantile(0.75)),
            n_windows_observed=("value", "count"),
            n_windows_missing=("value", lambda x: x.isna().sum()),
        )
        .reset_index()
    )


def missingness_summary(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Missingness per famiglia, condizione e livello spaziale."""

    group_columns = [
        "group",
        "condition",
        "feature_family",
        "measure",
        "band",
        "spatial_level",
    ]

    return (
        canonical
        .groupby(
            group_columns,
            observed=True,
            dropna=False,
        )
        .agg(
            n_values=("value", "size"),
            n_missing=("value", lambda x: x.isna().sum()),
        )
        .assign(
            missing_fraction=lambda frame:
                frame["n_missing"] / frame["n_values"]
        )
        .reset_index()
        .sort_values(
            "missing_fraction",
            ascending=False,
        )
    )


def robust_outlier_report(
    subject_summary: pd.DataFrame,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Identifica possibili outlier tecnici usando mediana e MAD.

    La soglia è calcolata tra soggetti, non tra finestre.
    Il gruppo HC/MS non entra nel calcolo della soglia.
    """

    strata = [
        "condition",
        *FEATURE_KEYS,
    ]

    data = subject_summary.copy()

    groupers = [
        data[column]
        for column in strata
    ]

    median = data.groupby(
        groupers,
        observed=True,
    )["window_median"].transform("median")

    mad = data.groupby(
        groupers,
        observed=True,
    )["window_median"].transform(
        lambda values: np.median(
            np.abs(values - np.median(values))
        )
    )

    data["reference_median"] = median
    data["mad"] = mad

    denominator = 1.4826 * mad

    data["robust_z"] = np.where(
        denominator > 0,
        (data["window_median"] - median) / denominator,
        np.nan,
    )

    data["technical_outlier_flag"] = (
        data["robust_z"].abs() > threshold
    )

    return data.loc[
        data["technical_outlier_flag"]
    ].sort_values(
        "robust_z",
        key=lambda values: values.abs(),
        ascending=False,
    )


def condition_differences(
    subject_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcola CE-OE all'interno dello stesso soggetto.

    Il confronto è appaiato per costruzione.
    """

    index_columns = [
        "subject_id",
        "group",
        "target",
        *FEATURE_KEYS,
    ]

    paired = (
        subject_summary
        .pivot(
            index=index_columns,
            columns="condition",
            values="window_median",
        )
        .reset_index()
    )

    paired.columns.name = None

    paired = paired.dropna(
        subset=["CE", "OE"]
    )

    paired["CE_minus_OE"] = (
        paired["CE"] - paired["OE"]
    )

    return paired


def window_profiles(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce profili per finestra.

    Prima media sulle location per soggetto; successivamente
    statistiche fra soggetti.
    """

    subject_window = (
        canonical
        .groupby(
            [
                "subject_id",
                "group",
                "condition",
                "feature_family",
                "measure",
                "band",
                "spatial_level",
                "window",
            ],
            observed=True,
            dropna=False,
        )["value"]
        .mean()
        .rename("subject_window_mean")
        .reset_index()
    )

    return (
        subject_window
        .groupby(
            [
                "group",
                "condition",
                "feature_family",
                "measure",
                "band",
                "spatial_level",
                "window",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            median=("subject_window_mean", "median"),
            q25=("subject_window_mean", lambda x: x.quantile(0.25)),
            q75=("subject_window_mean", lambda x: x.quantile(0.75)),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )


def subject_correlation_matrix(
    subject_summary: pd.DataFrame,
    *,
    condition: str,
    spatial_level: str = "roi",
) -> pd.DataFrame:
    """
    Correlazioni calcolate su soggetti, non sulle singole finestre.

    Si consiglia di filtrare a ROI per mantenere la matrice leggibile.
    """

    subset = subject_summary.loc[
        (subject_summary["condition"] == condition)
        & (subject_summary["spatial_level"] == spatial_level)
    ].copy()

    subset["feature_name"] = (
        subset["measure"].astype(str)
        + "__"
        + subset["band"].astype(str)
        + "__"
        + subset["location"].astype(str)
    )

    wide = subset.pivot(
        index="subject_id",
        columns="feature_name",
        values="window_median",
    )

    return wide.corr(
        method="spearman",
        min_periods=10,
    )