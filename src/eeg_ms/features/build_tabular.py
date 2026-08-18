"""Costruzione della rappresentazione tabellare subject-level."""

from collections.abc import Sequence

import pandas as pd

from eeg_ms.config import (
    CANONICAL_DATA,
    SUBJECT_FEATURES_ROI,
    SUBJECTS_DATA,
)


FEATURE_DESCRIPTOR_COLUMNS = [
    "feature_family",
    "measure",
    "band",
    "spatial_level",
    "location",
]


def aggregate_windows(
    canonical: pd.DataFrame,
    spatial_levels: Sequence[str],
) -> pd.DataFrame:
    """
    Calcola statistiche intra-soggetto sulle 18 finestre.

    Non usa informazioni provenienti da altri soggetti e quindi
    non causa leakage fra training e test.
    """

    subset = canonical.loc[
        canonical["spatial_level"].isin(spatial_levels)
    ].copy()

    group_columns = [
        "subject_id",
        "condition",
        *FEATURE_DESCRIPTOR_COLUMNS,
    ]

    aggregated = (
        subset
        .groupby(
            group_columns,
            observed=True,
            dropna=False,
        )
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            median=("value", "median"),
            q25=("value", lambda x: x.quantile(0.25)),
            q75=("value", lambda x: x.quantile(0.75)),
            n_observed=("value", "count"),
        )
        .reset_index()
    )

    aggregated["iqr"] = (
        aggregated["q75"] - aggregated["q25"]
    )

    return aggregated


def create_condition_views(
    aggregated: pd.DataFrame,
    statistics: Sequence[str],
    condition_views: Sequence[str],
) -> pd.DataFrame:
    """
    Conserva CE e OE e calcola il contrasto appaiato CE−OE.
    """

    long_statistics = aggregated.melt(
        id_vars=[
            "subject_id",
            "condition",
            *FEATURE_DESCRIPTOR_COLUMNS,
        ],
        value_vars=list(statistics),
        var_name="statistic",
        value_name="feature_value",
    )

    index_columns = [
        "subject_id",
        *FEATURE_DESCRIPTOR_COLUMNS,
        "statistic",
    ]

    paired = (
        long_statistics
        .pivot(
            index=index_columns,
            columns="condition",
            values="feature_value",
        )
        .reset_index()
    )

    paired.columns.name = None

    if "CE" not in paired.columns:
        paired["CE"] = pd.NA

    if "OE" not in paired.columns:
        paired["OE"] = pd.NA

    paired["CE_minus_OE"] = (
        paired["CE"] - paired["OE"]
    )

    unavailable_views = (
        set(condition_views) - set(paired.columns)
    )

    if unavailable_views:
        raise ValueError(
            f"Viste di condizione non disponibili: "
            f"{sorted(unavailable_views)}"
        )

    condition_long = paired.melt(
        id_vars=index_columns,
        value_vars=list(condition_views),
        var_name="condition_view",
        value_name="feature_value",
    )

    return condition_long


def create_feature_names(
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Crea nomi leggibili e univoci per le colonne ML."""

    result = feature_frame.copy()

    result["feature_name"] = (
        result["feature_family"].astype(str)
        + "__"
        + result["measure"].astype(str)
        + "__"
        + result["band"].astype(str)
        + "__"
        + result["spatial_level"].astype(str)
        + "__"
        + result["location"].astype(str)
        + "__"
        + result["statistic"].astype(str)
        + "__"
        + result["condition_view"].astype(str)
    )

    return result


def build_subject_feature_matrix(
    canonical: pd.DataFrame,
    subjects: pd.DataFrame,
    *,
    spatial_levels: Sequence[str] = ("roi",),
    statistics: Sequence[str] = ("median", "iqr"),
    condition_views: Sequence[str] = (
        "CE",
        "OE",
        "CE_minus_OE",
    ),
) -> pd.DataFrame:
    """Produce una matrice con una sola riga per soggetto."""

    aggregated = aggregate_windows(
        canonical=canonical,
        spatial_levels=spatial_levels,
    )

    condition_frame = create_condition_views(
        aggregated=aggregated,
        statistics=statistics,
        condition_views=condition_views,
    )

    condition_frame = create_feature_names(
        condition_frame
    )

    duplicated = condition_frame.duplicated(
        ["subject_id", "feature_name"]
    )

    if duplicated.any():
        raise ValueError(
            "Sono presenti feature duplicate per soggetto."
        )

    feature_matrix = (
        condition_frame
        .pivot(
            index="subject_id",
            columns="feature_name",
            values="feature_value",
        )
        .reset_index()
    )

    feature_matrix.columns.name = None

    metadata = subjects[
        ["subject_id", "group", "target", "visit"]
    ].drop_duplicates()

    result = metadata.merge(
        feature_matrix,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )

    if result["subject_id"].duplicated().any():
        raise ValueError(
            "La matrice finale contiene soggetti duplicati."
        )

    feature_columns = [
        column
        for column in result.columns
        if column not in {
            "subject_id",
            "group",
            "target",
            "visit",
        }
    ]

    completely_missing = [
        column
        for column in feature_columns
        if result[column].isna().all()
    ]

    if completely_missing:
        raise ValueError(
            "Feature interamente mancanti: "
            f"{completely_missing[:10]}"
        )

    return result


def main() -> None:
    canonical = pd.read_parquet(CANONICAL_DATA)
    subjects = pd.read_csv(SUBJECTS_DATA)

    subject_features = build_subject_feature_matrix(
        canonical=canonical,
        subjects=subjects,
        spatial_levels=("roi",),
        statistics=("median", "iqr"),
        condition_views=(
            "CE",
            "OE",
            "CE_minus_OE",
        ),
    )

    SUBJECT_FEATURES_ROI.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_features.to_parquet(
        SUBJECT_FEATURES_ROI,
        index=False,
        compression="zstd",
    )

    n_metadata_columns = 4
    n_features = (
        subject_features.shape[1] - n_metadata_columns
    )

    print(f"Soggetti: {subject_features.shape[0]}")
    print(f"Feature: {n_features}")
    print(f"Shape completa: {subject_features.shape}")
    print(f"Output: {SUBJECT_FEATURES_ROI}")


if __name__ == "__main__":
    main()