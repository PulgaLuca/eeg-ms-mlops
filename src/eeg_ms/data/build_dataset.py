"""Costruzione riproducibile del dataset canonico."""

import pandas as pd

from eeg_ms.config import (
    CANONICAL_DATA,
    CHANNELS_FILE,
    DATA_PROCESSED,
    SUBJECTS_DATA,
)
from eeg_ms.data.audit import run_audit
from eeg_ms.data.canonical import CANONICAL_COLUMNS, feature_file_to_long, validate_canonical_dataset
from eeg_ms.data.discovery import discover_feature_files
from eeg_ms.data.loading import load_channel_metadata


def build_canonical_dataset() -> pd.DataFrame:
    """
    Esegue audit, conversione e salvataggio.
    Il dataset non viene costruito in presenza di errori strutturali.
    """

    issues = run_audit()

    for issue in issues:
        print(
            f"[{issue.severity.upper()}] "
            f"{issue.code}: {issue.message}"
        )

    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise RuntimeError(f"Costruzione interrotta: trovati {len(errors)} errori nell'audit.")

    records = discover_feature_files()
    channels, rois = load_channel_metadata(CHANNELS_FILE)
    roi_names = list(rois)

    frames = [
        feature_file_to_long(
            metadata=record,
            channels=channels,
            roi_names=roi_names,
        )
        for record in records
    ]

    canonical = pd.concat(frames, ignore_index=True,)[CANONICAL_COLUMNS]

    validate_canonical_dataset(canonical)

    canonical = canonical.sort_values(
        [
            "subject_id",
            "condition",
            "feature_family",
            "measure",
            "band",
            "spatial_level",
            "location_index",
            "window",
        ],
        ignore_index=True,
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True,)
    canonical.to_parquet(CANONICAL_DATA, index=False, compression="zstd",)

    subjects = (
        canonical[
            [
                "subject_id",
                "group",
                "target",
                "visit",
            ]
        ]
        .drop_duplicates()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    if subjects["subject_id"].duplicated().any():
        raise ValueError("subjects.csv conterrebbe soggetti duplicati.")

    subjects.to_csv(SUBJECTS_DATA, index=False,)

    return canonical


def main() -> None:
    canonical = build_canonical_dataset()

    print(f"Righe canoniche: {len(canonical):,}")
    print(f"Soggetti: {canonical['subject_id'].nunique()}")
    print(f"Output in: {CANONICAL_DATA}")


if __name__ == "__main__":
    main()
