"""Conversione delle matrici MATLAB nel dataset canonico long."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from eeg_ms.config import (
    BANDS,
    DATA_RAW,
    N_WINDOWS,
    TARGET_MAP,
)
from eeg_ms.data.loading import load_feature_file
from eeg_ms.data.parsing import FileMetadata

CANONICAL_COLUMNS = [
    "subject_id",
    "group",
    "target",
    "visit",
    "condition",
    "feature_family",
    "measure",
    "band",
    "spatial_level",
    "location",
    "location_index",
    "window",
    "value",
    "source_file",
]


def matrix_to_long(
    metadata: FileMetadata,
    matrix: np.ndarray,
    locations: Sequence[str],
    measure: str,
    band: str,
    spatial_level: str,
) -> pd.DataFrame:
    """
    Converte una matrice [location, window] in formato long.

    L'ordine C garantisce:
    location 1, finestre 1..18;
    location 2, finestre 1..18;
    ecc.
    """

    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"{metadata.path.name}: attesa matrice 2D, "
            f"ricevuta shape {matrix.shape}"
        )

    if matrix.shape[0] != len(locations):
        raise ValueError(
            f"{metadata.path.name}: trovate {matrix.shape[0]} location, "
            f"attese {len(locations)}"
        )

    if matrix.shape[1] < N_WINDOWS:
        raise ValueError(
            f"{metadata.path.name}: trovate solamente "
            f"{matrix.shape[1]} finestre; ne servono almeno {N_WINDOWS}"
        )

    # WARNING: mentengo soltanto le prime 18 finestre in quanto in alcune registrazioni raw, alcune finestre sono da 27x18-19-20-21
    matrix = matrix[:, :N_WINDOWS]

    n_locations = len(locations)

    return pd.DataFrame(
        {
            "subject_id": metadata.subject_id,
            "group": metadata.group,
            "target": TARGET_MAP[metadata.group],
            "visit": metadata.visit,
            "condition": metadata.condition,
            "feature_family": metadata.family,
            "measure": measure,
            "band": band,
            "spatial_level": spatial_level,
            "location": np.repeat(locations, N_WINDOWS),
            "location_index": np.repeat(
                np.arange(1, n_locations + 1),
                N_WINDOWS,
            ),
            "window": np.tile(
                np.arange(1, N_WINDOWS + 1),
                n_locations,
            ),
            "value": matrix.reshape(-1, order="C"),
            "source_file": str(
                metadata.path.relative_to(DATA_RAW)
            ),
        }
    )

def psd_to_long(
    metadata: FileMetadata,
    data: dict[str, np.ndarray],
    channels: list[str],
    roi_names: list[str],
) -> pd.DataFrame:
    """Converte PSD e indici derivati nel formato canonico."""

    frames: list[pd.DataFrame] = []

    # Feature PSD calcolate per ciascuna banda.
    for band in BANDS:
        band_specifications = [
            (
                f"PSD_{band}",
                "absolute_power",
                "channel",
                channels,
            ),
            (
                f"PSD_{band}_rel",
                "relative_power",
                "channel",
                channels,
            ),
            (
                f"PSD_{band}_ROI",
                "absolute_power",
                "roi",
                roi_names,
            ),
            (
                f"PSD_{band}_ROI_rel",
                "relative_power",
                "roi",
                roi_names,
            ),
        ]

        for (
            variable,
            measure,
            spatial_level,
            locations,
        ) in band_specifications:
            frames.append(
                matrix_to_long(
                    metadata=metadata,
                    matrix=data[variable],
                    locations=locations,
                    measure=measure,
                    band=band,
                    spatial_level=spatial_level,
                )
            )

    # ATTENZIONE: questo blocco deve essere fuori dal ciclo BANDS.
    derived_specifications = [
        (
            "PSD_theta_alpha",
            "theta_alpha_ratio",
            "channel",
            channels,
        ),
        (
            "PSD_theta_alpha_ROI",
            "theta_alpha_ratio",
            "roi",
            roi_names,
        ),
        (
            "PSD_delta_alpha",
            "delta_alpha_ratio",
            "channel",
            channels,
        ),
        (
            "PSD_delta_alpha_ROI",
            "delta_alpha_ratio",
            "roi",
            roi_names,
        ),
        (
            "SFR",
            "sfr",
            "channel",
            channels,
        ),
        (
            "SFR_ROI",
            "sfr",
            "roi",
            roi_names,
        ),
    ]

    for (
        variable,
        measure,
        spatial_level,
        locations,
    ) in derived_specifications:
        frames.append(
            matrix_to_long(
                metadata=metadata,
                matrix=data[variable],
                locations=locations,
                measure=measure,
                band="derived",
                spatial_level=spatial_level,
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )

def complexity_to_long(
    metadata: FileMetadata,
    data: dict[str, np.ndarray],
    channels: list[str],
    roi_names: list[str],
) -> pd.DataFrame:
    """Converte spectral entropy nel formato canonico."""

    channel_frame = matrix_to_long(
        metadata=metadata,
        matrix=data["spectral_all"],
        locations=channels,
        measure="spectral_entropy",
        band="broadband",
        spatial_level="channel",
    )

    roi_frame = matrix_to_long(
        metadata=metadata,
        matrix=data["spectral_all_ROI"],
        locations=roi_names,
        measure="spectral_entropy",
        band="broadband",
        spatial_level="roi",
    )

    return pd.concat(
        [channel_frame, roi_frame],
        ignore_index=True,
    )


def feature_file_to_long(
    metadata: FileMetadata,
    channels: list[str],
    roi_names: list[str],
) -> pd.DataFrame:
    """Carica e converte un singolo file MATLAB."""

    data = load_feature_file(metadata)

    if metadata.family == "psd":
        return psd_to_long(
            metadata=metadata,
            data=data,
            channels=channels,
            roi_names=roi_names,
        )

    if metadata.family == "complexity":
        return complexity_to_long(
            metadata=metadata,
            data=data,
            channels=channels,
            roi_names=roi_names,
        )

    raise ValueError(
        f"Famiglia di feature non supportata: {metadata.family}"
    )


def validate_canonical_dataset(frame: pd.DataFrame) -> None:
    """Verifica le proprietà fondamentali del dataset canonico."""

    missing_columns = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Colonne canoniche mancanti: {sorted(missing_columns)}")

    identity_columns = [
        "subject_id",
        "condition",
        "feature_family",
        "measure",
        "band",
        "spatial_level",
        "location",
        "window",
    ]

    duplicated = frame.duplicated(identity_columns)

    if duplicated.any():
        examples = frame.loc[duplicated, identity_columns,].head()

        raise ValueError(f"Osservazioni canoniche duplicate:\n {examples.to_string(index=False)}")

    target_counts = (
        frame[["subject_id", "target"]]
        .drop_duplicates()
        .groupby("subject_id")["target"]
        .nunique()
    )

    if (target_counts != 1).any():
        raise ValueError("Uno stesso subject_id è associato a target differenti.")

    invalid_windows = ~frame["window"].between(1, N_WINDOWS)
    if invalid_windows.any():
        raise ValueError("Sono presenti indici di finestra non validi.")

    if np.isinf(frame["value"]).any():
        raise ValueError("Il dataset contiene valori infiniti.")
    