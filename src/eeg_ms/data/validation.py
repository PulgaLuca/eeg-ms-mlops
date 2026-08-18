"""Controlli strutturali, numerici e di completezza del dataset."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from eeg_ms.config import BANDS, CONDITIONS, N_CHANNELS, N_ROIS, N_WINDOWS
from eeg_ms.data.parsing import FileMetadata

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


CHANNEL_SHAPE = (N_CHANNELS, N_WINDOWS)
ROI_SHAPE = (N_ROIS, N_WINDOWS)

PSD_EXPECTED: dict[str, tuple[int, int]] = {}

for band in BANDS:
    PSD_EXPECTED[f"PSD_{band}"] = CHANNEL_SHAPE
    PSD_EXPECTED[f"PSD_{band}_rel"] = CHANNEL_SHAPE
    PSD_EXPECTED[f"PSD_{band}_ROI"] = ROI_SHAPE
    PSD_EXPECTED[f"PSD_{band}_ROI_rel"] = ROI_SHAPE

PSD_EXPECTED.update(
    {
        "PSD_delta_alpha": CHANNEL_SHAPE,
        "PSD_delta_alpha_ROI": ROI_SHAPE,
        "PSD_theta_alpha": CHANNEL_SHAPE,
        "PSD_theta_alpha_ROI": ROI_SHAPE,
        "PSD_tot": CHANNEL_SHAPE,
        "PSD_tot_ROI": ROI_SHAPE,
        "SFR": CHANNEL_SHAPE,
        "SFR_ROI": ROI_SHAPE,
    }
)

COMPLEXITY_EXPECTED = {
    "spectral_all": CHANNEL_SHAPE,
    "spectral_all_ROI": ROI_SHAPE,
}

EXPECTED_VARIABLES = {
    "psd": PSD_EXPECTED,
    "complexity": COMPLEXITY_EXPECTED,
}


def validate_feature_data(
    metadata: FileMetadata,
    data: dict[str, np.ndarray],
) -> list[ValidationIssue]:
    """Controlla variabili, shape, tipo e valori di un file."""

    issues: list[ValidationIssue] = []
    expected = EXPECTED_VARIABLES[metadata.family]
    actual_names = set(data)
    expected_names = set(expected)

    for name in sorted(expected_names - actual_names):
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_variable",
                message=f"Variabile obbligatoria assente: {name}",
                path=str(metadata.path),
            )
        )

    extra_names = sorted(actual_names - expected_names)
    if extra_names:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="extra_variables",
                message=f"Variabili extra presenti: {extra_names}",
                path=str(metadata.path),
            )
        )

    for name, expected_shape in expected.items():
        if name not in data:
            continue

        array = np.asarray(data[name])

        if array.shape != expected_shape:
            if (
                metadata.family == "complexity"
                and name in {"spectral_all", "spectral_all_ROI"}
                and array.ndim == 2
                and array.shape[0] == expected_shape[0]
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="unexpected_window_count",
                        message=(
                            f"{name}: shape {array.shape}, "
                            f"attesa {expected_shape}; "
                            f"numero di finestre diverso da N_WINDOWS={N_WINDOWS}"
                        ),
                        path=str(metadata.path),
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid_shape",
                        message=(
                            f"{name}: shape {array.shape}, "
                            f"attesa {expected_shape}"
                        ),
                        path=str(metadata.path),
                    )
                )

            continue

        if not np.issubdtype(array.dtype, np.number):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="non_numeric",
                    message=f"{name}: dtype non numerico {array.dtype}",
                    path=str(metadata.path),
                )
            )
            continue

        if not np.isfinite(array).all():
            n_invalid = int((~np.isfinite(array)).sum())
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="non_finite_values",
                    message=f"{name}: {n_invalid} valori NaN o Inf",
                    path=str(metadata.path),
                )
            )

        if metadata.family == "psd" and not name.endswith("_rel"):
            if np.any(array < 0):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="negative_psd",
                        message=f"{name}: contiene potenze negative",
                        path=str(metadata.path),
                    )
                )

        is_relative_psd = (
            metadata.family == "psd" and name.endswith("_rel")
        )
        is_normalized_entropy = metadata.family == "complexity"

        if is_relative_psd or is_normalized_entropy:
            tolerance = 1e-6
            outside_range = (
                (array < -tolerance) | (array > 1 + tolerance)
            )

            if np.any(outside_range):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="outside_unit_interval",
                        message=f"{name}: valori fuori dall'intervallo [0, 1]",
                        path=str(metadata.path),
                    )
                )

    return issues


def validate_dataset_completeness(
    records: list[FileMetadata],
) -> list[ValidationIssue]:
    """Controlla duplicati e combinazioni mancanti per soggetto."""

    issues: list[ValidationIssue] = []

    keys = [
        (
            record.subject_id,
            record.family,
            record.condition,
        )
        for record in records
    ]

    for key, count in Counter(keys).items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_record",
                    message=f"Combinazione duplicata {key}: {count} file",
                )
            )

    subjects = sorted({record.subject_id for record in records})
    actual = set(keys)

    for subject_id in subjects:
        for family in EXPECTED_VARIABLES:
            for condition in CONDITIONS:
                key = (subject_id, family, condition)

                if key not in actual:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="missing_record",
                            message=f"Combinazione mancante: {key}",
                        )
                    )

    return issues


def validate_channel_metadata(
    channels: list[str],
    rois: dict[str, list[str]],
    source: Path,
) -> list[ValidationIssue]:
    """Controlla ordine, unicità e composizione delle ROI."""

    issues: list[ValidationIssue] = []

    if len(channels) != N_CHANNELS:
        issues.append(
            ValidationIssue(
                "error",
                "invalid_channel_count",
                f"Trovati {len(channels)} canali; attesi {N_CHANNELS}",
                str(source),
            )
        )

    if len(set(channels)) != len(channels):
        issues.append(
            ValidationIssue(
                "error",
                "duplicate_channels",
                "Le etichette dei canali non sono univoche",
                str(source),
            )
        )

    known_channels = set(channels)

    for roi_name, roi_channels in rois.items():
        unknown = sorted(set(roi_channels) - known_channels)

        if unknown:
            issues.append(
                ValidationIssue(
                    "error",
                    "unknown_roi_channels",
                    f"{roi_name} contiene canali sconosciuti: {unknown}",
                    str(source),
                )
            )

    assigned = {
        channel
        for roi_channels in rois.values()
        for channel in roi_channels
    }
    unassigned = sorted(known_channels - assigned)

    if unassigned:
        issues.append(
            ValidationIssue(
                "warning",
                "channels_without_roi",
                f"Canali non assegnati alle ROI: {unassigned}",
                str(source),
            )
        )

    return issues