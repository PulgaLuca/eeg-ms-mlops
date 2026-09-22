"""CLI per controllare il dataset raw e produrre gli inventari."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_ms.config import CHANNELS_FILE, DATA_INTERIM, DATA_RAW
from eeg_ms.data.discovery import discover_feature_files
from eeg_ms.data.loading import load_channel_metadata, load_feature_file
from eeg_ms.data.validation import (
    ValidationIssue,
    validate_channel_metadata,
    validate_dataset_completeness,
    validate_feature_data,
)


def calculate_sha256(path: Path) -> str:
    """Calcola l'impronta del file per garantirne la tracciabilità."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def run_audit(
    raw_dir: Path = DATA_RAW,
    output_dir: Path = DATA_INTERIM,
) -> list[ValidationIssue]:
    """Esegue l'audit completo e salva manifest e report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_feature_files(raw_dir)
    issues = validate_dataset_completeness(records)

    file_rows: list[dict] = []
    variable_rows: list[dict] = []

    for metadata in records:
        data = load_feature_file(metadata)
        file_issues = validate_feature_data(metadata, data)
        issues.extend(file_issues)

        file_rows.append(
            {
                "relative_path": str(metadata.path.relative_to(raw_dir)),
                "family": metadata.family,
                "group": metadata.group,
                "subject_id": metadata.subject_id,
                "subject_number": metadata.subject_number,
                "visit": metadata.visit,
                "condition": metadata.condition,
                "file_size_bytes": metadata.path.stat().st_size,
                "sha256": calculate_sha256(metadata.path),
                "n_variables": len(data),
                "has_errors": any(issue.severity == "error" for issue in file_issues),
            }
        )

        for variable_name, value in sorted(data.items()):
            array = np.asarray(value)
            numeric = np.issubdtype(array.dtype, np.number)

            variable_rows.append(
                {
                    "relative_path": str(metadata.path.relative_to(raw_dir)),
                    "subject_id": metadata.subject_id,
                    "family": metadata.family,
                    "condition": metadata.condition,
                    "variable": variable_name,
                    "shape": "x".join(map(str, array.shape)),
                    "dtype": str(array.dtype),
                    "n_nan": (int(np.isnan(array).sum()) if numeric else None),
                    "n_inf": (int(np.isinf(array).sum()) if numeric else None),
                    "minimum": (float(np.nanmin(array)) if numeric else None),
                    "maximum": (float(np.nanmax(array)) if numeric else None),
                }
            )

    channels, rois = load_channel_metadata(CHANNELS_FILE)
    issues.extend(
        validate_channel_metadata(
            channels=channels,
            rois=rois,
            source=CHANNELS_FILE,
        )
    )

    pd.DataFrame(file_rows).to_csv(output_dir / "file_manifest.csv", index=False,)
    pd.DataFrame(variable_rows).to_csv(output_dir / "variable_inventory.csv",index=False,)

    report = {
        "status": (
            "failed"
            if any(issue.severity == "error" for issue in issues)
            else "passed"
        ),
        "n_files": len(records),
        "n_subjects": len({record.subject_id for record in records}),
        "n_errors": sum(
            issue.severity == "error" for issue in issues
        ),
        "n_warnings": sum(
            issue.severity == "warning" for issue in issues
        ),
        "channels": channels,
        "rois": rois,
        "issues": [issue.to_dict() for issue in issues],
    }

    with (output_dir / "validation_report.json").open("w", encoding="utf-8",) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return issues


def main() -> None:
    issues = run_audit()

    errors = [issue for issue in issues if issue.severity == "error"]

    for issue in issues:
        print(
            f"[{issue.severity.upper()}] "
            f"{issue.code}: {issue.message}"
        )

    if errors:
        raise SystemExit(f"Audit fallito con {len(errors)} errori.")

    print("Audit completato senza errori bloccanti.")


if __name__ == "__main__":
    main()
    