"""Supporto leggero per tracciare le esecuzioni sperimentali."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd

from eeg_ms.config import LATEST_RUN_FILE, RUNS_DIR


def json_default(value: Any) -> Any:
    """Converte oggetti non JSON in una rappresentazione leggibile."""

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "item"):
        return value.item()

    return repr(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def dataframe_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
    }


def file_summary(path: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }

    if path.exists():
        summary.update(
            {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    if frame is not None:
        summary["dataframe"] = dataframe_summary(frame)

    return summary


def git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def create_run_directory() -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    LATEST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_FILE.write_text(f"{run_dir}\n", encoding="utf-8")
    return run_id, run_dir


def latest_run_directory() -> Path | None:
    if not LATEST_RUN_FILE.exists():
        return None

    run_dir = Path(LATEST_RUN_FILE.read_text(encoding="utf-8").strip())
    return run_dir if run_dir.is_dir() else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def mark_latest_run_failed(error: BaseException) -> None:
    run_dir = latest_run_directory()
    metadata_path = run_dir / "metadata.json" if run_dir else None

    if metadata_path is None or not metadata_path.exists():
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    )
    write_json(metadata_path, metadata)


def copy_input(source: Path, run_dir: Path, name: str) -> Path:
    destination = run_dir / "inputs" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }