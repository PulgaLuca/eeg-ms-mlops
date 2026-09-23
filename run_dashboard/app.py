from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, abort, render_template, send_file


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"

app = Flask(__name__)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_timestamp(value: str | None) -> str:
    timestamp = parse_timestamp(value)
    return timestamp.astimezone().strftime("%d/%m/%Y %H:%M") if timestamp else "-"


def format_duration(seconds: Any) -> str:
    if seconds is None:
        return "-"
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f} min"


def model_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "results" / "metrics" / "nested_cv_fold_metrics.csv"
    if not metrics_path.exists():
        return []

    try:
        metrics = pd.read_csv(metrics_path)
        metric_names = [
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
        ]
        available = [name for name in metric_names if name in metrics]
        summary = metrics.groupby("model", as_index=False)[available].mean()
        rows = summary.to_dict(orient="records")
        for row in rows:
            row["model"] = str(row["model"])
        return sorted(rows, key=lambda row: row.get("balanced_accuracy", 0), reverse=True)
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return []


def run_record(run_dir: Path) -> dict[str, Any]:
    metadata = read_json(run_dir / "metadata.json")
    subjects = metadata.get("subjects", {})
    return {
        "run_id": metadata.get("run_id", run_dir.name),
        "status": metadata.get("status", "unknown"),
        "started_at": format_timestamp(metadata.get("started_at_utc")),
        "finished_at": format_timestamp(metadata.get("finished_at_utc")),
        "duration": format_duration(metadata.get("duration_seconds")),
        "n_subjects": subjects.get("n_subjects", "-"),
        "n_ms": subjects.get("group_counts", {}).get("ms", "-"),
        "n_hc": subjects.get("group_counts", {}).get("hc", "-"),
        "metadata": metadata,
        "models": model_rows(run_dir),
        "path": run_dir,
    }


def all_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    run_dirs = [path for path in RUNS_DIR.iterdir() if path.is_dir() and path.name.startswith("run_")]
    return sorted((run_record(path) for path in run_dirs), key=lambda run: run["started_at"], reverse=True)


def get_run_or_404(run_id: str) -> dict[str, Any]:
    if not run_id.startswith("run_"):
        abort(404)
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir() or run_dir.parent != RUNS_DIR:
        abort(404)
    return run_record(run_dir)


@app.template_filter("metric")
def metric(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


@app.route("/")
def index():
    runs = all_runs()
    completed = sum(run["status"] == "completed" for run in runs)
    failed = sum(run["status"] == "failed" for run in runs)
    return render_template(
        "index.html",
        runs=runs,
        completed=completed,
        failed=failed,
        total=len(runs),
        project_root=PROJECT_ROOT.name,
    )


@app.route("/runs/<run_id>")
def run_detail(run_id: str):
    run = get_run_or_404(run_id)
    metadata = run["metadata"]
    model_catalog = metadata.get("models", [])
    inputs = metadata.get("inputs", {})
    return render_template(
        "run_detail.html",
        run=run,
        metadata=metadata,
        model_catalog=model_catalog,
        inputs=inputs,
    )


@app.route("/runs/<run_id>/files/<path:relative_path>")
def run_file(run_id: str, relative_path: str):
    get_run_or_404(run_id)
    run_dir = RUNS_DIR / run_id
    requested = (run_dir / relative_path).resolve()
    if run_dir not in requested.parents or not requested.is_file():
        abort(404)
    return send_file(requested, as_attachment=True, download_name=requested.name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
