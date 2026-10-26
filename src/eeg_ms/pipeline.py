"""Interactive, configurable end-to-end EEG-MS experiment entry point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

import pandas as pd

from eeg_ms.config import (
    CANONICAL_DATA,
    DATA_INTERIM,
    RUNS_DIR,
    SUBJECTS_DATA,
)
from eeg_ms.data.build_dataset import build_canonical_dataset
from eeg_ms.experiments.config import (
    DEFAULT_EXPERIMENT_CONFIG,
    load_experiment_config,
    write_config_snapshot,
)
from eeg_ms.experiments.runner import run_configured_models
from eeg_ms.experiments.wizard import collect_experiment_config
from eeg_ms.features.build_tabular import build_configured_feature_matrix
from eeg_ms.modeling.models import get_model_specifications
from eeg_ms.quality.run_quality_control import run_quality_control
from eeg_ms.quality.configured_feature_plots import create_configured_feature_plots
from eeg_ms.tracking import (
    environment_summary,
    file_summary,
    git_revision,
    mark_latest_run_failed,
    write_json,
    create_run_directory,
)
from eeg_ms.evaluation.bootstrap import bootstrap_confidence_intervals
from eeg_ms.evaluation.metrics import (
    aggregate_oof_predictions,
    calculate_aggregated_oof_metrics,
    summarize_fold_metrics,
)
from eeg_ms.evaluation.plots import (
    plot_bootstrap_intervals,
    plot_confusion_matrices,
    plot_fold_metric_distributions,
    plot_oof_precision_recall_curves,
    plot_oof_roc_curves,
    plot_subject_probabilities,
)


def _resolved_selection(values: list[str], available: list[str], name: str) -> list[str]:
    if "all" in values:
        return available
    unknown = set(values) - set(available)
    if unknown:
        raise ValueError(f"Selezione {name} non presente nei dati: {sorted(unknown)}")
    return values


def _evaluate_run(results_dir: Path, validation: dict) -> None:
    print("[5/5] Valutazione: aggregazione delle predizioni out-of-fold...", flush=True)
    metrics = pd.read_csv(results_dir / "fold_metrics.csv")
    predictions = pd.read_csv(results_dir / "subject_predictions.csv")
    expected_repeats = int(validation.get("n_repeats", 1))
    aggregated = aggregate_oof_predictions(predictions, expected_repeats=expected_repeats)
    tables_dir = results_dir / "evaluation" / "tables"
    figures_dir = results_dir / "evaluation" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fold_summary = summarize_fold_metrics(metrics)
    oof_metrics = calculate_aggregated_oof_metrics(aggregated)
    intervals = bootstrap_confidence_intervals(
        aggregated,
        n_bootstrap=5_000,
        confidence_level=0.95,
        random_seed=int(validation.get("random_seed", 42)),
    )
    print("[5/5] Bootstrap completato; salvo tabelle e grafici di valutazione...", flush=True)
    fold_summary.to_csv(tables_dir / "fold_metric_summary.csv", index=False)
    oof_metrics.to_csv(tables_dir / "aggregated_oof_metrics.csv", index=False)
    aggregated.to_csv(tables_dir / "subject_oof_predictions.csv", index=False)
    intervals.to_csv(tables_dir / "bootstrap_confidence_intervals.csv", index=False)

    plot_fold_metric_distributions(metrics, figures_dir / "fold_metric_distributions.png")
    plot_oof_roc_curves(aggregated, figures_dir / "oof_roc_curves.png")
    plot_oof_precision_recall_curves(aggregated, figures_dir / "oof_precision_recall_curves.png")
    plot_confusion_matrices(aggregated, figures_dir / "confusion_matrices.png")
    plot_subject_probabilities(aggregated, figures_dir / "subject_probabilities.png")
    plot_bootstrap_intervals(
        intervals,
        figures_dir / "bootstrap_balanced_accuracy.png",
        metric="balanced_accuracy",
    )
    print("[5/5] Valutazione completata.", flush=True)


def run_pipeline(config: dict) -> Path:
    """Execute audit/loading, QC, feature building, training, and evaluation."""
    run_id, run_dir = create_run_directory()
    started = datetime.now(timezone.utc)
    model_specs = get_model_specifications(
        random_seed=int(config["validation"].get("random_seed", 42)),
        selected_models=list(config["models"]),
    )
    config["resolved_models"] = [
        {
            "name": spec.name,
            "scale_features": spec.scale_features,
            "estimator_parameters": spec.estimator.get_params(),
            "parameter_grid": spec.parameter_grid,
        }
        for spec in model_specs
    ]
    write_config_snapshot(config, run_dir)
    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started.isoformat(),
        "configuration_file": "configuration.yaml",
        "git": git_revision(),
        "environment": environment_summary(),
        "configuration": config,
        "validation_config": {
            "outer_n_splits": config["validation"]["n_splits"],
            "outer_n_repeats": config["validation"].get("n_repeats", 1),
            "inner_n_splits": config["validation"].get("inner_n_splits"),
            "primary_metric": config["validation"].get("primary_metric") or "Non applicabile",
            "random_seed": config["validation"].get("random_seed", 42),
        },
        "models": config["resolved_models"],
    }
    write_json(run_dir / "metadata.json", metadata)
    try:
        print(f"Run {run_id} avviato. Output: {run_dir}", flush=True)
        print("[1/5] Audit, loading e costruzione del dataset canonico...", flush=True)
        canonical = build_canonical_dataset()
        subjects = pd.read_csv(SUBJECTS_DATA)
        print(
            f"[1/5] Dataset pronto: {subjects['subject_id'].nunique()} soggetti, "
            f"{len(canonical):,} righe canoniche.",
            flush=True,
        )
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CANONICAL_DATA, inputs_dir / CANONICAL_DATA.name)
        shutil.copy2(SUBJECTS_DATA, inputs_dir / SUBJECTS_DATA.name)
        audit_dir = inputs_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        for audit_name in ("file_manifest.csv", "variable_inventory.csv", "validation_report.json"):
            audit_path = DATA_INTERIM / audit_name
            if audit_path.exists():
                shutil.copy2(audit_path, audit_dir / audit_name)
        metadata["inputs"] = {
            "canonical_features": file_summary(CANONICAL_DATA, canonical),
            "subjects": file_summary(SUBJECTS_DATA, subjects),
        }
        metadata["subjects"] = {
            "n_subjects": int(subjects["subject_id"].nunique()),
            "group_counts": {str(key): int(value) for key, value in subjects["group"].value_counts().items()},
            "target_counts": {str(key): int(value) for key, value in subjects["target"].value_counts().items()},
        }

        print("[2/5] Controllo qualità: calcolo riepiloghi e grafici...", flush=True)
        run_quality_control(
            canonical,
            subjects,
            tables_dir=run_dir / "quality" / "tables",
            figures_dir=run_dir / "quality" / "figures",
        )
        print("[2/5] Controllo qualità completato.", flush=True)

        feature_config = config["features"]
        resolved = {
            "aggregation_unit": feature_config["aggregation_unit"],
            "statistics": feature_config.get("statistics", []),
            "condition_views": feature_config["condition_views"],
            "feature_families": _resolved_selection(
                feature_config["feature_families"], sorted(canonical["feature_family"].unique()), "famiglie"
            ),
            "measures": _resolved_selection(
                feature_config["measures"], sorted(canonical["measure"].unique()), "misure"
            ),
            "bands": _resolved_selection(
                feature_config["bands"], sorted(canonical["band"].unique()), "bande"
            ),
            "spatial_levels": _resolved_selection(
                feature_config["spatial_levels"], sorted(canonical["spatial_level"].unique()), "spatial levels"
            ),
            "locations": feature_config.get("locations", []),
        }
        if resolved["locations"] == ["all"]:
            resolved["locations"] = sorted(canonical["location"].unique())
        elif resolved["locations"]:
            unknown_locations = set(resolved["locations"]) - set(canonical["location"].unique())
            if unknown_locations:
                raise ValueError(f"Posizioni non presenti nei dati: {sorted(unknown_locations)}")
        config["resolved_features"] = resolved
        write_config_snapshot(config, run_dir)
        print(
            "[3/5] Feature engineering: "
            f"unità={resolved['aggregation_unit']}; famiglie={resolved['feature_families']}; "
            f"misure={resolved['measures']}; bande={resolved['bands']}; "
            f"spazio={resolved['spatial_levels']}; condizioni={resolved['condition_views']}; "
            f"statistiche={resolved['statistics'] or 'non applicabili'}.",
            flush=True,
        )
        matrix = build_configured_feature_matrix(canonical, subjects, **resolved)
        feature_path = run_dir / "intermediate" / "configured_features.parquet"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(feature_path, index=False, compression="zstd")
        print("[3/5] Creo grafici specifici per le feature selezionate...", flush=True)
        configured_figure_names = create_configured_feature_plots(
            matrix,
            run_dir / "quality" / "figures",
        )
        print(
            f"[3/5] Grafici specifici creati: {', '.join(configured_figure_names[:2])}.",
            flush=True,
        )
        metadata["resolved_feature_selection"] = resolved
        metadata["feature_matrix"] = {"rows": len(matrix), "n_features": len(matrix.columns) - 4 - int("window" in matrix)}
        print(
            f"[3/5] Matrice pronta: {len(matrix):,} righe × "
            f"{metadata['feature_matrix']['n_features']:,} feature; salvata in {feature_path}.",
            flush=True,
        )

        print(
            f"[4/5] Classificazione: {len(model_specs)} modelli, "
            f"{config['validation']['n_splits']} fold × "
            f"{config['validation'].get('n_repeats', 1)} ripetizioni, "
            f"strategia={config['validation']['strategy']}.",
            flush=True,
        )
        metrics, predictions, selected = run_configured_models(
            matrix,
            config,
            run_dir / "results",
        )
        _evaluate_run(run_dir / "results", config["validation"])
        metadata["outputs"] = {
            "configuration": "configuration.yaml",
            "features": str(feature_path.relative_to(run_dir)),
            "fold_metrics": "results/fold_metrics.csv",
            "subject_predictions": "results/subject_predictions.csv",
            "selected_features": "results/selected_features.csv",
            "evaluation": "results/evaluation",
        }
        metadata["status"] = "completed"
        finished = datetime.now(timezone.utc)
        metadata["finished_at_utc"] = finished.isoformat()
        metadata["duration_seconds"] = (finished - started).total_seconds()
        write_json(run_dir / "metadata.json", metadata)
        print(f"\nRun completato: {run_id}")
        print(f"Risultati: {run_dir}")
        print(pd.read_csv(run_dir / "results" / "evaluation" / "tables" / "aggregated_oof_metrics.csv").to_string(index=False))
        return run_dir
    except Exception as error:
        mark_latest_run_failed(error)
        raise


def _save_wizard_config(config: dict) -> Path:
    experiment_configs = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    experiment_configs.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", config["name"]).strip("_") or "experiment"
    path = experiment_configs / f"{stem}.yaml"
    if path.exists():
        path = experiment_configs / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    import yaml
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Configurazione salvata: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline EEG-MS configurabile")
    parser.add_argument("--config", type=Path, help="Usa una configurazione YAML senza aprire la procedura interattiva")
    parser.add_argument("--wizard", action="store_true", help="Apri la procedura guidata di configurazione")
    args = parser.parse_args()

    if args.config:
        config = load_experiment_config(args.config)
    elif args.wizard or not DEFAULT_EXPERIMENT_CONFIG.exists():
        config = collect_experiment_config()
        _save_wizard_config(config)
    else:
        config = load_experiment_config(DEFAULT_EXPERIMENT_CONFIG)
        print(f"Configurazione predefinita caricata da {DEFAULT_EXPERIMENT_CONFIG}; usare --wizard per crearne una nuova.")
    run_pipeline(config)


if __name__ == "__main__":
    main()
