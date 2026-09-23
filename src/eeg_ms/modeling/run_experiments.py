"""Confronto dei modelli mediante nested repeated CV.
Il file crea soltanto gli outer split.

Durante run_experiments.py, per ogni outer fold:

l’outer test set viene tenuto completamente da parte;
il training set viene ulteriormente diviso tramite inner CV;
l’inner CV sceglie gli iperparametri;
il modello migliore viene riaddestrato sull’intero outer training set;
viene valutato sull’outer test set.

Tutti i soggetti
├── Outer training
│   ├── Inner training
│   └── Inner validation
└── Outer test


In particolare vengono prodotti i seguenti file:
- nested_cv_fold_metrics.csv: dettaglio delle prestazioni;
- model_comparison_summary.csv: confronto sintetico;
- nested_cv_predictions.csv: comportamento sui singoli soggetti;
- nested_cv_selected_features.csv: interpretabilità e stabilità delle feature.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd

from eeg_ms.config import (
    CONFIGS,
    METRICS_DIR,
    OUTER_FOLDS_FILE,
    PREDICTIONS_DIR,
    SUBJECT_FEATURES_CHANNEL,
    SUBJECT_FEATURES_ROI,
    SUBJECTS_DATA,
    SELECTED_FEATURES_FILE,
    VALIDATION_CONFIG,
)
from eeg_ms.modeling.models import get_model_specifications
from eeg_ms.tracking import (
    copy_input,
    create_run_directory,
    environment_summary,
    file_summary,
    git_revision,
    mark_latest_run_failed,
    write_json,
)
from eeg_ms.validation.config import load_validation_config
from eeg_ms.validation.nested_cv import run_nested_cv


def run_experiment() -> None:
    started_at = datetime.now(timezone.utc)
    run_id, run_dir = create_run_directory()
    config = load_validation_config()
    subjects = pd.read_csv(SUBJECTS_DATA)
    assignments = pd.read_csv(OUTER_FOLDS_FILE)

    input_paths = {
        "subjects": SUBJECTS_DATA,
        "outer_assignments": OUTER_FOLDS_FILE,
        "subject_features_roi": SUBJECT_FEATURES_ROI,
        "subject_features_channel": SUBJECT_FEATURES_CHANNEL,
    }

    input_summaries = {}
    for input_name, input_path in input_paths.items():
        input_summaries[input_name] = file_summary(input_path)
        copy_input(input_path, run_dir, input_path.name)

    copy_input(VALIDATION_CONFIG, run_dir, VALIDATION_CONFIG.name)
    for model_config_path in (
        CONFIGS / "dummy.yaml",
        CONFIGS / "logistic_elastic_net.yaml",
        CONFIGS / "lda.yaml",
        CONFIGS / "knn.yaml",
        CONFIGS / "xgboost.yaml",
    ):
        copy_input(model_config_path, run_dir, model_config_path.name)

    model_specs = get_model_specifications(random_seed=config.random_seed)
    model_catalog = [
        {
            "name": specification.name,
            "scale_features": specification.scale_features,
            "estimator_parameters": specification.estimator.get_params(),
            "parameter_grid": specification.parameter_grid,
        }
        for specification in model_specs
    ]

    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "command": "python -m eeg_ms.modeling.run_experiments",
        "git": git_revision(),
        "environment": environment_summary(),
        "validation_config": asdict(config),
        "inputs": input_summaries,
        "subjects": {
            "n_subjects": int(subjects["subject_id"].nunique()),
            "group_counts": {
                str(key): int(value)
                for key, value in subjects["group"].value_counts().items()
            },
            "target_counts": {
                str(key): int(value)
                for key, value in subjects["target"].value_counts().items()
            },
        },
        "models": model_catalog,
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "model_catalog.json", {"models": model_catalog})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_selected_features: list[pd.DataFrame] = []

    specifications = model_specs
    feature_datasets = {"roi": SUBJECT_FEATURES_ROI, "channel": SUBJECT_FEATURES_CHANNEL}

    for spatial_level, feature_path in feature_datasets.items():
        subject_features = pd.read_parquet(feature_path)

        for specification in specifications:
            result_model_name = f"{spatial_level}__{specification.name}"
            print(f"Avvio modello: {result_model_name}")

            metrics, predictions, selected_features = run_nested_cv(
                subject_features=subject_features,
                subjects=subjects,
                assignments=assignments,
                estimator=specification.estimator,
                parameter_grid=specification.parameter_grid,
                config=config,
                model_name=result_model_name,
                scale_features=specification.scale_features,
            )

            all_selected_features.append(selected_features)
            all_metrics.append(metrics)
            all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)

    metrics_path = METRICS_DIR / "nested_cv_fold_metrics.csv"
    predictions_path = PREDICTIONS_DIR / "nested_cv_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    metric_columns = ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "f1"]

    summary = metrics.groupby("model")[metric_columns].agg(["mean", "std", "median"]).sort_values(("balanced_accuracy", "mean"), ascending=False)
    summary_path = METRICS_DIR / "model_comparison_summary.csv"
    summary.to_csv(summary_path)

    selected_features = pd.concat(all_selected_features, ignore_index=True)
    SELECTED_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    selected_features.to_csv(SELECTED_FEATURES_FILE, index=False)

    run_outputs = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "summary": summary_path,
        "selected_features": SELECTED_FEATURES_FILE,
    }
    for output_name, output_path in run_outputs.items():
        destination = run_dir / "results" / output_name / output_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, destination)

    finished_at = datetime.now(timezone.utc)
    metadata.update(
        {
            "status": "completed",
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "outputs": {
                name: str(path.relative_to(run_dir))
                for name, path in {
                    output_name: run_dir / "results" / output_name / output_path.name
                    for output_name, output_path in run_outputs.items()
                }.items()
            },
        }
    )
    write_json(run_dir / "metadata.json", metadata)

    print("\nConfronto modelli:")
    print(summary.to_string())


def main() -> None:
    try:
        run_experiment()
    except Exception as error:
        mark_latest_run_failed(error)
        raise


if __name__ == "__main__":
    main()
