"""Run configured models with subject-safe validation and run-local outputs."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold

from eeg_ms.modeling.models import ModelSpecification, get_model_specifications
from eeg_ms.validation.nested_cv import calculate_binary_metrics
from eeg_ms.validation.preprocessing import build_ml_pipeline


def _splitter(n_splits: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _validate_group_counts(frame: pd.DataFrame, n_splits: int) -> None:
    per_subject = frame[["subject_id", "target"]].drop_duplicates()
    if per_subject["subject_id"].duplicated().any():
        raise ValueError("Un soggetto è associato a target multipli.")
    counts = per_subject.groupby("target")["subject_id"].nunique()
    if counts.min() < n_splits:
        raise ValueError(
            f"La classe meno rappresentata ha {counts.min()} soggetti; "
            f"non può sostenere {n_splits} fold stratificati."
        )


def _get_probability(estimator, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(frame)[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(frame)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
    else:
        raise TypeError(f"{type(estimator).__name__} non espone probabilità o decision_function.")
    return np.asarray(probabilities, dtype=float)


def _selected_feature_rows(
    pipeline,
    model_name: str,
    repeat: int,
    fold: int,
    feature_columns: list[str],
) -> list[dict]:
    selector = pipeline.named_steps.get("selector")
    if selector is None or not hasattr(selector, "get_support"):
        return []
    imputer = pipeline.named_steps.get("imputer")
    expanded_names = (
        np.asarray(imputer.get_feature_names_out(feature_columns))
        if imputer is not None and hasattr(imputer, "get_feature_names_out")
        else np.asarray(feature_columns)
    )
    names = expanded_names[selector.get_support()]
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).reshape(-1)
    else:
        values = np.full(len(names), np.nan)
    return [
        {
            "model": model_name,
            "repeat": repeat,
            "fold": fold,
            "feature": str(name),
            "importance": float(value),
            "absolute_importance": float(abs(value)) if np.isfinite(value) else np.nan,
        }
        for name, value in zip(names, values, strict=True)
    ]


def run_configured_models(
    feature_frame: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run selected classifiers; epoch rows are always split by subject groups."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = config["validation"]
    n_splits = int(validation["n_splits"])
    n_repeats = int(validation.get("n_repeats", 1))
    _validate_group_counts(feature_frame, n_splits)

    excluded = {"subject_id", "group", "target", "visit", "window"}
    feature_columns = [column for column in feature_frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("Nessuna feature selezionata per la classificazione.")
    X = feature_frame[feature_columns]
    y = feature_frame["target"].astype(int).to_numpy()
    groups = feature_frame["subject_id"].astype(str).to_numpy()
    epoch_mode = config["features"]["aggregation_unit"] == "epoch"

    specifications = get_model_specifications(
        random_seed=int(validation.get("random_seed", 42)),
        selected_models=list(config["models"]),
    )
    if not specifications:
        raise ValueError("Nessuno dei modelli selezionati è abilitato nella configurazione.")

    metric_rows: list[dict] = []
    subject_prediction_rows: list[dict] = []
    epoch_prediction_rows: list[dict] = []
    selected_rows: list[dict] = []
    strategy = validation["strategy"]

    total_folds = n_splits * n_repeats
    n_models = len(specifications)
    print(
        f"  ML: {len(X):,} righe, {len(feature_columns):,} feature, "
        f"{len(np.unique(groups))} soggetti; valutazione a livello soggetto.",
        flush=True,
    )
    for model_index, spec in enumerate(specifications, start=1):
        model_name = spec.name
        print(f"  ML {model_index}/{n_models}: modello {model_name}...", flush=True)
        for repeat in range(1, n_repeats + 1):
            outer_cv = _splitter(n_splits, int(validation.get("random_seed", 42)) + repeat * 100)
            outer_splits = list(outer_cv.split(X, y, groups))
            for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
                fold_number = (repeat - 1) * n_splits + fold
                fold_started = perf_counter()
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                groups_train, groups_test = groups[train_idx], groups[test_idx]
                if set(groups_train) & set(groups_test):
                    raise RuntimeError("Data leakage: un soggetto compare in train e test.")
                if np.unique(y_test).size < 2:
                    raise ValueError(
                        f"Outer fold senza entrambe le classi (repeat={repeat}, fold={fold}); "
                        "ridurre i fold o verificare la distribuzione dei soggetti."
                    )

                pipeline = build_ml_pipeline(spec.estimator, scale_features=spec.scale_features)
                best_params: dict = {}
                inner_score = np.nan
                if strategy == "nested":
                    inner_n = int(validation["inner_n_splits"])
                    _validate_group_counts(
                        pd.DataFrame({"subject_id": groups_train, "target": y_train}), inner_n
                    )
                    inner_cv = _splitter(inner_n, int(validation.get("random_seed", 42)) + repeat * 1000 + fold)
                    inner_splits = list(inner_cv.split(X_train, y_train, groups_train))
                    if any(np.unique(y_train[valid_idx]).size < 2 for _, valid_idx in inner_splits):
                        raise ValueError(
                            f"Inner fold senza entrambe le classi per {model_name}, "
                            f"repeat={repeat}, fold={fold}. Ridurre il numero di fold o verificare i gruppi."
                        )
                    n_candidates = len(ParameterGrid(spec.parameter_grid))
                    print(
                        f"    Fold {fold_number}/{total_folds}: tuning di {n_candidates} configurazioni × "
                        f"{inner_n} fold interni; train {len(set(groups_train))} soggetti, "
                        f"test {len(set(groups_test))}.",
                        flush=True,
                    )
                    search = GridSearchCV(
                        pipeline,
                        param_grid=spec.parameter_grid,
                        scoring=str(validation.get("primary_metric", "roc_auc")),
                        cv=inner_splits,
                        refit=True,
                        # Evita processi loky e directory di memmapping temporanee
                        # che possono generare warning di cleanup su Windows.
                        n_jobs=1,
                        error_score="raise",
                    )
                    search.fit(X_train, y_train, groups=groups_train)
                    fitted = search.best_estimator_
                    best_params = search.best_params_
                    inner_score = float(search.best_score_)
                else:
                    print(
                        f"    Fold {fold_number}/{total_folds}: fit senza tuning; "
                        f"train {len(set(groups_train))} soggetti, test {len(set(groups_test))}.",
                        flush=True,
                    )
                    pipeline.fit(X_train, y_train)
                    fitted = pipeline

                probability = _get_probability(fitted, X_test)
                test_rows = feature_frame.iloc[test_idx].copy()
                if epoch_mode:
                    epoch_prediction_rows.extend(
                        {
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "subject_id": row.subject_id,
                            "window": int(row.window),
                            "target": int(row.target),
                            "probability_ms": float(prob),
                        }
                        for row, prob in zip(test_rows.itertuples(), probability, strict=True)
                    )
                per_subject = pd.DataFrame({
                    "subject_id": test_rows["subject_id"].to_numpy(),
                    "target": y_test,
                    "probability_ms": probability,
                }).groupby(["subject_id", "target"], as_index=False).agg(
                    probability_ms=("probability_ms", "mean")
                )
                fold_metrics = calculate_binary_metrics(
                    per_subject["target"].to_numpy(),
                    per_subject["probability_ms"].to_numpy(),
                )
                elapsed = perf_counter() - fold_started
                print(
                    f"    Fold {fold_number}/{total_folds} completato in {elapsed:.1f}s — "
                    f"AUC {fold_metrics['roc_auc']:.3f}, balanced accuracy "
                    f"{fold_metrics['balanced_accuracy']:.3f}, sensibilità "
                    f"{fold_metrics['sensitivity']:.3f}, specificità "
                    f"{fold_metrics['specificity']:.3f}.",
                    flush=True,
                )
                metric_rows.append({
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    **fold_metrics,
                    "inner_score": inner_score,
                    "best_parameters": json.dumps(best_params, sort_keys=True),
                    "n_test_subjects": int(per_subject["subject_id"].nunique()),
                    "n_test_rows": int(len(test_idx)),
                })
                subject_prediction_rows.extend(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "subject_id": row.subject_id,
                        "target": int(row.target),
                        "probability_ms": float(row.probability_ms),
                    }
                    for row in per_subject.itertuples(index=False)
                )
                selected_rows.extend(
                    _selected_feature_rows(fitted, model_name, repeat, fold, feature_columns)
                )
        print(f"  ML {model_index}/{n_models}: {model_name} completato.", flush=True)

    metrics = pd.DataFrame(metric_rows)
    subject_predictions = pd.DataFrame(subject_prediction_rows)
    epoch_predictions = pd.DataFrame(epoch_prediction_rows)
    selected_features = pd.DataFrame(selected_rows)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    subject_predictions.to_csv(output_dir / "subject_predictions.csv", index=False)
    if epoch_mode:
        epoch_predictions.to_csv(output_dir / "epoch_predictions.csv", index=False)
    selected_features.to_csv(output_dir / "selected_features.csv", index=False)
    print(f"  ML: completato. File dei risultati salvati in {output_dir}.", flush=True)
    return metrics, subject_predictions, selected_features
