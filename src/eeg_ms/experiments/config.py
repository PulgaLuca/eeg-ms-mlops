"""Loading, validation, and persistence for user experiment configurations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from eeg_ms.config import EXPERIMENT_CONFIGS


DEFAULT_EXPERIMENT_CONFIG = EXPERIMENT_CONFIGS / "default.yaml"


def load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    required = {"name", "features", "validation", "models"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Configurazione senza sezioni: {sorted(missing)}")
    features = config["features"]
    if features.get("aggregation_unit") not in {"subject", "epoch"}:
        raise ValueError("features.aggregation_unit deve essere subject oppure epoch.")
    for key in ("feature_families", "measures", "bands", "spatial_levels", "condition_views"):
        if not features.get(key):
            raise ValueError(f"Selezionare almeno un valore in features.{key}.")
    if features["aggregation_unit"] == "subject" and not features.get("statistics"):
        raise ValueError("La modalità subject richiede almeno una statistica.")
    if features["aggregation_unit"] == "epoch" and features.get("statistics"):
        raise ValueError("In modalità epoch statistics deve essere vuoto: ogni finestra è una riga.")
    allowed_views = {"CE", "OE", "CE_minus_OE"}
    if set(features["condition_views"]) - allowed_views:
        raise ValueError("condition_views contiene valori non supportati.")
    validation = config["validation"]
    if validation.get("strategy") not in {"nested", "stratified"}:
        raise ValueError("validation.strategy deve essere nested oppure stratified.")
    if int(validation.get("n_splits", 0)) < 2:
        raise ValueError("validation.n_splits deve essere almeno 2.")
    if int(validation.get("n_repeats", 1)) < 1:
        raise ValueError("validation.n_repeats deve essere almeno 1.")
    if validation["strategy"] == "nested" and int(validation.get("inner_n_splits", 0)) < 2:
        raise ValueError("Nested CV richiede inner_n_splits almeno 2.")
    if not config["models"]:
        raise ValueError("Selezionare almeno un modello.")
def write_config_snapshot(config: dict[str, Any], run_dir: Path) -> Path:
    path = run_dir / "configuration.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    (run_dir / "configuration.sha256").write_text(hashlib.sha256(canonical).hexdigest() + "\n", encoding="utf-8")
    return path
