"""Definizione centralizzata dei modelli e degli spazi di ricerca."""

from dataclasses import dataclass
from typing import Any

import yaml

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from eeg_ms.config import CONFIGS
from eeg_ms.modeling.xgboost_balanced import BalancedXGBClassifier


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]
    scale_features: bool


def _load_model_config(model_name: str) -> dict[str, Any]:
    path = CONFIGS / f"{model_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Configurazione mancante per {model_name}: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Configurazione non valida: {path}")

    if not config.get("enabled", True):
        return {}

    required_keys = {"scale_features", "estimator", "parameter_grid"}
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise ValueError(
            f"{path}: chiavi mancanti: {sorted(missing_keys)}"
        )

    if not isinstance(config["parameter_grid"], dict):
        raise ValueError(f"{path}: parameter_grid deve essere una mappa")

    return config


def _build_specification(
    model_name: str,
    estimator: BaseEstimator,
    config: dict[str, Any],
) -> ModelSpecification:
    return ModelSpecification(
        name=model_name,
        estimator=estimator,
        parameter_grid=config["parameter_grid"],
        scale_features=bool(config["scale_features"]),
    )


def get_model_specifications(
    random_seed: int,
) -> list[ModelSpecification]:
    """Restituisce i modelli attivi configurati nella cartella configs."""

    specifications: list[ModelSpecification] = []

    dummy_config = _load_model_config("dummy")
    if dummy_config:
        specifications.append(
            _build_specification(
                "dummy",
                DummyClassifier(**dummy_config["estimator"]),
                dummy_config,
            )
        )

    logistic_config = _load_model_config("logistic_elastic_net")
    if logistic_config:
        estimator_parameters = dict(logistic_config["estimator"])
        estimator_parameters["random_state"] = random_seed
        specifications.append(
            _build_specification(
                "logistic_elastic_net",
                LogisticRegression(**estimator_parameters),
                logistic_config,
            )
        )

    lda_config = _load_model_config("lda")
    if lda_config:
        specifications.append(
            _build_specification(
                "lda",
                LinearDiscriminantAnalysis(**lda_config["estimator"]),
                lda_config,
            )
        )

    knn_config = _load_model_config("knn")
    if knn_config:
        specifications.append(
            _build_specification(
                "knn",
                KNeighborsClassifier(**knn_config["estimator"]),
                knn_config,
            )
        )

    random_forest_config = _load_model_config("random_forest")
    if random_forest_config:
        estimator_parameters = dict(random_forest_config["estimator"])
        estimator_parameters["random_state"] = random_seed
        specifications.append(
            _build_specification(
                "random_forest",
                RandomForestClassifier(**estimator_parameters),
                random_forest_config,
            )
        )

    svm_config = _load_model_config("svm")
    if svm_config:
        estimator_parameters = dict(svm_config["estimator"])
        estimator_parameters["random_state"] = random_seed
        specifications.append(
            _build_specification(
                "svm",
                SVC(**estimator_parameters),
                svm_config,
            )
        )

    xgboost_config = _load_model_config("xgboost")
    if xgboost_config:
        estimator_parameters = dict(xgboost_config["estimator"])
        estimator_parameters["random_state"] = random_seed
        specifications.append(
            _build_specification(
                "xgboost",
                BalancedXGBClassifier(**estimator_parameters),
                xgboost_config,
            )
        )

    if not specifications:
        raise ValueError("Nessun modello attivo configurato in configs/")

    return specifications
