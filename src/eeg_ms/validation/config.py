"""Configurazione tipizzata della strategia di validazione."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from eeg_ms.config import VALIDATION_CONFIG


@dataclass(frozen=True)
class ValidationConfig:
    random_seed: int
    outer_n_splits: int
    outer_n_repeats: int
    inner_n_splits: int
    primary_metric: str


def load_validation_config(
    path: Path = VALIDATION_CONFIG,
) -> ValidationConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return ValidationConfig(
        random_seed=int(raw["random_seed"]),
        outer_n_splits=int(raw["outer_cv"]["n_splits"]),
        outer_n_repeats=int(raw["outer_cv"]["n_repeats"]),
        inner_n_splits=int(raw["inner_cv"]["n_splits"]),
        primary_metric=str(raw["model_selection"]["primary_metric"]),
    )
