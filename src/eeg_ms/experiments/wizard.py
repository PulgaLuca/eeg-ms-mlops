"""Small terminal wizard that writes a reusable experiment YAML."""

from __future__ import annotations

from typing import Sequence
from eeg_ms.config import CHANNELS_FILE, N_ROIS
from eeg_ms.data.loading import load_channel_metadata
from eeg_ms.experiments.config import validate_experiment_config


MODELS = ["dummy", "logistic_elastic_net", "lda", "knn", "random_forest", "svm", "xgboost"]
STATISTICS = ["mean", "median", "q25", "q75", "iqr", "std", "min", "max"]
CONDITIONS = ["CE", "OE", "CE_minus_OE"]
SELECTION_METRICS = ["roc_auc", "balanced_accuracy", "average_precision"]


def _choose(title: str, options: Sequence[str], *, multiple: bool) -> list[str]:
    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    suffix = "numeri separati da virgola, oppure 'all'" if multiple else "numero"
    while True:
        raw = input(f"Selezione ({suffix}): ").strip().lower()
        if multiple and raw in {"all", "tutte", "tutti"}:
            return list(options)
        try:
            indexes = [int(part.strip()) for part in raw.split(",")]
            if not multiple and len(indexes) != 1:
                raise ValueError
            selected = list(dict.fromkeys(options[index - 1] for index in indexes))
            if selected and all(1 <= index <= len(options) for index in indexes):
                return selected
        except (ValueError, IndexError):
            pass
        print("Selezione non valida; riprova.")


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if not raw else int(raw)


def _choose_locations(spatial_levels: list[str]) -> list[str]:
    if not CHANNELS_FILE.exists():
        print("Metadati dei canali non ancora disponibili: locations vuoto significa tutte le posizioni; i nomi specifici si possono impostare nel YAML dopo l'audit.")
        return []
    channels, _ = load_channel_metadata(CHANNELS_FILE)
    available = []
    if "channel" in spatial_levels:
        available.extend(channels)
    if "roi" in spatial_levels:
        available.extend(f"R{index}" for index in range(1, N_ROIS + 1))
    return _choose("Posizioni (facoltativo)", ["tutte", *available], multiple=True)[1:] if available else []


def collect_experiment_config() -> dict:
    print("Configurazione esperimento EEG-MS")
    name = input("Nome esperimento [experiment]: ").strip() or "experiment"
    unit = _choose("Unità osservazionale", ["subject", "epoch"], multiple=False)[0]
    if unit == "epoch":
        print("Ogni finestra diventa una riga; le finestre dello stesso soggetto restano unite negli split. Le statistiche tra finestre non si applicano.")
    statistic_options = _choose("Statistiche tra finestre", STATISTICS, multiple=True) if unit == "subject" else []
    families = _choose("Famiglie disponibili", ["psd", "complexity"], multiple=True)
    measures = _choose(
        "Misure disponibili",
        ["absolute_power", "relative_power", "theta_alpha_ratio", "delta_alpha_ratio", "sfr", "spectral_entropy"],
        multiple=True,
    )
    bands = _choose("Bande", ["delta", "theta", "alpha", "beta", "gamma", "derived", "broadband"], multiple=True)
    spatial = _choose("Livello spaziale", ["channel", "roi"], multiple=True)
    locations = _choose_locations(spatial)
    conditions = _choose("Viste di condizione", CONDITIONS, multiple=True)
    models = _choose("Modelli da eseguire", MODELS, multiple=True)
    strategy = _choose("Validazione", ["nested", "stratified"], multiple=False)[0]
    n_splits = _ask_int("Fold esterni", 4)
    repeats = _ask_int("Ripetizioni esterne", 5)
    inner_splits = _ask_int("Fold interni (solo nested)", 3) if strategy == "nested" else None
    primary_metric = (
        _choose("Metrica per la selezione interna", SELECTION_METRICS, multiple=False)[0]
        if strategy == "nested"
        else None
    )

    config = {
        "name": name,
        "features": {
            "aggregation_unit": unit,
            "statistics": statistic_options,
            "feature_families": families,
            "measures": measures,
            "bands": bands,
            "spatial_levels": spatial,
            "locations": locations,
            "condition_views": conditions,
        },
        "validation": {
            "strategy": strategy,
            "n_splits": n_splits,
            "n_repeats": repeats,
            "inner_n_splits": inner_splits,
            "primary_metric": primary_metric,
            "random_seed": 42,
        },
        "models": models,
    }
    validate_experiment_config(config)
    return config
