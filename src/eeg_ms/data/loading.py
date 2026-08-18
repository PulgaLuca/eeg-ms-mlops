"""Caricamento dei file MATLAB senza trasformazioni statistiche."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from eeg_ms.config import N_ROIS
from eeg_ms.data.parsing import FileMetadata


def load_feature_file(
    metadata: FileMetadata,
) -> dict[str, np.ndarray]:
    """Carica tutte le variabili utente presenti in un file di feature."""

    try:
        raw = loadmat(
            metadata.path,
            simplify_cells=True,
        )
    except NotImplementedError as exc:
        raise ValueError(
            f"{metadata.path} sembra essere MATLAB v7.3/HDF5. "
            "Questo dataset richiederebbe h5py."
        ) from exc

    return {
        name: np.asarray(value)
        for name, value in raw.items()
        if not name.startswith("__")
    }


def load_channel_metadata(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Carica ordine dei canali e composizione delle ROI."""

    raw: dict[str, Any] = loadmat(path, simplify_cells=True)

    chanlocs = raw.get("chanlocs")
    if chanlocs is None:
        raise KeyError(f"Variabile 'chanlocs' assente in {path}")

    channels = [str(channel["labels"]) for channel in chanlocs]

    rois: dict[str, list[str]] = {}
    for index in range(1, N_ROIS + 1):
        roi_name = f"R{index}"

        if roi_name not in raw:
            raise KeyError(f"Variabile {roi_name!r} assente in {path}")

        values = np.atleast_1d(raw[roi_name])
        rois[roi_name] = [str(value) for value in values.tolist()]

    return channels, rois