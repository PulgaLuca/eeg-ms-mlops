"""Topomap EEG costruite dal dataset canonico e da canali.mat."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat


Reducer = str | Callable[[pd.Series], float]


def load_eeg_info(
    channels_file: Path,
    *,
    head_radius_m: float = 0.095,
) -> mne.Info:
    """Crea un oggetto MNE Info dalle coordinate EEGLAB in canali.mat.

    Le coordinate X/Y/Z del file sono su una sfera unitaria. Il fattore
    ``head_radius_m`` le porta a una scala realistica in metri, senza alterare
    la geometria relativa usata per l'interpolazione topografica.
    """

    raw = loadmat(channels_file, simplify_cells=True)
    chanlocs = raw.get("chanlocs")
    if chanlocs is None:
        raise KeyError(f"Variabile 'chanlocs' assente in {channels_file}")

    if isinstance(chanlocs, dict):
        chanlocs = [chanlocs]

    ch_names: list[str] = []
    ch_pos: dict[str, np.ndarray] = {}

    for channel in chanlocs:
        name = str(channel["labels"])
        xyz = np.asarray(
            [channel["X"], channel["Y"], channel["Z"]],
            dtype=float,
        )

        if not np.isfinite(xyz).all():
            raise ValueError(f"Coordinate non valide per il canale {name}")

        ch_names.append(name)
        ch_pos[name] = xyz * head_radius_m

    if len(ch_names) != len(set(ch_names)):
        raise ValueError("Etichette duplicate in chanlocs")

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame="head",
    )

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=1.0,
        ch_types="eeg",
    )
    info.set_montage(montage)
    return info


def _apply_transform(values: pd.Series, transform: str | None) -> pd.Series:
    if transform is None:
        return values.astype(float)

    if transform == "log10":
        if (values <= 0).any():
            raise ValueError("log10 richiede valori strettamente positivi")
        return np.log10(values.astype(float))

    raise ValueError(f"Trasformazione non supportata: {transform}")


def _filter_channel_data(
    canonical: pd.DataFrame,
    *,
    measure: str,
    band: str,
    condition: str | None = None,
    group: str | None = None,
    subject_id: str | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    mask = (
        canonical["spatial_level"].eq("channel")
        & canonical["measure"].eq(measure)
        & canonical["band"].eq(band)
    )

    if condition is not None:
        mask &= canonical["condition"].eq(condition)
    if group is not None:
        mask &= canonical["group"].eq(group)
    if subject_id is not None:
        mask &= canonical["subject_id"].eq(subject_id)
    if window is not None:
        mask &= canonical["window"].eq(window)

    subset = canonical.loc[
        mask,
        ["subject_id", "group", "condition", "location", "window", "value"],
    ].copy()

    if subset.empty:
        raise ValueError(
            "Nessun dato per i filtri richiesti: "
            f"measure={measure}, band={band}, condition={condition}, "
            f"group={group}, subject_id={subject_id}, window={window}"
        )

    return subset


def _ordered_values(
    values_by_channel: pd.Series,
    ch_names: Sequence[str],
) -> np.ndarray:
    ordered = values_by_channel.reindex(ch_names)

    if ordered.isna().any():
        missing = ordered.index[ordered.isna()].tolist()
        raise ValueError(f"Valori mancanti per i canali: {missing}")

    return ordered.to_numpy(dtype=float)


def subject_channel_map(
    canonical: pd.DataFrame,
    info: mne.Info,
    *,
    subject_id: str,
    condition: str,
    measure: str,
    band: str,
    window_statistic: Reducer = "median",
    window: int | None = None,
    transform: str | None = None,
) -> np.ndarray:
    """Valori topografici di un soggetto, su una finestra o aggregati."""

    subset = _filter_channel_data(
        canonical,
        subject_id=subject_id,
        condition=condition,
        measure=measure,
        band=band,
        window=window,
    )
    subset["value"] = _apply_transform(subset["value"], transform)

    values = subset.groupby("location", observed=True)["value"].agg(
        window_statistic
    )
    return _ordered_values(values, info.ch_names)


def group_channel_map(
    canonical: pd.DataFrame,
    info: mne.Info,
    *,
    group: str,
    condition: str,
    measure: str,
    band: str,
    window_statistic: Reducer = "median",
    subject_statistic: Reducer = "mean",
    transform: str | None = None,
) -> np.ndarray:
    """Topomap di gruppo con uguale peso assegnato a ogni soggetto."""

    subset = _filter_channel_data(
        canonical,
        group=group,
        condition=condition,
        measure=measure,
        band=band,
    )
    subset["value"] = _apply_transform(subset["value"], transform)

    subject_values = (
        subset.groupby(
            ["subject_id", "location"],
            observed=True,
        )["value"]
        .agg(window_statistic)
        .rename("subject_value")
        .reset_index()
    )

    group_values = subject_values.groupby(
        "location",
        observed=True,
    )["subject_value"].agg(subject_statistic)

    return _ordered_values(group_values, info.ch_names)


def paired_condition_map(
    canonical: pd.DataFrame,
    info: mne.Info,
    *,
    group: str,
    measure: str,
    band: str,
    window_statistic: Reducer = "median",
    subject_statistic: Reducer = "mean",
    contrast: str = "CE_minus_OE",
    transform: str | None = None,
) -> np.ndarray:
    """Topomap della differenza CE-OE o OE-CE, calcolata per soggetto."""

    subset = _filter_channel_data(
        canonical,
        group=group,
        measure=measure,
        band=band,
    )
    subset["value"] = _apply_transform(subset["value"], transform)

    subject_condition = (
        subset.groupby(
            ["subject_id", "condition", "location"],
            observed=True,
        )["value"]
        .agg(window_statistic)
        .rename("subject_value")
        .reset_index()
    )

    paired = subject_condition.pivot(
        index=["subject_id", "location"],
        columns="condition",
        values="subject_value",
    ).dropna(subset=["CE", "OE"])

    if contrast == "CE_minus_OE":
        paired["difference"] = paired["CE"] - paired["OE"]
    elif contrast == "OE_minus_CE":
        paired["difference"] = paired["OE"] - paired["CE"]
    else:
        raise ValueError(f"Contrasto non supportato: {contrast}")

    group_difference = paired.groupby(
        "location",
        observed=True,
    )["difference"].agg(subject_statistic)

    return _ordered_values(group_difference, info.ch_names)


def plot_topomap(
    values: np.ndarray,
    info: mne.Info,
    *,
    title: str,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    vlim: tuple[float, float] | None = None,
    colorbar: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Disegna una singola topomap con MNE."""

    if ax is None:
        figure, ax = plt.subplots(figsize=(5, 4.5))
    else:
        figure = ax.figure

    image, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        **({"vlim": vlim} if vlim is not None else {}),
        sensors=True,
        contours=6,
        outlines="head",
    )
    ax.set_title(title)

    if colorbar:
        figure.colorbar(image, ax=ax, shrink=0.75)

    return figure, ax


def plot_group_comparison(
    canonical: pd.DataFrame,
    info: mne.Info,
    *,
    condition: str,
    measure: str,
    band: str,
    transform: str | None = None,
) -> plt.Figure:
    """HC, MS e differenza MS-HC con scale confrontabili."""

    hc = group_channel_map(
        canonical,
        info,
        group="hc",
        condition=condition,
        measure=measure,
        band=band,
        transform=transform,
    )
    ms = group_channel_map(
        canonical,
        info,
        group="ms",
        condition=condition,
        measure=measure,
        band=band,
        transform=transform,
    )
    difference = ms - hc

    common_vlim = (float(min(hc.min(), ms.min())), float(max(hc.max(), ms.max())))
    difference_limit = float(np.max(np.abs(difference)))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    plot_topomap(hc, info, title=f"HC · {condition}", ax=axes[0], vlim=common_vlim)
    plot_topomap(ms, info, title=f"MS · {condition}", ax=axes[1], vlim=common_vlim)
    plot_topomap(
        difference,
        info,
        title="MS − HC",
        ax=axes[2],
        cmap="RdBu_r",
        vlim=(-difference_limit, difference_limit),
    )
    figure.suptitle(f"{measure} · {band}")
    figure.tight_layout()
    return figure


def plot_condition_contrasts(
    canonical: pd.DataFrame,
    info: mne.Info,
    *,
    measure: str,
    band: str,
    transform: str | None = None,
) -> plt.Figure:
    """Differenza appaiata CE-OE per HC e MS."""

    hc = paired_condition_map(
        canonical,
        info,
        group="hc",
        measure=measure,
        band=band,
        transform=transform,
    )
    ms = paired_condition_map(
        canonical,
        info,
        group="ms",
        measure=measure,
        band=band,
        transform=transform,
    )
    limit = float(np.max(np.abs(np.concatenate([hc, ms]))))

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    plot_topomap(
        hc,
        info,
        title="HC · CE − OE",
        ax=axes[0],
        cmap="RdBu_r",
        vlim=(-limit, limit),
    )
    plot_topomap(
        ms,
        info,
        title="MS · CE − OE",
        ax=axes[1],
        cmap="RdBu_r",
        vlim=(-limit, limit),
    )
    figure.suptitle(f"Contrasto appaiato · {measure} · {band}")
    figure.tight_layout()
    return figure

