"""Split stratificati e riproducibili a livello soggetto."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from eeg_ms.validation.config import ValidationConfig


REQUIRED_SUBJECT_COLUMNS = {
    "subject_id",
    "group",
    "target",
}


def validate_subject_table(subjects: pd.DataFrame) -> None:
    """Verifica che subjects.csv abbia una riga per soggetto."""

    missing_columns = REQUIRED_SUBJECT_COLUMNS - set(subjects.columns)
    if missing_columns:
        raise ValueError(
            f"Colonne mancanti in subjects.csv: "
            f"{sorted(missing_columns)}"
        )

    if subjects["subject_id"].isna().any():
        raise ValueError("Sono presenti subject_id mancanti.")

    if subjects["subject_id"].duplicated().any():
        duplicates = subjects.loc[
            subjects["subject_id"].duplicated(keep=False),
            ["subject_id", "group", "target"],
        ]

        raise ValueError(
            "subjects.csv deve avere una sola riga per soggetto:\n"
            f"{duplicates.to_string(index=False)}"
        )

    expected_targets = {
        "hc": 0,
        "ms": 1,
    }

    invalid_target = subjects.apply(
        lambda row: expected_targets.get(row["group"]) != row["target"],
        axis=1,
    )

    if invalid_target.any():
        raise ValueError(
            "Incoerenza tra group e target in subjects.csv."
        )

    if subjects["target"].nunique() != 2:
        raise ValueError(
            "La classificazione richiede esattamente due classi."
        )


def check_split_feasibility(
    targets: pd.Series,
    n_splits: int,
    context: str,
) -> None:
    """Ogni classe deve avere almeno n_splits soggetti."""

    counts = targets.value_counts()

    if counts.min() < n_splits:
        raise ValueError(
            f"{context}: la classe meno rappresentata contiene "
            f"{counts.min()} soggetti, ma sono richiesti "
            f"{n_splits} fold."
        )


def generate_outer_assignments(
    subjects: pd.DataFrame,
    config: ValidationConfig,
) -> pd.DataFrame:
    """
    Genera l'assegnazione dei soggetti agli outer test fold.

    Ogni soggetto compare una volta come test in ogni ripetizione.
    """

    subjects = (
        subjects
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    validate_subject_table(subjects)

    check_split_feasibility(
        targets=subjects["target"],
        n_splits=config.outer_n_splits,
        context="Outer CV",
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=config.outer_n_splits,
        n_repeats=config.outer_n_repeats,
        random_state=config.random_seed,
    )

    dummy_features = np.zeros((len(subjects), 1))
    rows: list[dict] = []

    for split_number, (_, test_indices) in enumerate(
        splitter.split(
            dummy_features,
            subjects["target"],
        )
    ):
        repeat = (
            split_number // config.outer_n_splits
        ) + 1

        fold = (
            split_number % config.outer_n_splits
        ) + 1

        test_subjects = subjects.iloc[test_indices]

        for row in test_subjects.itertuples(index=False):
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "subject_id": row.subject_id,
                    "group": row.group,
                    "target": row.target,
                }
            )

    assignments = pd.DataFrame(rows)

    validate_outer_assignments(
        assignments=assignments,
        subjects=subjects,
        config=config,
    )

    return assignments


def validate_outer_assignments(
    assignments: pd.DataFrame,
    subjects: pd.DataFrame,
    config: ValidationConfig,
) -> None:
    """Controlla copertura, stratificazione e unicità dei fold."""

    expected_subjects = set(subjects["subject_id"])

    for repeat in range(1, config.outer_n_repeats + 1):
        repeat_rows = assignments.loc[
            assignments["repeat"] == repeat
        ]

        counts = repeat_rows["subject_id"].value_counts()

        if set(counts.index) != expected_subjects:
            raise ValueError(
                f"Ripetizione {repeat}: copertura soggetti incompleta."
            )

        if not (counts == 1).all():
            raise ValueError(
                f"Ripetizione {repeat}: un soggetto compare in "
                "più outer test fold."
            )

        for fold in range(1, config.outer_n_splits + 1):
            fold_rows = repeat_rows.loc[
                repeat_rows["fold"] == fold
            ]

            if fold_rows["target"].nunique() != 2:
                raise ValueError(
                    f"Repeat {repeat}, fold {fold}: "
                    "outer test senza entrambe le classi."
                )


def get_outer_split(
    subjects: pd.DataFrame,
    assignments: pd.DataFrame,
    repeat: int,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ricostruisce outer training e outer test tramite subject_id."""

    test_ids = set(
        assignments.loc[
            (assignments["repeat"] == repeat)
            & (assignments["fold"] == fold),
            "subject_id",
        ]
    )

    is_test = subjects["subject_id"].isin(test_ids)

    train_subjects = subjects.loc[~is_test].copy()
    test_subjects = subjects.loc[is_test].copy()

    assert_no_subject_leakage(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
    )

    return train_subjects, test_subjects


def assert_no_subject_leakage(
    train_subjects: pd.DataFrame,
    test_subjects: pd.DataFrame,
) -> None:
    """Fallisce se un soggetto compare in entrambi gli insiemi."""

    train_ids = set(train_subjects["subject_id"])
    test_ids = set(test_subjects["subject_id"])

    overlap = train_ids & test_ids

    if overlap:
        raise RuntimeError(
            f"Data leakage: soggetti presenti in train e test: "
            f"{sorted(overlap)}"
        )


def make_inner_cv(
    training_targets: pd.Series,
    config: ValidationConfig,
    repeat: int,
    fold: int,
) -> StratifiedKFold:
    """Crea l'inner CV usando esclusivamente l'outer training."""

    check_split_feasibility(
        targets=training_targets,
        n_splits=config.inner_n_splits,
        context=f"Inner CV, repeat={repeat}, fold={fold}",
    )

    inner_seed = (
        config.random_seed
        + repeat * 1_000
        + fold
    )

    return StratifiedKFold(
        n_splits=config.inner_n_splits,
        shuffle=True,
        random_state=inner_seed,
    )


def iter_outer_splits(
    subjects: pd.DataFrame,
    assignments: pd.DataFrame,
    config: ValidationConfig,
) -> Iterator[
    tuple[int, int, pd.DataFrame, pd.DataFrame]
]:
    """Itera su tutti gli outer split salvati."""

    for repeat in range(1, config.outer_n_repeats + 1):
        for fold in range(1, config.outer_n_splits + 1):
            train_subjects, test_subjects = get_outer_split(
                subjects=subjects,
                assignments=assignments,
                repeat=repeat,
                fold=fold,
            )

            yield (
                repeat,
                fold,
                train_subjects,
                test_subjects,
            )
