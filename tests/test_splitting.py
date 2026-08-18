import pandas as pd

from eeg_ms.validation.config import ValidationConfig
from eeg_ms.validation.splitting import (
    generate_outer_assignments,
    get_outer_split,
)


def make_subjects() -> pd.DataFrame:
    hc = pd.DataFrame(
        {
            "subject_id": [f"hc_{i:02d}" for i in range(1, 9)],
            "group": "hc",
            "target": 0,
        }
    )

    ms = pd.DataFrame(
        {
            "subject_id": [f"ms_{i:02d}" for i in range(1, 25)],
            "group": "ms",
            "target": 1,
        }
    )

    return pd.concat([hc, ms], ignore_index=True)


def make_config() -> ValidationConfig:
    return ValidationConfig(
        random_seed=42,
        outer_n_splits=4,
        outer_n_repeats=5,
        inner_n_splits=3,
        primary_metric="roc_auc",
    )


def test_no_subject_leakage():
    subjects = make_subjects()
    config = make_config()

    assignments = generate_outer_assignments(
        subjects,
        config,
    )

    for repeat in range(1, 6):
        for fold in range(1, 5):
            train, test = get_outer_split(
                subjects,
                assignments,
                repeat,
                fold,
            )

            assert set(train["subject_id"]).isdisjoint(
                set(test["subject_id"])
            )


def test_each_subject_is_test_once_per_repeat():
    subjects = make_subjects()
    config = make_config()

    assignments = generate_outer_assignments(
        subjects,
        config,
    )

    counts = assignments.groupby(
        ["repeat", "subject_id"]
    ).size()

    assert (counts == 1).all()


def test_each_test_fold_contains_both_classes():
    subjects = make_subjects()
    config = make_config()

    assignments = generate_outer_assignments(
        subjects,
        config,
    )

    class_counts = (
        assignments
        .groupby(["repeat", "fold"])["target"]
        .nunique()
    )

    assert (class_counts == 2).all()