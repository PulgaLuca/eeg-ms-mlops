"""Genera e salva gli outer fold a livello soggetto."""

import pandas as pd

from eeg_ms.config import (
    OUTER_FOLDS_FILE,
    SPLITS_DIR,
    SUBJECTS_DATA,
)
from eeg_ms.validation.config import load_validation_config
from eeg_ms.validation.splitting import generate_outer_assignments


def main() -> None:
    config = load_validation_config()
    subjects = pd.read_csv(SUBJECTS_DATA)

    assignments = generate_outer_assignments(
        subjects=subjects,
        config=config,
    )

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_csv(
        OUTER_FOLDS_FILE,
        index=False,
    )

    summary = (
        assignments
        .groupby(
            ["repeat", "fold", "group"],
            observed=True,
        )
        .size()
        .rename("n_test_subjects")
        .reset_index()
    )

    print(summary.to_string(index=False))
    print(f"\nSplit salvati in: {OUTER_FOLDS_FILE}")


if __name__ == "__main__":
    main()