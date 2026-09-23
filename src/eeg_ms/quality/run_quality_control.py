"""Esegue l'EDA iniziale e salva tabelle e figure."""

import pandas as pd

from eeg_ms.config import (
    CANONICAL_DATA,
    QC_FIGURES,
    QC_TABLES,
    SUBJECTS_DATA,
)
from eeg_ms.quality.plots import (
    plot_condition_differences,
    plot_correlation_heatmap,
    plot_group_balance,
    plot_subject_distributions,
)
from eeg_ms.quality.summaries import (
    condition_differences,
    missingness_summary,
    robust_outlier_report,
    subject_correlation_matrix,
    subject_level_summary,
)


def main() -> None:
    canonical = pd.read_parquet(CANONICAL_DATA)
    subjects = pd.read_csv(SUBJECTS_DATA)

    QC_TABLES.mkdir(parents=True, exist_ok=True)
    QC_FIGURES.mkdir(parents=True, exist_ok=True)

    subject_summary = subject_level_summary(canonical)
    missingness = missingness_summary(canonical)
    outliers = robust_outlier_report(subject_summary)
    paired = condition_differences(subject_summary)

    subject_summary.to_parquet(QC_TABLES/"subject_level_summary.parquet", index=False,)
    missingness.to_csv(QC_TABLES/"missingness_summary.csv", index=False,)
    outliers.to_csv(QC_TABLES/"technical_outlier_candidates.csv", index=False,)
    paired.to_parquet(QC_TABLES/"condition_differences.parquet",index=False,)

    plot_group_balance(subjects, QC_FIGURES/"group_balance.png",)
    plot_subject_distributions(
        subject_summary,
        measure="relative_power",
        band="alpha",
        spatial_level="roi",
        condition="CE",
        output_path=(QC_FIGURES/"alpha_relative_roi_ce_distribution.png"),
    )

    plot_condition_differences(
        paired,
        measure="relative_power",
        band="alpha",
        spatial_level="roi",
        output_path=(QC_FIGURES/"alpha_relative_roi_ce_minus_oe.png"),
    )

    correlations = subject_correlation_matrix(
        subject_summary,
        condition="CE",
        spatial_level="roi",
    )

    correlations.to_csv(QC_TABLES/"roi_correlations_ce.csv")

    plot_correlation_heatmap(correlations, QC_FIGURES / "roi_correlations_ce.png",)

    print(f"Soggetti: {subjects.shape[0]}")
    print(f"Righe canoniche: {canonical.shape[0]:,}")
    print(f"Possibili outlier tecnici: {outliers.shape[0]}")


if __name__ == "__main__":
    main()