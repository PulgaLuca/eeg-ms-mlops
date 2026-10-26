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


def run_quality_control(
    canonical: pd.DataFrame,
    subjects: pd.DataFrame,
    tables_dir=QC_TABLES,
    figures_dir=QC_FIGURES,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("  QC 1/5: riepilogo subject-level...", flush=True)
    subject_summary = subject_level_summary(canonical)
    print("  QC 2/5: dati mancanti, outlier candidati e differenze tra condizioni...", flush=True)
    missingness = missingness_summary(canonical)
    outliers = robust_outlier_report(subject_summary)
    paired = condition_differences(subject_summary)

    subject_summary.to_parquet(tables_dir/"subject_level_summary.parquet", index=False,)
    missingness.to_csv(tables_dir/"missingness_summary.csv", index=False,)
    outliers.to_csv(tables_dir/"technical_outlier_candidates.csv", index=False,)
    paired.to_parquet(tables_dir/"condition_differences.parquet",index=False,)

    print("  QC 3/5: grafici di bilanciamento e distribuzione...", flush=True)
    plot_group_balance(subjects, figures_dir/"group_balance.png",)
    plot_subject_distributions(
        subject_summary,
        measure="relative_power",
        band="alpha",
        spatial_level="roi",
        condition="CE",
        output_path=(figures_dir/"alpha_relative_roi_ce_distribution.png"),
    )

    plot_condition_differences(
        paired,
        measure="relative_power",
        band="alpha",
        spatial_level="roi",
        output_path=(figures_dir/"alpha_relative_roi_ce_minus_oe.png"),
    )

    correlations = subject_correlation_matrix(
        subject_summary,
        condition="CE",
        spatial_level="roi",
    )

    correlations.to_csv(tables_dir/"roi_correlations_ce.csv")

    print("  QC 4/5: heatmap delle correlazioni...", flush=True)
    plot_correlation_heatmap(correlations, figures_dir / "roi_correlations_ce.png",)

    print("  QC 5/5: report salvati.", flush=True)
    print(f"Soggetti: {subjects.shape[0]}")
    print(f"Righe canoniche: {canonical.shape[0]:,}")
    print(f"Possibili outlier tecnici: {outliers.shape[0]}")


def main() -> None:
    canonical = pd.read_parquet(CANONICAL_DATA)
    subjects = pd.read_csv(SUBJECTS_DATA)
    run_quality_control(canonical, subjects)


if __name__ == "__main__":
    main()
