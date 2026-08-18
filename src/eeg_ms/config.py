"""
Paths and structural contracts about the dataset.
No logic here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"

CHANNELS_FILE = DATA_RAW / "chanlocs.mat"

CANONICAL_DATA = DATA_PROCESSED / "canonical_features.parquet"
SUBJECTS_DATA = DATA_PROCESSED / "subjects.csv"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"

QC_TABLES = REPORTS / "tables"
QC_FIGURES = REPORTS / "figures" / "quality_control"

CONFIGS = ROOT / "configs"
VALIDATION_CONFIG = CONFIGS / "validation.yaml"

SPLITS_DIR = DATA_PROCESSED / "splits"
OUTER_FOLDS_FILE = SPLITS_DIR / "outer_test_folds.csv"

FEATURE_CONFIGS = CONFIGS / "features"

SUBJECT_FEATURES_ROI = (
    DATA_PROCESSED / "subject_features_roi.parquet"
)

METRICS_DIR = ARTIFACTS / "metrics"
PREDICTIONS_DIR = ARTIFACTS / "predictions"

EVALUATION_TABLES = (
    REPORTS / "tables" / "model_evaluation"
)

EVALUATION_FIGURES = (
    REPORTS / "figures" / "model_evaluation"
)

FOLD_METRICS_FILE = (
    METRICS_DIR / "nested_cv_fold_metrics.csv"
)

OOF_PREDICTIONS_FILE = (
    PREDICTIONS_DIR / "nested_cv_predictions.csv"
)

SELECTED_FEATURES_FILE = (
    ARTIFACTS / "selected_features"
    / "nested_cv_selected_features.csv"
)

TARGET_MAP = {
    "hc": 0,
    "ms": 1,
}

FAMILY_DIRECTORIES = {
    "psd": "PSD",
    "complexity": "Complexity"
}

GROUPS = ("hc", "ms")
CONDITIONS = ("CE", "OE")
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

GROUP_VISIT = {
    "hc": "CTR",
    "ms": "T0"
}

N_CHANNELS = 27
N_ROIS = 6
N_WINDOWS = 18
