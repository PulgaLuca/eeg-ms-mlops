from pathlib import Path

import numpy as np

from eeg_ms.data.parsing import FileMetadata
from eeg_ms.data.validation import validate_feature_data


def test_wrong_entropy_shape_is_detected():
    metadata = FileMetadata(
        path=Path("example.mat"),
        family="complexity",
        group="hc",
        subject_number=1,
        subject_id="hc_01",
        visit="CTR",
        condition="CE",
    )

    data = {
        "spectral_all": np.zeros((26, 18)),
        "spectral_all_ROI": np.zeros((6, 18)),
    }

    issues = validate_feature_data(metadata, data)

    assert any(
        issue.code == "invalid_shape"
        for issue in issues
    )
