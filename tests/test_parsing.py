from pathlib import Path

import pytest

from eeg_ms.data.parsing import parse_filename


def test_parse_hc_psd_filename():
    metadata = parse_filename(
        Path("PSDrelative_ID_01_CTR_CE.mat"),
        family="psd",
        group="hc",
    )

    assert metadata.subject_id == "hc_01"
    assert metadata.condition == "CE"
    assert metadata.visit == "CTR"


def test_parse_filename_without_underscore_after_id():
    metadata = parse_filename(
        Path("SpectralEntropy_ID01_CTR_OE.mat"),
        family="complexity",
        group="hc",
    )

    assert metadata.subject_id == "hc_01"


def test_reject_group_visit_inconsistency():
    with pytest.raises(ValueError, match="Incoerenza"):
        parse_filename(
            Path("SpectralEntropy_ID_01_T0_CE.mat"),
            family="complexity",
            group="hc",
        )


def test_reject_download_duplicate_suffix():
    with pytest.raises(ValueError, match="Nome non valido"):
        parse_filename(
            Path("SpectralEntropy_ID_01_T0_CE(1).mat"),
            family="complexity",
            group="ms",
        )
