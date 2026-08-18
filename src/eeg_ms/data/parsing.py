"""
Contents Metadata extraction and validation in filenames
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from eeg_ms.config import GROUP_VISIT

FeatureFamily = Literal["psd", "complexity"]
Group = Literal["hc", "ms"]
Condition = Literal["CE", "OE"]


@dataclass(frozen=True)
class FileMetadata:
    path: Path
    family: FeatureFamily
    group: Group
    subject_number: int
    subject_id: str
    visit: str
    condition: Condition


PATTERNS = {
    "psd": re.compile(
        r"^PSDrelative_ID_?(?P<subject>\d+)_"
        r"(?P<visit>CTR|T0)_(?P<condition>CE|OE)\.mat$"
    ),
    "complexity": re.compile(
        r"^SpectralEntropy_ID_?(?P<subject>\d+)_"
        r"(?P<visit>CTR|T0)_(?P<condition>CE|OE)\.mat$"
    ),
}


def parse_filename(path: Path, family: FeatureFamily, group: Group) -> FileMetadata:
    """Trasforma nome e posizione del file in metadati strutturati"""

    match = PATTERNS[family].fullmatch(path.name)

    if match is None:
        raise ValueError(f"Nome non valido per la famiglia {family!r}: {path.name}")

    subject_number = int(match.group("subject"))
    visit = match.group("visit")
    condition = match.group("condition")

    expected_visit = GROUP_VISIT[group]
    if visit != expected_visit:
        raise ValueError(
            f"Incoerenza gruppo/visita in {path.name}: "
            f"cartella={group!r}, visita={visit!r}, "
            f"attesa={expected_visit!r}"
        )

    return FileMetadata(
        path=path,
        family=family,
        group=group,
        subject_number=subject_number,
        subject_id=f"{group}_{subject_number:02d}",
        visit=visit,
        condition=condition,
    )