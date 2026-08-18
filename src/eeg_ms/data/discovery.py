"""Individuazione deterministica dei file del dataset raw."""

from pathlib import Path

from eeg_ms.config import DATA_RAW, FAMILY_DIRECTORIES, GROUPS
from eeg_ms.data.parsing import FileMetadata, parse_filename


def discover_feature_files(
    raw_dir: Path = DATA_RAW,
) -> list[FileMetadata]:
    """Trova e interpreta tutti i file di feature attesi."""

    records: list[FileMetadata] = []

    for family, directory_name in FAMILY_DIRECTORIES.items():
        family_dir = raw_dir / directory_name

        if not family_dir.is_dir():
            raise FileNotFoundError(
                f"Directory della famiglia {family!r} assente: {family_dir}"
            )

        for group in GROUPS:
            group_dir = family_dir / group

            if not group_dir.is_dir():
                raise FileNotFoundError(
                    f"Directory del gruppo {group!r} assente: {group_dir}"
                )

            for path in sorted(group_dir.glob("*.mat")):
                records.append(
                    parse_filename(
                        path=path,
                        family=family,
                        group=group,
                    )
                )

    if not records:
        raise FileNotFoundError(
            f"Nessun file MATLAB trovato sotto {raw_dir}"
        )

    return records