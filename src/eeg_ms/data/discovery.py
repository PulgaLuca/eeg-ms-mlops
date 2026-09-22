"""Individuazione deterministica dei file del dataset raw."""

from pathlib import Path

from eeg_ms.config import DATA_RAW, FAMILY_DIRECTORIES, GROUPS
from eeg_ms.data.parsing import FileMetadata, parse_filename


def discover_feature_files(
    raw_dir: Path = DATA_RAW,
) -> list[FileMetadata]:
    """Trova e interpreta tutti i file di feature attesi."""

    # "records" conserva il risultato normalizzato di ogni file MATLAB rilevato,
    # così il resto del pipeline può lavorare con i metadati coerenti e standardizzati.
    records: list[FileMetadata] = []

    # Il dataset raw è strutturato per famiglia (es. PSD, Complexity) (dunque estendibile
    # per ogni tipo di feature nuova introdotta) e poi per gruppo (es. HC, MS). 
    # Questo loop attraversa proprio quella gerarchia per costruire
    # l'elenco completo di file da analizzare in modo deterministico
    for family, directory_name in FAMILY_DIRECTORIES.items():
        family_dir = raw_dir/directory_name

        # verifica preventiva che la root della famiglia esista, senza questo controllo
        # il dataset sarebbe incompleto e il successivo scan sarebbe ambiguo
        if not family_dir.is_dir():
            raise FileNotFoundError(f"Directory della famiglia {family!r} assente: {family_dir}")

        for group in GROUPS:
            group_dir = family_dir/group

            # anche i gruppi devono essere presenti per la struttura attesa
            if not group_dir.is_dir():
                raise FileNotFoundError(f"Directory del gruppo {group!r} assente: {group_dir}")

            for path in sorted(group_dir.glob("*.mat")):
                records.append(
                    parse_filename(
                        path=path,
                        family=family,
                        group=group,
                    )
                )

    if not records:
        raise FileNotFoundError(f"Nessun file MATLAB trovato sotto {raw_dir}")

    return records
