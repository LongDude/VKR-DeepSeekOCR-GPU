from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import filetype

from .state import StateStore, file_fingerprint

@dataclass(frozen=True)
class ScanItem:
    input_path: Path
    mime: str
    fingerprint: str


def scan_directory(raw_dir: Path, state: StateStore) -> Iterable[ScanItem]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    for p in raw_dir.iterdir():
        if not p.is_file():
            continue

        try:
            fp = file_fingerprint(p)
        except Exception:
            continue

        if state.is_processed(fp):
            continue

        kind = filetype.guess(p)
        if kind is None:
            continue

        yield ScanItem(input_path=p, mime=kind.mime, fingerprint=fp)