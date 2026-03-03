from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import hashlib


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class StateStore:
    db_path: Path

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS processed (
                    fingerprint TEXT PRIMARY KEY,
                    input_path TEXT NOT NULL,
                    processed_at TEXT NOT NULL
                )
                """
            )

    def is_processed(self, fingerprint: str) -> bool:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("SELECT 1 FROM processed WHERE fingerprint = ?", (fingerprint,))
            return cur.fetchone() is not None

    def mark_processed(self, fingerprint: str, input_path: str) -> None:
        from datetime import datetime
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO processed (fingerprint, input_path, processed_at) VALUES (?, ?, ?)",
                (fingerprint, input_path, datetime.utcnow().isoformat()),
            )