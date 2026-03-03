from pathlib import Path


def read_result_text_from_dir(result_dir: str | Path) -> str | None:
    path = Path(result_dir)
    if path.exists():
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in (".mmd", ".txt", ".md"):
                text = child.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text
    return None