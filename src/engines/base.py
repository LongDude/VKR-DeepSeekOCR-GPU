from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class OcrResult:
    text: str
    duration_s: float
    chars: int


class OcrEngine(Protocol):
    name: str

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def infer(self, image_path: Path, prompt: str) -> OcrResult: ...