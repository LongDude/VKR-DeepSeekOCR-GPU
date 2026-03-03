from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
from contextlib import contextmanager


@dataclass(frozen=True)
class PageImage:
    page_no: int
    image_path: Path


class DocumentHandler(Protocol):
    def can_handle(self, mime: str) -> bool: ...

    @contextmanager
    def iter_pages(self, input_path: Path) -> Iterable[PageImage]:
        yield []