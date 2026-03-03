from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .base import DocumentHandler, PageImage


class ImageHandler:
    def __init__(self, supported_mimes: set[str]):
        self.supported_mimes = supported_mimes

    def can_handle(self, mime: str) -> bool:
        return mime in self.supported_mimes

    @contextmanager
    def iter_pages(self, input_path: Path) -> Iterable[PageImage]:
        yield [PageImage(page_no=1, image_path=input_path)]