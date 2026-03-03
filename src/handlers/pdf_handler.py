import io
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image

from .base import DocumentHandler, PageImage


class PdfHandler:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def can_handle(self, mime: str) -> bool:
        return mime == "application/pdf"

    @contextmanager
    def iter_pages(self, input_path: Path) -> Iterable[PageImage]:
        doc = fitz.open(str(input_path))
        temp_dir = Path(tempfile.mkdtemp(prefix="pdf_pages_"))
        pages: list[PageImage] = []
        try:
            for i in range(len(doc)):
                page = doc.load_page(i)
                mat = fitz.Matrix(self.dpi / 36, self.dpi / 36)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("ppm")))
                out_path = temp_dir / f"page_{i+1}.png"
                img.save(out_path)
                pages.append(PageImage(page_no=i + 1, image_path=out_path))
            yield pages
        finally:
            doc.close()
            shutil.rmtree(temp_dir, ignore_errors=True)