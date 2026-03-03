from .base import DocumentHandler
from .image_handler import ImageHandler
from .pdf_handler import PdfHandler


def default_handlers(parse_pdf: bool, parse_png: bool, parse_jpg: bool) -> list[DocumentHandler]:
    handlers: list[DocumentHandler] = []

    if parse_pdf:
        handlers.append(PdfHandler(dpi=150))

    image_mimes: set[str] = set()
    if parse_png:
        image_mimes.add("image/png")
    if parse_jpg:
        image_mimes.update({"image/jpg", "image/jpeg"})

    if image_mimes:
        handlers.append(ImageHandler(supported_mimes=image_mimes))

    return handlers