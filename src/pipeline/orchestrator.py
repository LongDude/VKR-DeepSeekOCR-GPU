from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ocr_app.engines.base import OcrEngine
from ocr_app.handlers.base import DocumentHandler


@dataclass(frozen=True)
class Job:
    input_path: Path
    mime: str
    fingerprint: str
    output_root: Path
    prompt: str


class Orchestrator:
    def __init__(self, handlers: Sequence[DocumentHandler]):
        self.handlers = list(handlers)

    def _select_handler(self, mime: str) -> DocumentHandler:
        for h in self.handlers:
            if h.can_handle(mime):
                return h
        raise ValueError(f"No handler for mime='{mime}'")

    def run_job(self, engine: OcrEngine, job: Job) -> Path:
        handler = self._select_handler(job.mime)

        out_dir = job.output_root / job.input_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"{job.input_path.stem}.{engine.name}.md"

        parts: list[str] = []
        with handler.iter_pages(job.input_path) as pages:
            for page in pages:
                prompt = f"{job.prompt} (Page {page.page_no})" if page.page_no > 1 else job.prompt
                res = engine.infer(page.image_path, prompt)
                parts.append(res.text)

        md_path.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
        return md_path