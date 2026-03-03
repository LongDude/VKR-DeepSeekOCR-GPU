import os
import traceback
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Optional

from ocr_app.engines.base import OcrEngine
from ocr_app.pipeline.orchestrator import Orchestrator, Job


@dataclass(frozen=True)
class WorkerSpec:
    worker_name: str
    cuda_visible_devices: str | None


def worker_main(
    spec: WorkerSpec,
    engine: OcrEngine,
    orchestrator: Orchestrator,
    in_q: Queue,
    out_q: Queue,
) -> None:
    """
    Процесс-воркер:
    - фиксирует CUDA_VISIBLE_DEVICES
    - загружает модель
    - обрабатывает Job-ы своей очереди
    """
    try:
        if spec.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices

        engine.load()

        while True:
            job = in_q.get()
            if job is None:
                break

            try:
                md_path = orchestrator.run_job(engine, job)
                out_q.put(("ok", job.fingerprint, str(job.input_path), str(md_path), spec.worker_name))
            except Exception as e:
                out_q.put(("err", job.fingerprint, str(job.input_path), repr(e), spec.worker_name))
    except Exception:
        out_q.put(("fatal", "", "", traceback.format_exc(), spec.worker_name))
    finally:
        try:
            engine.unload()
        except Exception:
            pass