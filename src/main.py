import sys
import time
import os
from pathlib import Path
from multiprocessing import get_context, Queue

from ocr_app.config import load_config
from ocr_app.logging_setup import setup_logging
from ocr_app.engines.registry import build_engines
from ocr_app.handlers.registry import default_handlers
from ocr_app.pipeline.state import StateStore
from ocr_app.pipeline.scanner import scan_directory
from ocr_app.pipeline.orchestrator import Orchestrator, Job
from ocr_app.pipeline.worker import worker_main, WorkerSpec


DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

def main() -> int:
    # force encoding
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg_path = Path("ModelConfig.ini")
    cfg, local = load_config(cfg_path)

    if local.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = local.cuda_visible_devices

    logger = setup_logging(local.results_dir, local.log_format)
    logger.info("Starting OCR pipeline")
    logger.info("RAW dir: %s", local.raw_files_dir)
    logger.info("RESULTS dir: %s", local.results_dir)

    # Состояния для отслеживания файлов
    state = StateStore(db_path=local.results_dir / "state.db")
    state.init()

    # адаптеры и оркестратор
    handlers = default_handlers(local.parse_pdf, local.parse_png, local.parse_jpg)
    orchestrator = Orchestrator(handlers=handlers)

    # Модели обработки
    engines = build_engines(cfg)
    logger.info("Configured engines: %s", [e.name for e in engines])

    # Контекст рабочих
    ctx = get_context("spawn")
    in_queues: list[Queue] = []
    out_q: Queue = ctx.Queue()

    # Паттер Object Pool: создаем рабочие процессы для запуска параллельных моделей
    procs = []
    for idx, engine in enumerate(engines):
        in_q = ctx.Queue()
        in_queues.append(in_q)
        spec = WorkerSpec(worker_name=f"worker-{idx}", cuda_visible_devices=None)
        p = ctx.Process(
            target=worker_main,
            args=(spec, engine, orchestrator, in_q, out_q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    rr = 0  # round-robin
    try:
        while True:
            # Сканирование файлов
            found = 0
            dispatched = 0
            for item in scan_directory(local.raw_files_dir, state):
                found += 1
                job = Job(
                    input_path=item.input_path,
                    mime=item.mime,
                    fingerprint=item.fingerprint,
                    output_root=local.results_dir,
                    prompt=DEFAULT_PROMPT,
                )

                in_queues[rr % len(in_queues)].put(job)
                rr += 1
                dispatched += 1

            time.sleep(0.05)
            # Сбор результатов (non-blocking short poll)
            drained = 0
            while True:
                try:
                    status, fp, in_path, info, wname = out_q.get_nowait()
                except Exception:
                    break

                drained += 1
                if status == "ok":
                    state.mark_processed(fp, in_path)
                    logger.info("[%s] OK: %s -> %s", wname, in_path, info)
                elif status == "err":
                    # ошибочные файлы не помечаем как processed
                    logger.error("[%s] ERR: %s (%s)", wname, in_path, info)
                else:
                    logger.critical("[%s] FATAL: %s", wname, info)

            if found:
                logger.info("Scan: found=%d dispatched=%d results=%d", found, dispatched, drained)

            # Режим единственного прогона: ждем завершения работ и выходим
            if not local.cycle_parsing:
                time.sleep(0.2)
                while True:
                    try:
                        status, fp, in_path, info, wname = out_q.get_nowait()
                    except Exception:
                        break
                    if status == "ok":
                        state.mark_processed(fp, in_path)
                        logger.info("[%s] OK: %s -> %s", wname, in_path, info)
                    elif status == "err":
                        logger.error("[%s] ERR: %s (%s)", wname, in_path, info)
                    else:
                        logger.critical("[%s] FATAL: %s", wname, info)
                break

            logger.info("Sleep %ds", local.cycle_interval_s)
            time.sleep(local.cycle_interval_s)

    finally:
        # остановка запущенных рабочих
        for q in in_queues:
            q.put(None)
        for p in procs:
            p.join(timeout=5)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())