import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(results_dir: Path, log_format: str) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)

    log_name = datetime.now().strftime(log_format)
    log_path = results_dir / log_name

    logger = logging.getLogger("ocr_app")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # синглтон
    if not logger.handlers:
        fh = logging.FileHandler(filename=str(log_path), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger