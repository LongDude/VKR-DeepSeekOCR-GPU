from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path


@dataclass(frozen=True)
class LocalConfig:
    root: Path
    raw_files_dir: Path
    results_dir: Path

    model_name: str
    local_model_path: str
    local_token_path: str

    force_download: bool
    parse_pdf: bool
    parse_png: bool
    parse_jpg: bool

    free_gpu: bool
    cycle_parsing: bool
    cycle_interval_s: int

    log_format: str
    cuda_visible_devices: str | None

# парсер конфигурации
def _yes(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"yes", "y", "1", "true", "on"}


def load_config(path: Path) -> tuple[ConfigParser, LocalConfig]:
    cfg = ConfigParser(interpolation=None)
    cfg.read(path, encoding="utf-8")

    # Project root
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "raw_files"
    results_dir = root / "processed_files"

    local = cfg["Local"]

    cuda_visible = local.get("CUDA_VISIBLE_DEVICES", fallback=None)

    lc = LocalConfig(
        root=root,
        raw_files_dir=raw_dir,
        results_dir=results_dir,
        model_name=local.get("MODEL_NAME"),
        local_model_path=local.get("LOCAL_MODEL_PATH"),
        local_token_path=local.get("LOCAL_TOKEN_PATH"),
        force_download=_yes(local.get("FORCE_DOWNLOAD", "no")),
        parse_pdf=_yes(local.get("PARSE_PDF", "yes")),
        parse_png=_yes(local.get("PARSE_PNG", "yes")),
        parse_jpg=_yes(local.get("PARSE_JPG", "yes")),
        free_gpu=_yes(local.get("FREE_GPU", "no")),
        cycle_parsing=_yes(local.get("CYCLE_PARSING", "yes")),
        cycle_interval_s=int(local.get("CYCLE_INTERVAL", "5")),
        log_format=local.get("LOG_FORMAT", "%Y-%m-%d.log"),
        cuda_visible_devices=cuda_visible,
    )

    return cfg, lc