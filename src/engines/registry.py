from configparser import ConfigParser

from .base import OcrEngine
from .deepseek_engine import DeepSeekEngine


def build_engines(cfg: ConfigParser) -> list[OcrEngine]:
    """
    1) Совместимость со старым конфигом: используем [Local] и один DeepSeek.
    2) Новый режим: [Engines] enabled=deepseek0,deepseek1 ... и секции [DeepSeek.deepseek0] и т.п.
    """

    if "Engines" not in cfg:
        local = cfg["Local"]
        device = local.get("DEVICE", fallback="cuda:0")
        return [
            DeepSeekEngine(
                model_name=local.get("MODEL_NAME"),
                local_model_path=local.get("LOCAL_MODEL_PATH"),
                local_token_path=local.get("LOCAL_TOKEN_PATH"),
                force_download=local.get("FORCE_DOWNLOAD", "no").strip().lower() == "yes",
                device=device,
            )
        ]

    enabled = [x.strip() for x in cfg["Engines"].get("enabled", "").split(",") if x.strip()]
    engines: list[OcrEngine] = []

    for engine_id in enabled:
        section = f"DeepSeek.{engine_id}"
        if section not in cfg:
            raise ValueError(f"Missing config section: [{section}]")

        s = cfg[section]
        engines.append(
            DeepSeekEngine(
                model_name=s.get("MODEL_NAME"),
                local_model_path=s.get("LOCAL_MODEL_PATH"),
                local_token_path=s.get("LOCAL_TOKEN_PATH"),
                force_download=s.get("FORCE_DOWNLOAD", "no").strip().lower() == "yes",
                device=s.get("DEVICE", fallback="cuda:0"),
            )
        )

    if not engines:
        raise ValueError("No engines configured (empty [Engines].enabled).")

    return engines