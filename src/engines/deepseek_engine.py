import os
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer

from .base import OcrEngine, OcrResult
from .utils import read_result_text_from_dir


class DeepSeekEngine:
    """
    Адаптер к DeepSeek-OCR модели.
    """
    name = "deepseek-ocr"

    def __init__(
        self,
        model_name: str,
        local_model_path: str,
        local_token_path: str,
        force_download: bool,
        device: str = "cuda:0",
        torch_dtype_load=torch.float16,
        torch_dtype_run=torch.bfloat16,
        trust_remote_code: bool = True,
        use_safetensors: bool = True,
    ):
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.local_token_path = local_token_path
        self.force_download = force_download
        self.device = device

        self.torch_dtype_load = torch_dtype_load
        self.torch_dtype_run = torch_dtype_run
        self.trust_remote_code = trust_remote_code
        self.use_safetensors = use_safetensors

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        model_path = Path(self.local_model_path)
        token_path = Path(self.local_token_path)

        if self.force_download or not model_path.exists():
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                use_safetensors=self.use_safetensors,
                device_map={"": 0} if self.device.startswith("cuda") else None,
                torch_dtype=self.torch_dtype_load,
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(model_path))
        else:
            self.model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=self.trust_remote_code,
                use_safetensors=self.use_safetensors,
                device_map={"": 0} if self.device.startswith("cuda") else None,
                torch_dtype=self.torch_dtype_load,
            )

        self.model = self.model.eval()
        if self.device.startswith("cuda"):
            self.model = self.model.cuda()
        self.model = self.model.to(self.torch_dtype_run)

        if self.force_download or not token_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(token_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(token_path), trust_remote_code=True)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, image_path: Path, prompt: str) -> OcrResult:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        start = perf_counter()
        with tempfile.TemporaryDirectory(prefix="deepseek_") as temp_output_dir:
            self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                base_size=1024,
                image_size=512,
                crop_mode=True,
                test_compress=True,
                save_results=True,
                output_path=temp_output_dir,
            )
            text = read_result_text_from_dir(temp_output_dir) or ""

        dur = perf_counter() - start
        return OcrResult(text=text, duration_s=dur, chars=len(text))