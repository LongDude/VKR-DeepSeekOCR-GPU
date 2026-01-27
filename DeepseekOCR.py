import importlib
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
import tempfile
from time import *
from pathlib import Path
from datetime import *
from sys import stdout
import sys
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    handlers=[
        logging.FileHandler(filename=f"{datetime.now().strftime('logs/deepseek_ocr_%d%m%Y_%H%M.log')}", mode="a"), 
        logging.StreamHandler(stream=stdout)
        ],
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("test record")


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# --- Deepseek ---

model_name = 'deepseek-ai/DeepSeek-OCR'
# model_name = "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

torch.backends.cudnn.benchmark = True
max_memory = {
    0: "7.5GiB",      # GPU 0
    "cpu": "32GiB", # CPU offload
}

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    max_memory=max_memory,
)


def pdf_to_images(pdf_path, dpi=150):
    """Конвертирует PDF в список изображений страниц"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi/36, dpi/36)  # Повышаем DPI для лучшего качества
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data))
        images.append(image)
    
    doc.close()
    return images


def read_result_text_from_dir(result_dir):
    """Попытка считать выведенный текст DeepSeek-а, если возвращаемое значение пустое."""
    path = Path(result_dir)
    result_file = path / "result.mmd"
    if result_file.exists():
        return result_file.read_text(encoding="utf-8", errors="ignore").strip()
    if path.exists():
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in (".mmd", ".txt", ".md"):
                text = child.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text
    return None


def process_pdf_document(pdf_path, output_path, logfile=None, prompt_template="<image>\n<|grounding|>Convert the document to markdown. "):
    """???????????? ???? PDF ???????? ? ????????? ?????????"""
    start_time = perf_counter()
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s to images", pdf_path)
    page_images = pdf_to_images(pdf_path)
    total_pages = len(page_images)
    logger.info("PDF split into %d page(s)", total_pages)

    all_results = []
    page_durations = []

    for i, image in enumerate(page_images):
        page_number = i + 1
        logger.info("Processing page %d/%d", page_number, total_pages)
        temp_image_path = f"temp_page_{page_number}.png"
        image.save(temp_image_path)

        prompt = f"{prompt_template} (Page {page_number})"
        page_start = perf_counter()

        with tempfile.TemporaryDirectory(prefix="deepseek_") as temp_output_dir:
            inference_output = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                base_size=1024,
                image_size=512,
                crop_mode=True,
                test_compress=True,
                save_results=True,
                output_path=temp_output_dir,
            )

            page_text = inference_output or read_result_text_from_dir(temp_output_dir) or ""

        if not page_text:
            logger.warning("No OCR text extracted for page %d", page_number)
        else:
            logger.debug("Page %d produced %d characters", page_number, len(page_text))

        duration = perf_counter() - page_start
        page_durations.append(duration)
        avg_duration = sum(page_durations) / len(page_durations)
        logger.info("Page %d completed in %.2fs (avg %.2fs)", page_number, duration, avg_duration)

        page_result = {
            "page_number": page_number,
            "content": page_text,
            "image_path": temp_image_path
        }
        all_results.append(page_result)
        os.remove(temp_image_path)

    total_duration = perf_counter() - start_time
    avg_total = (sum(page_durations) / len(page_durations)) if page_durations else 0
    logger.info(
        "Finished OCR of %s in %.2fs (avg %.2fs per page)",
        pdf_path,
        total_duration,
        avg_total
    )

    save_results(all_results, output_path)
    return all_results

def save_results(results, output_path):
    """Сохраняет результаты в разных форматах"""

    # logger.info("Writing text summary to %s", output_path)
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     for result in results:
    #         f.write(f"=== Page {result['page_number']} ===\n")
    #         f.write(result['content'])
    #         f.write("\n\n" + "="*50 + "\n\n")

    # json_path = output_path.replace('.txt', '.json')
    # logger.info("Writing JSON summary to %s", json_path)
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    md_path = output_path.replace('.txt', '.md')
    logger.info("Writing markdown summary to %s", md_path)
    with open(md_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"# Page {result['page_number']}\n\n")
            f.write(result['content'])
            f.write("\n\n---\n\n")


# Основной код
if __name__ == "__main__":
    raw_files_dir = './'

    pdf_file = 'C:\\VKR\\VKR-LocalMLModels\\24_2511.03951v1.pdf'
    output_path = 'C:\\VKR\\VKR-LocalMLModels\\24_2511.03951v1.txt'
    logfile = output_path.replace(".txt", ".log")


    # Обрабатываем весь PDF
    results = process_pdf_document(pdf_file, output_path, logfile)
    
    logger.info("Обработано %d страниц", len(results))
    torch.cuda.empty_cache()
    
    # Выводим краткую информацию
    for result in results:
        logger.info("Страница %d: %d символов", result['page_number'], len(result['content']))

    input("Enter для выхода...")