from transformers import AutoModel, AutoTokenizer
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
from pathlib import Path

# force-set terminal encoding if possible
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Logging to stdout and file
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
model_name = 'deepseek-ai/DeepSeek-OCR'


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

def process_png(model, tokenizer, image_path, output_path:Path, prompt_template="<image>\n<|grounding|>Convert the document to markdown. "):
    start_time = perf_counter()
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Processing image %s", image_path)

    with tempfile.TemporaryDirectory(prefix="deepseek_") as temp_output_dir:
        inference_output = model.infer(
            tokenizer,
            prompt=prompt_template,
            image_file=image_path,
            base_size=1024,
            image_size=512,
            crop_mode=True,
            test_compress=True,
            save_results=True,
            output_path=temp_output_dir,
        )

        img_text = inference_output or read_result_text_from_dir(temp_output_dir) or ""

    if not img_text:
        logger.warning("No OCR text extracted for image")
    else:
        logger.debug("Image %s produced %d characters", image_path, len(img_text))

    total_duration = perf_counter() - start_time
    logger.info("Image completed in %.2fs", total_duration)

    md_path = output_path.with_suffix('.md')
    logger.info("Writing markdown summary to %s", md_path)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(img_text)


def process_pdf_document(model, tokenizer, pdf_path, output_path, prompt_template="<image>\n<|grounding|>Convert the document to markdown. "):
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

def save_results(results, output_path: Path):
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

    md_path = output_path.with_name(f"{output_path.name}.md")
    logger.info("Writing markdown summary to %s", md_path)
    with open(md_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"# Page {result['page_number']}\n\n")
            f.write(result['content'])
            f.write("\n\n---\n\n")

def load_model():
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    return model

# Основной цикл
if __name__ == "__main__":
    model = None
    import pathlib
    import filetype
    import hashlib

    hash_name = lambda s: hashlib.sha256(s.encode()).hexdigest()

    processed_filenames = set()
    processed_filenames.add(hash_name(".gitkeep"))

    ROOT = pathlib.Path(__file__).parent
    raw_files_dir = ROOT / 'raw_files'
    results_dir = ROOT / 'processed_files'
    for processed_file_path in results_dir.iterdir():
        base_name = processed_file_path.stem
        hashed_base_name = hash_name(base_name)
        processed_filenames.add(hashed_base_name)

    logger.info("Starting parse cycle")
    while True:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = load_model()
        print(f"Allocated after reserving: {torch.cuda.memory_allocated() / (2**30):.2f} Gb")

        logger.info("Checking directory")
        found_files = 0
        for raw_file_path in raw_files_dir.iterdir():
            base_name = raw_file_path.stem
            hashed_base_name = hash_name(base_name)
            if hashed_base_name in processed_filenames:
                continue
            
            logger.info("Processing '%s'", base_name)
            
            kind = filetype.guess(raw_file_path)
            if kind is None:
                logger.error("Couldnt guess file type: %s", raw_file_path)
                continue

            found_files += 1
            file_mime = kind.mime
            results = ''
            match file_mime:
                case "application/pdf":
                    result = process_pdf_document(model, tokenizer, str(raw_file_path), results_dir / (raw_file_path.stem))
                    logger.info("Обработано %d страниц", len(results))
                    for result in results:
                        logger.info("Страница %d: %d символов", result['page_number'], len(result['content']))
                case "image/png":
                    result = process_png(model, tokenizer, str(raw_file_path), results_dir / (raw_file_path.stem))
                case _:
                    logger.log(f"Unsuppotred mime type '{file_mime}' for '{base_name}'")
            processed_filenames.add(hashed_base_name)
            
        logger.info("Found %d new files", found_files)
        del model
        del tokenizer
        torch.cuda.empty_cache()
        sleep(0.001) # ensuring that memory cleaned
        print(f"Allocated after cleaning: {torch.cuda.memory_allocated() / (2**30):.2f} Gb")
        logger.info("Wait for 5m")
        sleep(300)