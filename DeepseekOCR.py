from transformers import AutoModel, AutoTokenizer
import fitz  # PyMuPDF
import torch
from PIL import Image

import io
import os
import sys
import logging
import tempfile
import pathlib
import filetype
import hashlib

from time import *
from datetime import *
from configparser import ConfigParser

# force-set terminal encoding if possible
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = ConfigParser(interpolation=None)
config.read("ModelConfig.ini", encoding='utf-8')

# Logging to stdout and file
logging.basicConfig(
    handlers=[
        logging.FileHandler(filename=f"{datetime.now().strftime(config['Local']['LOG_FORMAT'])}", mode="a"), 
        logging.StreamHandler(stream=sys.stdout)
        ],
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent
RAW_FILES_DIR = ROOT / 'raw_files'
RESULTS_DIR = ROOT / 'processed_files'

def load_model_data() -> tuple[AutoModel, AutoTokenizer]:
    if config['Local']['FORCE_DOWNLOAD'] == "yes" or not os.path.exists(config['Local']["LOCAL_MODEL_PATH"]):
        logger.info("Downloading model")
        model = AutoModel.from_pretrained(
            config["Local"]["MODEL_NAME"],
            trust_remote_code=True,
            use_safetensors=True,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
        model.save_pretrained(config["Local"]["LOCAL_MODEL_PATH"])
        logger.info("Model saved to %s", config["Local"]["LOCAL_MODEL_PATH"])
    else:
        logger.info("Loading model from %s", config["Local"]["LOCAL_MODEL_PATH"])
        model = AutoModel.from_pretrained(
            config["Local"]["LOCAL_MODEL_PATH"],
            trust_remote_code=True,
            use_safetensors=True,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
    model = model.eval().cuda().to(torch.bfloat16)

    if config['Local']['FORCE_DOWNLOAD'] == "yes" or not os.path.exists(config['Local']["LOCAL_TOKEN_PATH"]):
        logger.info("Downloading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(config["Local"]["MODEL_NAME"], trust_remote_code=True)
        tokenizer.save_pretrained(config["Local"]["LOCAL_TOKEN_PATH"])
        logger.info("Tokenizer saved to %s", config["Local"]["LOCAL_TOKEN_PATH"])
    else:
        logger.info("Loading tokenizer from %s", config["Local"]["LOCAL_TOKEN_PATH"])
        tokenizer = AutoTokenizer.from_pretrained(config["Local"]["LOCAL_TOKEN_PATH"], trust_remote_code=True)

    logger.info(f"Allocated after reserving: {torch.cuda.memory_allocated() / (2**30):.2f} Gb")
    return (model,tokenizer)

def pdf_to_images(pdf_path, dpi=150):
    """Разбивает pdf на массив страниц-изображений"""
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
    path = pathlib.Path(result_dir)
    if path.exists():
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in (".mmd", ".txt", ".md"):
                text = child.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text
    return None

def infer_image(model, tokenizer, image_path: pathlib.Path, prompt_template="<image>\n<|grounding|>Convert the document to markdown."):
    stage_start_time = perf_counter()
    with tempfile.TemporaryDirectory(prefix="deepseek_") as temp_output_dir:
        model.infer(
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
        img_text = read_result_text_from_dir(temp_output_dir)
    stage_total_duration = perf_counter() - stage_start_time
        
    if not img_text:
        logger.warning("No OCR text extracted for image")

    return (stage_total_duration, len(img_text), img_text or "")

def process_image(model, tokenizer, image_path, output_path:pathlib.Path):
    """ Send single image to model """
    output_path_obj = pathlib.Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Processing image %s", image_path.name)
    duration, text_len, text = infer_image(model, tokenizer, image_path)
    logger.debug("Image %s produced %d characters in %.2fs", image_path.name, text_len, duration)


    # Save results to markdown
    md_path = output_path.with_name(f'{output_path.name}.md')
    logger.info("Writing markdown summary to %s", md_path)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(text)


def process_pdf_document(model, tokenizer, pdf_path, output_path, prompt_template="<image>\n<|grounding|>Convert the document to markdown. "):
    """Split PDF in multiple images and send them to model"""

    output_path_obj = pathlib.Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    md_path = output_path.with_name(f'{output_path.name}.md')
    output_file = open(md_path, "w", encoding='utf-8')
    logger.info("Markdown summary will be saved to %s", md_path)

    logger.info("Converting %s to multiple images", pdf_path)
    page_images = pdf_to_images(pdf_path)
    total_pages = len(page_images)
    logger.info("PDF split into %d page(s)", total_pages)

    total_timer_start = perf_counter()
    page_results = [] # Number / text_len / parse_time
    summary_time = 0
    with tempfile.TemporaryDirectory(prefix="document_pages_") as temp_images_dir:
        temp_image_basepath = pathlib.Path(temp_images_dir)
        for i, image in enumerate(page_images):
            page_number = i + 1
            logger.info("Processing page %d/%d", page_number, total_pages)
            temp_image_path = temp_image_basepath / f"page_{page_number}.png"
            image.save(temp_image_path)

            page_time, page_len, page_text = infer_image(
                model, tokenizer, 
                temp_image_path, 
                f"{prompt_template} (Page {page_number})"
            )

            page_results.append([page_number, page_len, page_time])
            summary_time += page_time
            curr_avg = summary_time / page_number

            logger.info("Page %d produced %d characters in %.2fs (avg %.2fs)", page_number, page_len, page_time, curr_avg)
            output_file.write(page_text)
            output_file.write("\n")

    output_file.close()
    total_time_end = perf_counter() - total_timer_start
    avg_total = summary_time / max(total_pages, 1)
    logger.info(
        "Finished OCR of %s in %.2fs (%.2fs real, avg %.2fs per page)",
        pdf_path,
        summary_time,
        total_time_end,
        avg_total
    )
    for page_data in page_results:
        logger.info("Страница %d: %d символов (%.2fs)", *page_data)


# Основной цикл
if __name__ == "__main__":
    hash_name = lambda s: hashlib.sha256(s.encode()).hexdigest()

    model, tokenizer = None, None
    processed_filenames = set()
    processed_filenames.add(hash_name(".gitkeep"))

    for processed_file_path in RESULTS_DIR.iterdir():
        base_name = processed_file_path.stem
        hashed_base_name = hash_name(base_name)
        processed_filenames.add(hashed_base_name)

    while True:
        logger.info("Parsing %s directory", RAW_FILES_DIR)
        if model is None or tokenizer is None:
            model, tokenizer = load_model_data()

        logger.info("Checking directory...")
        found_files = 0
        processed_files = 0
        for raw_file_path in RAW_FILES_DIR.iterdir():
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
            match file_mime:
                case "application/pdf":
                    if config['Local']['PARSE_PDF'] != 'yes':
                        logger.info("Поддержка pdf файлов отключена: %s", raw_file_path.name)
                        continue

                    process_pdf_document(model, tokenizer, raw_file_path, RESULTS_DIR / (raw_file_path.stem))
                
                case "image/png":
                    if config['Local']['PARSE_PNG'] != 'yes':
                        logger.info("Поддержка png файлов отключена: %s", raw_file_path.name)
                        continue
                    
                    process_image(model, tokenizer, raw_file_path, RESULTS_DIR / (raw_file_path.stem))
                case "image/jpg":
                    if config['Local']['PARSE_JPG'] != 'yes':
                        logger.info("Поддержка jpg файлов отключена: %s", raw_file_path.name)
                        continue
                    
                    process_image(model, tokenizer, raw_file_path, RESULTS_DIR / (raw_file_path.stem))
                case _:
                    logger.log(f"Unsuppotred mime type '{file_mime}' for '{base_name}'")

            processed_filenames.add(hashed_base_name)
        logger.info("Parsed %d new files", found_files)           

        # freeing allocated GPU recources
        if config['Local']['FREE_GPU'] == "yes":
            del model
            del tokenizer
            torch.cuda.empty_cache()
            sleep(0.001) # ensuring that memory cleaned
            print(f"Allocated after cleaning: {torch.cuda.memory_allocated() / (2**30):.2f} Gb")

        # Cycling parsing or single launch
        if config['Local']['CYCLE_PARSING'] == "yes":
            logger.info("Wait for %i", int(config['Local']['CYCLE_INTERVAL']))
            sleep(int(config['Local']['CYCLE_INTERVAL']))
        else:
            break

        
