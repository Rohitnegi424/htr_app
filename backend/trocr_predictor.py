from functools import lru_cache
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image


TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
HINDI_TROCR_MODEL_NAME = "aayushpuri01/TrOCR-Devanagari"
MODEL_LABELS = {
    TROCR_MODEL_NAME: "TrOCR",
    HINDI_TROCR_MODEL_NAME: "TrOCR",
}
TROCR_WEIGHT_BYTES = 1_330_000_000
DEVANAGARI_LANGUAGE_PREFIXES = ("hi", "mr", "ne", "sa")
_STATUS_LOCK = threading.Lock()
_STATUS = {
    "stage": "idle",
    "message": "TrOCR has not started loading.",
    "percent": 0,
}


def _set_status(stage, message, percent=None):
    with _STATUS_LOCK:
        _STATUS["stage"] = stage
        _STATUS["message"] = message
        if percent is not None:
            _STATUS["percent"] = max(0, min(100, int(percent)))


def _cache_dir_for_model(model_name):
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_name.replace('/', '--')}"
    )


def _largest_cached_weight_bytes(model_name):
    cache_dir = _cache_dir_for_model(model_name)
    if not cache_dir.exists():
        return 0

    largest = 0
    for path in cache_dir.rglob("*"):
        if path.is_file():
            name = path.name.lower()
            if name.endswith(".incomplete") or name in {"pytorch_model.bin", "model.safetensors"}:
                largest = max(largest, path.stat().st_size)

    return largest


def _estimate_download_percent(model_name):
    return int((_largest_cached_weight_bytes(model_name) / TROCR_WEIGHT_BYTES) * 100)


def _watch_download(stop_event, model_name):
    label = MODEL_LABELS.get(model_name, "TrOCR")
    while not stop_event.is_set():
        percent = _estimate_download_percent(model_name)
        _set_status(
            "downloading",
            f"Downloading {label} model... about {percent}% cached.",
            percent,
        )
        stop_event.wait(1)


def get_trocr_status():
    with _STATUS_LOCK:
        return dict(_STATUS)


def trocr_model_for_language(language):
    normalized = (language or "").strip().lower()
    if normalized.startswith(DEVANAGARI_LANGUAGE_PREFIXES):
        return HINDI_TROCR_MODEL_NAME
    return TROCR_MODEL_NAME


def segment_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [
        cv2.boundingRect(contour)[3]
        for contour in contours
        if cv2.boundingRect(contour)[2] >= 3 and cv2.boundingRect(contour)[3] >= 3
    ]
    median_height = int(np.median(heights)) if heights else 18
    median_height = max(12, min(median_height, 60))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(60, median_height * 8), max(4, median_height // 3)),
    )
    connected = cv2.dilate(binary, kernel, iterations=1)

    line_contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    img_height, img_width = gray.shape
    min_area = img_width * median_height * 0.08

    for contour in line_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(30, median_height * 3) or h < max(8, median_height // 2):
            continue
        if w * h < min_area:
            continue

        pad_y = max(6, int(h * 0.45))
        pad_x = max(8, int(median_height * 0.8))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_width, x + w + pad_x)
        y2 = min(img_height, y + h + pad_y)

        ink = np.mean(binary[y1:y2, x1:x2] > 0)
        if ink < 0.005:
            continue

        lines.append((x1, y1, x2, y2))

    if not lines:
        return [image]

    lines.sort(key=lambda box: box[1])
    return [image[y1:y2, x1:x2] for x1, y1, x2, y2 in lines]


@lru_cache(maxsize=2)
def load_trocr(model_name=TROCR_MODEL_NAME):
    label = MODEL_LABELS.get(model_name, "TrOCR")
    _set_status("starting", f"Starting {label}...", _estimate_download_percent(model_name))
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_download,
        args=(stop_event, model_name),
        daemon=True,
    )
    watcher.start()

    try:
        import torch
        from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor
    except ImportError as exc:
        stop_event.set()
        watcher.join(timeout=1)
        _set_status("error", "TrOCR packages are missing.", _estimate_download_percent(model_name))
        raise RuntimeError(
            "TrOCR needs the backend packages 'torch' and 'transformers'. "
            "Install them with: pip install torch transformers"
        ) from exc

    try:
        try:
            processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
        except Exception:
            if model_name == TROCR_MODEL_NAME:
                raise

            image_processor = ViTImageProcessor.from_pretrained(TROCR_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            processor = TrOCRProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )

        _set_status("loading", f"Loading {label} weights into memory...", 95)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        align_processor_image_size(processor, model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _set_status("ready", f"{label} is loaded on {device}.", 100)
    except Exception as exc:
        _set_status("error", f"{label} failed to load: {exc}", _estimate_download_percent(model_name))
        raise
    finally:
        stop_event.set()
        watcher.join(timeout=1)

    return processor, model, device


def align_processor_image_size(processor, model):
    image_processor = getattr(processor, "image_processor", None)
    encoder_config = getattr(getattr(model, "config", None), "encoder", None)
    image_size = getattr(encoder_config, "image_size", None)

    if image_processor is None or image_size is None:
        return

    if isinstance(image_size, (tuple, list)):
        height, width = image_size[:2]
    else:
        height = width = image_size

    size = {"height": int(height), "width": int(width)}
    image_processor.size = size

    if hasattr(image_processor, "crop_size"):
        image_processor.crop_size = size


def predict_trocr_image(image, model_name=TROCR_MODEL_NAME):
    processor, model, device = load_trocr(model_name)

    line_images = []
    for line in segment_lines(image):
        rgb = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
        line_images.append(Image.fromarray(rgb))

    pixel_values = processor(images=line_images, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=96)
    lines = processor.batch_decode(generated_ids, skip_special_tokens=True)
    text = "\n".join(line.strip() for line in lines if line.strip())

    return text.strip()
