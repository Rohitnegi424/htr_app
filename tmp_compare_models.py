import pathlib
import json
import zipfile
from typing import Any
import tensorflow as tf
import numpy as np
import cv2
from keras import Model
from keras import models
from keras import layers

ROOT = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
MODEL_PATH = ROOT / "htr_ctc_words_multilang_best_v4.keras"
PATCHED_MODEL_PATH = BACKEND_DIR / "runtime_model.keras"
CHARSET_PATH = BACKEND_DIR / "charset.json"
IMAGE_PATH = pathlib.Path(r"D:\project\test\a1.png")
INPUT_HEIGHT = 32
INPUT_WIDTH = 256


def ctc_loss(args):
    return args[0]


class PositionalEncoding(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = tf.pow(10000.0, -(2.0 * tf.floor(i / 2.0)) / tf.cast(d_model, tf.float32))
        angles = position * angle_rates
        even_mask = tf.cast(tf.math.floormod(tf.range(d_model), 2) == 0, tf.float32)[tf.newaxis, :]
        odd_mask = 1.0 - even_mask
        pos_encoding = tf.sin(angles) * even_mask + tf.cos(angles) * odd_mask
        return inputs + pos_encoding[tf.newaxis, :, :]


def patch_model_bundle(source: pathlib.Path, destination: pathlib.Path) -> None:
    def scrub(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("class_name") == "RandomContrast" and isinstance(obj.get("config"), dict):
                obj["config"].pop("value_range", None)
            if obj.get("class_name") == "Dense" and isinstance(obj.get("config"), dict):
                obj["config"].pop("quantization_config", None)
            for value in obj.values():
                scrub(value)
        elif isinstance(obj, list):
            for item in obj:
                scrub(item)

    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "config.json":
                config = json.loads(data.decode("utf-8"))
                scrub(config)
                data = json.dumps(config).encode("utf-8")
            zout.writestr(info, data)


def load_charset(path: pathlib.Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    idx_to_char = {i: c for i, c in enumerate(data)}
    return idx_to_char, len(data)


def safe_load_model(path: pathlib.Path):
    original_unlink = pathlib.Path.unlink

    def safe_unlink(self, *args, **kwargs):
        try:
            return original_unlink(self, *args, **kwargs)
        except PermissionError:
            return None

    pathlib.Path.unlink = safe_unlink
    try:
        m = models.load_model(
            str(path),
            custom_objects={"ctc_loss": ctc_loss, "PositionalEncoding": PositionalEncoding},
            compile=False,
            safe_mode=False,
        )
    finally:
        pathlib.Path.unlink = original_unlink
    return Model(inputs=m.inputs[0], outputs=m.get_layer('y_pred').output)


def preprocess_word(img: np.ndarray):
    if img is None or img.size == 0:
        raise ValueError('Invalid cropped image size.')
    img = img.astype(np.float32) / 255.0
    h, w = img.shape
    scale = INPUT_HEIGHT / h
    new_w = max(1, min(int(w * scale), INPUT_WIDTH))
    resized = cv2.resize(img, (new_w, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    padded[:, :new_w] = resized
    padded = np.expand_dims(padded, axis=-1)
    padded = np.expand_dims(padded, axis=0)
    return padded, max(1, new_w // 4)


def decode_prediction(prediction: np.ndarray, input_length: int, idx_to_char, blank_idx: int):
    sliced = prediction[:, :input_length, :]
    logits = tf.math.log(tf.convert_to_tensor(sliced, dtype=tf.float32) + 1e-8)
    sequence_length = tf.constant([input_length], dtype=tf.int32)
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=sequence_length,
        blank_index=blank_idx,
    )
    dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    tokens = dense[0] if len(dense) else []
    chars = []
    for token in tokens:
        if token == -1 or token == blank_idx:
            continue
        chars.append(idx_to_char.get(int(token), ""))
    return "".join(chars).strip()


def segment_words(img: np.ndarray):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_inv = cv2.morphologyEx(
        bin_inv,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours if cv2.boundingRect(contour)[2] >= 3]
    med_h = int(np.median(heights)) if heights else 20
    med_h = max(12, min(med_h, 80))

    kernel_x = max(12, int(med_h * 0.8))
    kernel_y = max(3, int(med_h * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_x, kernel_y))
    connected = cv2.dilate(bin_inv, kernel, iterations=1)

    word_contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []

    for contour in word_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(12, int(med_h * 0.6)) or h < max(10, int(med_h * 0.5)):
            continue

        pad_y = int(0.18 * h)
        pad_x = int(0.08 * h)
        y1 = max(0, y - pad_y)
        y2 = min(gray.shape[0], y + h + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(gray.shape[1], x + w + pad_x)
        crop = gray[y1:y2, x1:x2]
        ink = np.mean(bin_inv[y1:y2, x1:x2] > 0)
        if ink < 0.03:
            continue
        words.append((x1, y1, x2 - x1, y2 - y1, crop))

    if not words:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue
            pad_y = int(0.15 * h)
            pad_x = 2
            y1 = max(0, y - pad_y)
            y2 = min(gray.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(gray.shape[1], x + w + pad_x)
            crop = gray[y1:y2, x1:x2]
            words.append((x1, y1, x2 - x1, y2 - y1, crop))

    words.sort(key=lambda box: (box[1], box[0]))
    ordered_words = []
    line_thresh = max(10, int(np.median([h for _, _, _, h, _ in words]) * 0.6)) if words else 10
    lines = []

    for x, y, w, h, crop in words:
        cy = y + h // 2
        placed = False
        for line in lines:
            if abs(cy - line["cy"]) < line_thresh:
                line["items"].append((x, y, w, h, crop))
                line["cy"] = int(np.mean([item[1] + item[3] // 2 for item in line["items"]]))
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "items": [(x, y, w, h, crop)]})

    lines.sort(key=lambda line: line["cy"])
    for line in lines:
        line["items"].sort(key=lambda box: box[0])
        ordered_words.extend(line["items"])

    return [crop for _, _, _, _, crop in ordered_words if crop.size]


def run_model(model, img, idx_to_char, blank_idx):
    segments = segment_words(img)
    texts = []
    for seg in segments:
        if seg.shape[1] < 15 or seg.shape[0] < 10:
            continue
        batch, time_steps = preprocess_word(seg)
        preds = model(batch, training=False).numpy()
        texts.append(decode_prediction(preds, time_steps, idx_to_char, blank_idx))
    return " ".join(texts), len(segments)


if __name__ == '__main__':
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not PATCHED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Runtime model not found: {PATCHED_MODEL_PATH}")

    idx_to_char, blank_idx = load_charset(CHARSET_PATH)

    temp_original_patched = ROOT / "tmp_original_patched.keras"
    patch_model_bundle(MODEL_PATH, temp_original_patched)

    original = safe_load_model(temp_original_patched)
    patched = safe_load_model(PATCHED_MODEL_PATH)
    img = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {IMAGE_PATH}")

    original_text, original_segments = run_model(original, img, idx_to_char, blank_idx)
    patched_text, patched_segments = run_model(patched, img, idx_to_char, blank_idx)

    print("ORIGINAL MODEL (patched temp):")
    print(f"  source_path: {MODEL_PATH}")
    print(f"  segments: {original_segments}")
    print(f"  text: {original_text}\n")

    print("RUNTIME MODEL:")
    print(f"  source_path: {PATCHED_MODEL_PATH}")
    print(f"  segments: {patched_segments}")
    print(f"  text: {patched_text}\n")

    if original_text == patched_text:
        print("RESULT: texts are identical")
    else:
        print("RESULT: texts differ")
