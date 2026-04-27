import base64
import json
import pathlib
import re
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from keras import Model
from keras import models
from keras import layers


ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
MODEL_PATH = ROOT / "htr_ctc_words_multilang_best_v4.keras"
PATCHED_MODEL_PATH = BACKEND_DIR / "runtime_model.keras"
CHARSET_PATH = BACKEND_DIR / "charset.json"
HOST = "0.0.0.0"
PORT = 8765
INPUT_HEIGHT = 32
INPUT_WIDTH = 256
REQUEST_LIMIT_BYTES = 12 * 1024 * 1024
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
MIN_COMPONENT_AREA = 40
DEBUG_LOGS = True
MIN_WORD_WIDTH = 15
MIN_WORD_HEIGHT = 10
DEBUG_CROPS_DIR = BACKEND_DIR / "debug_crops"

MODEL: Model | None = None
IDX_TO_CHAR: dict[int, str] = {}
BLANK_IDX = 0
STARTUP_ERROR: str | None = None


def ctc_loss(args: list[tf.Tensor]) -> tf.Tensor:
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

    if source.is_dir():
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for path in sorted(source.rglob("*")):
                if path.is_dir():
                    continue
                relname = path.relative_to(source).as_posix()
                data = path.read_bytes()
                if relname == "config.json":
                    config = json.loads(data.decode("utf-8"))
                    scrub(config)
                    data = json.dumps(config).encode("utf-8")
                zout.writestr(relname, data)
        return

    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "config.json":
                config = json.loads(data.decode("utf-8"))
                scrub(config)
                data = json.dumps(config).encode("utf-8")
            zout.writestr(info, data)


def load_charset(path: pathlib.Path) -> tuple[dict[int, str], int]:
    if not path.exists():
        raise FileNotFoundError(
            "Missing backend/charset.json. Add the exact training charset as a JSON array of single-character strings."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not all(isinstance(item, str) and len(item) == 1 for item in data):
        raise ValueError("backend/charset.json must be a JSON array of single-character strings.")

    idx_to_char = {index: char for index, char in enumerate(data)}
    return idx_to_char, len(data)


def load_model() -> Model:
    original_unlink = pathlib.Path.unlink

    def safe_unlink(self: pathlib.Path, *args: Any, **kwargs: Any) -> None:
        try:
            original_unlink(self, *args, **kwargs)
        except PermissionError:
            # Keras sometimes leaves the extracted H5 file locked on Windows.
            return None

    pathlib.Path.unlink = safe_unlink
    try:
        model = models.load_model(
            str(PATCHED_MODEL_PATH),
            custom_objects={"ctc_loss": ctc_loss, "PositionalEncoding": PositionalEncoding},
            compile=False,
            safe_mode=False,
        )
    finally:
        pathlib.Path.unlink = original_unlink

    return Model(inputs=model.inputs[0], outputs=model.get_layer("y_pred").output)


def preprocess_word(img: np.ndarray) -> tuple[np.ndarray, int]:
    if img is None or img.size == 0:
        raise ValueError("Invalid cropped image size.")

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


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "blank"


def clear_debug_crops() -> None:
    DEBUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in DEBUG_CROPS_DIR.iterdir():
        if file_path.is_file():
            try:
                file_path.unlink()
            except OSError:
                pass


def save_debug_crop(index: int, segment: np.ndarray, text: str) -> None:
    DEBUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    label = sanitize_filename(text)
    target = DEBUG_CROPS_DIR / f"segment_{index:03d}_{label}.png"
    cv2.imwrite(str(target), segment)


def segment_words(img: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
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
    words: list[tuple[int, int, int, int, np.ndarray]] = []

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

    ordered_words: list[tuple[int, int, int, int, np.ndarray]] = []
    line_thresh = max(
        10,
        int(np.median([h for _, _, _, h, _ in words]) * 0.6),
    ) if words else 10
    lines: list[dict[str, Any]] = []

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

    crops = [crop for _, _, _, _, crop in ordered_words if crop.size]
    boxes = [(x, y, w, h) for x, y, w, h, crop in ordered_words if crop.size]
    if not crops:
        height, width = gray.shape
        return [gray], [(0, 0, width, height)]

    return crops, boxes


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unsupported or corrupt image.")
    return img


def decode_prediction(prediction: np.ndarray, input_length: int | None = None) -> str:
    if input_length is None:
        input_length = prediction.shape[1]

    sliced = prediction[:, :input_length, :]
    logits = tf.math.log(tf.convert_to_tensor(sliced, dtype=tf.float32) + 1e-8)
    sequence_length = tf.constant([input_length], dtype=tf.int32)
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=sequence_length,
        blank_index=BLANK_IDX,
    )
    dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    tokens = dense[0] if len(dense) else []
    chars: list[str] = []
    for token in tokens:
        if token == -1 or token == BLANK_IDX:
            continue
        chars.append(IDX_TO_CHAR.get(int(token), ""))
    return "".join(chars).strip()




def run_inference(image_b64: str, mode: str = "photo") -> dict[str, Any]:
    if MODEL is None:
        raise RuntimeError(STARTUP_ERROR or "Model is not loaded.")

    image_bytes = base64.b64decode(image_b64, validate=True)
    gray = decode_image_bytes(image_bytes)
    if DEBUG_LOGS:
        print(f"[recognize] mode={mode} image_shape={gray.shape}")

    if mode == "photo":
        segments, boxes = segment_words(gray)
        clear_debug_crops()
        if DEBUG_LOGS:
            print(f"[recognize] segments_found={len(segments)}")
        parts: list[str] = []
        for index, segment in enumerate(segments):
            if segment.shape[1] < MIN_WORD_WIDTH or segment.shape[0] < MIN_WORD_HEIGHT:
                if DEBUG_LOGS:
                    box = boxes[index] if index < len(boxes) else None
                    print(f"[recognize] segment={index} box={box} skipped=too_small")
                continue
            batch, time_steps = preprocess_word(segment)
            predictions = MODEL.predict(batch, verbose=0)
            text = decode_prediction(predictions, input_length=time_steps)
            save_debug_crop(index, segment, text)
            if DEBUG_LOGS:
                box = boxes[index] if index < len(boxes) else None
                print(
                    f"[recognize] segment={index} box={box} crop_shape={segment.shape} "
                    f"time_steps={time_steps} text={text!r}"
                )
            if text and not (len(text) == 1 and not text.isalnum()):
                parts.append(text)
        best_text = " ".join(parts).strip()
    else:
        batch, time_steps = preprocess_word(gray)
        predictions = MODEL.predict(batch, verbose=0)
        best_text = decode_prediction(predictions, input_length=time_steps)
        if DEBUG_LOGS:
            print(f"[recognize] canvas_text={best_text!r} time_steps={time_steps}")

    if DEBUG_LOGS:
        print(f"[recognize] final_text={best_text!r}")

    return {
        "id": str(int(tf.timestamp().numpy() * 1000)),
        "text": best_text,
        "latency_ms": 0,
        "model": "local-htr-ctc",
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "InkVoiceHTR/1.0"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "Not found"})
            return

        self._send(
            200,
            {
                "ok": STARTUP_ERROR is None,
                "model_path": str(MODEL_PATH.name),
                "charset_path": str(CHARSET_PATH.name),
                "blank_index": BLANK_IDX,
                "charset_size": len(IDX_TO_CHAR),
                "error": STARTUP_ERROR,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/recognize":
            self._send(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "Invalid content length"})
            return

        if length <= 0 or length > REQUEST_LIMIT_BYTES:
            self._send(413, {"error": "Payload too large or empty"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_base64 = payload["image_base64"]
            mode = str(payload.get("mode", "photo"))
            result = run_inference(image_base64, mode=mode)
        except KeyError:
            self._send(400, {"error": "Missing image_base64"})
            return
        except Exception as exc:
            self._send(500, {"error": str(exc)})
            return

        self._send(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))


def bootstrap() -> None:
    global MODEL, IDX_TO_CHAR, BLANK_IDX, STARTUP_ERROR

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        patch_model_bundle(MODEL_PATH, PATCHED_MODEL_PATH)
        IDX_TO_CHAR, BLANK_IDX = load_charset(CHARSET_PATH)
        MODEL = load_model()
        STARTUP_ERROR = None
    except Exception as exc:
        STARTUP_ERROR = str(exc)
        MODEL = None


def main() -> None:
    bootstrap()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"HTR backend listening on http://127.0.0.1:{PORT}")
    if STARTUP_ERROR:
        print(f"Startup warning: {STARTUP_ERROR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
