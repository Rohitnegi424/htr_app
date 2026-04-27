import cv2
import numpy as np
import tensorflow as tf
import unicodedata
import os
import re

# Keras compat: ignore older RandomContrast config arg "value_range"
_orig_rc_init = tf.keras.layers.RandomContrast.__init__
def _rc_init(self, factor, value_range=None, **kwargs):
    return _orig_rc_init(self, factor=factor, **kwargs)
tf.keras.layers.RandomContrast.__init__ = _rc_init

# CONFIG (V2 MODEL)
IMG_HEIGHT = 32
MAX_WIDTH = 256

MODEL_PATH = r"D:/project/models/htr_ctc_words_multilang_best_v2 (1).keras"
DATA_PATH  = r"D:/project/processed_data/data2.npz"
PARAGRAPH_IMAGE = r"D:\project\test\a1.png"

CROPPED_DIR = r"D:/project/cropped_words_v2_1"
os.makedirs(CROPPED_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found: {DATA_PATH}")

# CROPPED OUTPUT CLEANUP
def clear_cropped_dir():
    for name in os.listdir(CROPPED_DIR):
        p = os.path.join(CROPPED_DIR, name)
        if os.path.isfile(p):
            os.remove(p)

# DUMMY CTC LOSS (ONLY FOR MODEL LOADING)
def ctc_loss(args):
    return args[0]

# LOAD CHARSET (MUST MATCH TRAINING)
data = np.load(DATA_PATH, allow_pickle=True)
texts = data["labels"]

charset = sorted(set("".join(texts)))
idx_to_char = {i: c for i, c in enumerate(charset)}
blank_idx = len(charset)

print("Charset size:", len(charset))

# LOAD MODEL (INFERENCE ONLY)
full_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"ctc_loss": ctc_loss},
    compile=False
)

image_input = full_model.inputs[0]
y_pred = full_model.get_layer("y_pred").output

infer_model = tf.keras.models.Model(
    inputs=image_input,
    outputs=y_pred
)

print("Inference model loaded successfully.")

# WORD PREPROCESSING (MATCH TRAINING)
def preprocess_word(img):
    if img is None or img.size == 0:
        raise ValueError("Invalid word image")

    img = img.astype(np.float32) / 255.0
    h, w = img.shape

    scale = IMG_HEIGHT / h
    new_w = max(1, min(int(w * scale), MAX_WIDTH))

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    padded = np.zeros((IMG_HEIGHT, MAX_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img

    padded = padded[..., np.newaxis]
    padded = padded[np.newaxis, ...]

    time_steps = max(1, new_w // 4)
    return padded, time_steps

# SAFE FILENAME
def sanitize_filename(text):
    # Replace characters that are invalid on Windows/macOS/Linux
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "blank"

# CTC GREEDY DECODER
def ctc_decode(preds, input_len):
    preds = preds[:, :input_len, :]
    decoded, _ = tf.keras.backend.ctc_decode(
        preds,
        input_length=[input_len],
        greedy=True
    )
    return decoded[0].numpy()[0]

# WORD PREDICTION
def predict_word(word_img):
    img, time_steps = preprocess_word(word_img)
    preds = infer_model.predict(img, verbose=0)

    seq = ctc_decode(preds, time_steps)

    text = ""
    for idx in seq:
        if idx == -1 or idx == blank_idx:
            continue
        text += idx_to_char[idx]

    return unicodedata.normalize("NFC", text)

# LINE-AWARE WORD SEGMENTATION (STABLE)
def segment_words(paragraph_img):
    # Contour-based word segmentation tuned to training (word-level crops)
    gray = cv2.cvtColor(paragraph_img, cv2.COLOR_BGR2GRAY)

    # Binarize (text -> white)
    _, bin_inv = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small speckles
    bin_inv = cv2.morphologyEx(
        bin_inv,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )

    # Estimate typical character height to scale morphology
    contours, _ = cv2.findContours(
        bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[2] >= 3]
    med_h = int(np.median(heights)) if heights else 20
    med_h = max(12, min(med_h, 80))

    # Connect characters into words (horizontal emphasis)
    kx = max(12, int(med_h * 0.8))
    ky = max(3, int(med_h * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    connected = cv2.dilate(bin_inv, kernel, iterations=1)

    # Find word contours on connected mask
    word_contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    words = []
    for cnt in word_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter tiny boxes
        if w < max(12, int(med_h * 0.6)) or h < max(10, int(med_h * 0.5)):
            continue

        # Padding based on height
        pad_y = int(0.18 * h)
        pad_x = int(0.08 * h)

        y1 = max(0, y - pad_y)
        y2 = min(gray.shape[0], y + h + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(gray.shape[1], x + w + pad_x)

        crop = gray[y1:y2, x1:x2]
        pw = x2 - x1
        ph = y2 - y1

        # Skip low-ink crops
        ink = np.mean(bin_inv[y1:y2, x1:x2] > 0)
        if ink < 0.03:
            continue

        words.append((x1, y1, pw, ph, crop))

    # If nothing found, fall back to raw contours
    if not words:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            pad_y = int(0.15 * h)
            pad_x = 2
            y1 = max(0, y - pad_y)
            y2 = min(gray.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(gray.shape[1], x + w + pad_x)
            crop = gray[y1:y2, x1:x2]
            pw = x2 - x1
            ph = y2 - y1
            words.append((x1, y1, pw, ph, crop))

    # Sort top to bottom then left to right
    words.sort(key=lambda b: (b[1], b[0]))

    # Group into lines for proper ordering
    ordered_words = []
    line_thresh = max(10, int(np.median([h for _, _, _, h, _ in words]) * 0.6)) if words else 10
    lines = []

    for x, y, w, h, crop in words:
        cy = y + h // 2
        placed = False
        for line in lines:
            if abs(cy - line["cy"]) < line_thresh:
                line["items"].append((x, y, w, h, crop))
                line["cy"] = int(np.mean([b[1] + b[3] // 2 for b in line["items"]]))
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "items": [(x, y, w, h, crop)]})

    lines.sort(key=lambda l: l["cy"])
    for line in lines:
        line["items"].sort(key=lambda b: b[0])
        ordered_words.extend(line["items"])

    return ordered_words

# PARAGRAPH PREDICTION
def predict_paragraph(img_path):
    clear_cropped_dir()
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read paragraph image")

    words = segment_words(img)

    predictions = []

    for i, (_, _, _, _, word_img) in enumerate(words):
        if word_img.shape[1] < 15:
            continue

        pred = predict_word(word_img)
        predictions.append(pred)

        safe_pred = sanitize_filename(pred)
        cv2.imwrite(
            os.path.join(CROPPED_DIR, f"word_{i:03d}_{safe_pred}.png"),
            word_img
        )

    return " ".join(predictions)

# RUN
if __name__ == "__main__":
    print("\nPredicted paragraph:\n")
    print(predict_paragraph(PARAGRAPH_IMAGE))
    print("\nCropped words saved to:", CROPPED_DIR)
