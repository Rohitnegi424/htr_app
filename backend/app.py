from pathlib import Path
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
from difflib import SequenceMatcher
from functools import lru_cache
import json
import re
import urllib.parse
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
from model import load_model
from predictor import predict_word
from segmentation import segment_words
from trocr_predictor import get_trocr_status, predict_trocr_image, trocr_model_for_language

BASE_DIR = Path(__file__).resolve().parent
HINDI_MODEL_PATH = BASE_DIR.parent / "hindi_final.keras"
app = Flask(__name__)
CORS(app)

# Load once
with open(BASE_DIR / "charset.json", "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

model = load_model(str(BASE_DIR / "model.weights.h5"), len(char_to_idx) + 1)

with open(BASE_DIR / "hindi_charset.json", "r", encoding="utf-8") as f:
    hindi_char_to_idx = json.load(f)

hindi_idx_to_char = {v: k for k, v in hindi_char_to_idx.items()}
HINDI_MODEL_LABEL = "Hindi HTR"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ENGLISH_LANGUAGE_CODES = {"en", "en-us", "en-gb", "en-au", "en-ca"}
OCR_CONFUSIONS = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "7": "t",
    "8": "b",
})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    selected_model = (request.form.get("model") or "custom").strip().lower()
    language = (request.form.get("language") or "en-US").strip()

    if selected_model == "trocr":
        try:
            text = predict_trocr_image(img, trocr_model_for_language(language))
            return jsonify({
                "text": text,
                "words": [text] if text else [],
                "model": "trocr",
                "modelLabel": "TrOCR",
            })
        except Exception as exc:
            return jsonify({"error": f"Could not run TrOCR: {exc}"}), 500

    if selected_model == "hindi":
        try:
            hindi_model = load_hindi_model()
            text = predict_hindi_image(
                img,
                hindi_model,
                hindi_idx_to_char,
                hindi_model.output_shape[-1] - 1,
                get_model_image_size(hindi_model),
            )

            if not text.strip():
                text = predict_trocr_image(img, trocr_model_for_language("hi-IN"))
                return jsonify({
                    "text": text,
                    "words": [text] if text else [],
                    "model": "hindi",
                    "modelLabel": f"{HINDI_MODEL_LABEL} (fallback)",
                })

            return jsonify({
                "text": text,
                "words": [text],
                "model": "hindi",
                "modelLabel": HINDI_MODEL_LABEL,
            })
        except Exception as exc:
            return jsonify({"error": f"Could not run Hindi HTR: {exc}"}), 500

    results = predict_segmented_words(img, model)

    final_text = " ".join(results)

    return jsonify({
        "text": final_text,
        "words": results,
        "model": "custom",
        "modelLabel": "Custom HTR",
    })


@lru_cache(maxsize=1)
def load_hindi_model():
    if not HINDI_MODEL_PATH.exists():
        raise FileNotFoundError(f"Hindi model not found: {HINDI_MODEL_PATH}")

    image_input = tf.keras.layers.Input(shape=(32, 256, 1), name="image")
    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        name="conv2d",
    )(image_input)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)
    x = tf.keras.layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        name="conv2d_1",
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)
    x = tf.keras.layers.Reshape((64, -1), name="reshape")(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True),
        name="bidirectional",
    )(x)
    y_pred = tf.keras.layers.Dense(
        101,
        activation="softmax",
        name="dense",
    )(x)

    hindi_model = tf.keras.Model(inputs=image_input, outputs=y_pred)
    hindi_model.load_weights(HINDI_MODEL_PATH)
    return hindi_model


def get_model_image_size(selected_model):
    input_shape = selected_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if not input_shape or len(input_shape) < 4:
        return None

    height, width = input_shape[1], input_shape[2]
    if not height or not width:
        return None

    return int(height), int(width)


def predict_hindi_word(selected_model, img, idx_to_char_map, blank_token, image_size):
    if image_size is None:
        return predict_word(selected_model, img, idx_to_char_map, blank_token)

    model_input = preprocess_hindi_image(img, image_size)
    preds = selected_model.predict(model_input, verbose=0)
    input_len = preds.shape[1]
    return decode_ctc_text(preds, input_len, idx_to_char_map, blank_token)


def preprocess_hindi_image(img, image_size):
    target_height, target_width = image_size

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = crop_to_ink(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    h, w = gray.shape
    scale = target_height / h
    new_w = max(1, min(int(w * scale), target_width))
    resized = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target_height, target_width), dtype=np.float32)
    padded[:, :new_w] = resized.astype(np.float32) / 255.0
    return padded[np.newaxis, ..., np.newaxis]


def crop_to_ink(gray):
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    coords = cv2.findNonZero(binary)
    if coords is None:
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    pad_y = max(2, int(h * 0.18))
    pad_x = max(4, int(h * 0.25))
    y1 = max(0, y - pad_y)
    y2 = min(gray.shape[0], y + h + pad_y)
    x1 = max(0, x - pad_x)
    x2 = min(gray.shape[1], x + w + pad_x)
    return gray[y1:y2, x1:x2]


def predict_hindi_image(img, selected_model, idx_to_char_map, blank_token, image_size):
    if image_size is None:
        image_size = (32, 256)

    model_input = preprocess_hindi_image(img, image_size)
    preds = selected_model.predict(model_input, verbose=0)
    return decode_ctc_text(preds, preds.shape[1], idx_to_char_map, blank_token)


def decode_ctc_text(preds, input_len, idx_to_char_map, blank_token):
    decoded = tf.keras.backend.ctc_decode(
        preds[:, :input_len, :],
        input_length=np.array([input_len]),
        greedy=True,
    )[0][0].numpy()[0]

    text = ""
    for idx in decoded:
        idx = int(idx)
        if idx == -1 or idx == blank_token:
            continue
        text += idx_to_char_map.get(idx, "")

    return text


def predict_segmented_words(
    img,
    selected_model,
    idx_to_char_map=None,
    blank_token=None,
    image_size=None,
):
    words = segment_words(img)
    results = []

    for (_, _, _, _, word_img) in words:
        if word_img.shape[1] < 15:
            continue

        if image_size is None:
            text = predict_word(selected_model, word_img, idx_to_char_map, blank_token)
        else:
            text = predict_hindi_word(
                selected_model,
                word_img,
                idx_to_char_map,
                blank_token,
                image_size,
            )
        if text:
            results.append(text)

    return results


@app.route("/tts", methods=["POST"])
def text_to_speech():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lang = (data.get("lang") or "en").strip().lower()

    if not text:
        return jsonify({"error": "No text"}), 400

    try:
        audio = BytesIO()
        gTTS(text=text[:4500], lang=lang).write_to_fp(audio)
        audio.seek(0)
        return send_file(audio, mimetype="audio/mpeg", download_name="speech.mp3")
    except ValueError:
        return jsonify({"error": f"Language '{lang}' is not supported for read aloud"}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not generate speech: {exc}"}), 500


@app.route("/trocr-status", methods=["GET"])
def trocr_status():
    return jsonify(get_trocr_status())


def _is_english(language):
    return (language or "en-US").strip().lower() in ENGLISH_LANGUAGE_CODES


@lru_cache(maxsize=1)
def load_english_words():
    words_path = BASE_DIR.parent / "words_new.txt"
    words = {}

    if not words_path.exists():
        return words

    with open(words_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9 or parts[1] == "err":
                continue

            word = parts[-1].lower()
            if not word.isalpha() or len(word) < 2:
                continue

            words[word] = words.get(word, 0) + 1

    return words


def preserve_case(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def edits_one(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [left + right[1:] for left, right in splits if right]
    transposes = [
        left + right[1] + right[0] + right[2:]
        for left, right in splits
        if len(right) > 1
    ]
    replaces = [
        left + letter + right[1:]
        for left, right in splits
        if right
        for letter in letters
    ]
    inserts = [
        left + letter + right
        for left, right in splits
        for letter in letters
    ]
    return set(deletes + transposes + replaces + inserts)


def best_local_replacement(token, dictionary):
    normalized = token.lower()
    if len(normalized) < 3 or normalized in dictionary:
        return token

    candidates = set()
    confusion_fixed = normalized.translate(OCR_CONFUSIONS)
    if confusion_fixed in dictionary:
        candidates.add(confusion_fixed)

    candidates.update(candidate for candidate in edits_one(normalized) if candidate in dictionary)

    if not candidates:
        return token

    replacement = max(
        candidates,
        key=lambda item: (
            SequenceMatcher(None, normalized, item).ratio(),
            dictionary.get(item, 0),
            -abs(len(item) - len(normalized)),
        ),
    )
    return preserve_case(token, replacement)


def local_autocorrect(text, language):
    if not _is_english(language):
        return text

    dictionary = load_english_words()
    if not dictionary:
        return text

    def replace(match):
        return best_local_replacement(match.group(0), dictionary)

    return WORD_RE.sub(replace, text)


def is_safe_languagetool_match(match):
    rule = match.get("rule") or {}
    issue_type = rule.get("issueType", "")
    category = (rule.get("category") or {}).get("id", "")

    # For OCR output, style/grammar suggestions are often too eager. Keep the
    # online pass focused on spelling-level fixes and let local correction
    # handle common OCR typos when LanguageTool is unavailable.
    return issue_type == "misspelling" or category == "TYPOS"


def apply_corrections(text, matches):
    corrected = text

    safe_matches = [match for match in matches if is_safe_languagetool_match(match)]

    for match in sorted(safe_matches, key=lambda item: item.get("offset", 0), reverse=True):
        replacements = match.get("replacements") or []
        if not replacements:
            continue

        offset = match.get("offset")
        length = match.get("length")

        if not isinstance(offset, int) or not isinstance(length, int):
            continue

        replacement = replacements[0].get("value", "")
        corrected = corrected[:offset] + replacement + corrected[offset + length:]

    return corrected


@app.route("/autocorrect", methods=["POST"])
def autocorrect():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    language = (data.get("language") or "en-US").strip()

    if not text:
        return jsonify({"error": "No text"}), 400

    form = urllib.parse.urlencode({
        "text": text,
        "language": language,
    }).encode("utf-8")

    corrected = text
    online_matches = 0
    online_available = False

    try:
        req = urllib.request.Request(
            "https://api.languagetool.org/v2/check",
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))

        online_available = True
        matches = result.get("matches", [])
        online_matches = len(matches)
        corrected = apply_corrections(text, matches)
    except Exception:
        # Public LanguageTool can fail because of network/rate limits. Still
        # return a deterministic local correction instead of failing the UI.
        corrected = text

    corrected = local_autocorrect(corrected, language)

    return jsonify({
        "corrected": corrected,
        "changed": corrected != text,
        "matches": online_matches,
        "online": online_available,
    })


if __name__ == "__main__":
    app.run(debug=True)
