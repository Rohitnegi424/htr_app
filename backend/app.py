from pathlib import Path
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import json
import urllib.parse
import urllib.request
import cv2
import numpy as np
from model import load_model
from predictor import predict_word
from segmentation import segment_words

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app)

# Load once
with open(BASE_DIR / "charset.json", "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

model = load_model(str(BASE_DIR / "model.weights.h5"), len(char_to_idx) + 1)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    words = segment_words(img)

    results = []

    for (_, _, _, _, word_img) in words:
        if word_img.shape[1] < 15:
            continue

        text = predict_word(model, word_img)
        results.append(text)

    final_text = " ".join(results)

    return jsonify({
        "text": final_text,
        "words": results
    })


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


def apply_corrections(text, matches):
    corrected = text

    for match in sorted(matches, key=lambda item: item.get("offset", 0), reverse=True):
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

    try:
        req = urllib.request.Request(
            "https://api.languagetool.org/v2/check",
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))

        corrected = apply_corrections(text, result.get("matches", []))
        return jsonify({
            "corrected": corrected,
            "changed": corrected != text,
            "matches": len(result.get("matches", [])),
        })
    except Exception as exc:
        return jsonify({"error": f"Could not autocorrect text: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
