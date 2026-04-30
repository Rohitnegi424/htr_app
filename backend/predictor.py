import cv2
import numpy as np
import tensorflow as tf
import unicodedata
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMG_HEIGHT = 32
IMG_WIDTH = 256

with open(BASE_DIR / "charset.json", "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

idx_to_char = {v: k for k, v in char_to_idx.items()}
blank_idx = len(char_to_idx)


def preprocess_word(img):
    img = img.astype(np.float32) / 255.0
    h, w = img.shape

    scale = IMG_HEIGHT / h
    new_w = max(1, min(int(w * scale), IMG_WIDTH))

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    padded[:, :new_w] = img

    model_input = padded[np.newaxis, ..., np.newaxis]
    time_steps = max(1, new_w // 4)

    return model_input, time_steps


def decode(preds, input_len):
    preds = preds[:, :input_len, :]
    decoded = tf.keras.backend.ctc_decode(
        preds,
        input_length=np.array([input_len]),
        greedy=True
    )[0][0].numpy()[0]

    text = ""
    for idx in decoded:
        if idx == -1 or idx == blank_idx:
            continue
        text += idx_to_char.get(int(idx), "")

    return unicodedata.normalize("NFC", text)


def predict_word(model, img):
    inp, t = preprocess_word(img)
    preds = model.predict(inp, verbose=0)
    return decode(preds, t)