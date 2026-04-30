# ==============================================================
#  app.py  —  Handwritten Recognition System  (v2)
#  ─────────────────────────────────────────────────────────────
#  NEW FEATURES in this version:
#    🌐  Translate Button  – translate the predicted digit-word
#        into 10 built-in languages (no internet needed)
#    🔊  Speak Aloud Button – speak the result using the OS
#        built-in TTS engine (no extra install needed)
#    🔌  Predict function is LEFT EMPTY so you can plug in
#        your own model code inside  _run_my_model()
#
#  HOW TO RUN:
#      python app.py
#
#  DEPENDENCIES:
#      pip install pillow
#      (numpy / scikit-learn only if you use the stub model)
# ==============================================================

import math
import os
import platform
import subprocess
import threading
import unicodedata
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk   # pip install pillow
from tensorflow.keras import layers, models


# ──────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────────────────────
BG_DARK    = "#1e1e2e"
BG_PANEL   = "#2a2a3e"
BG_CANVAS  = "#12121e"
ACCENT     = "#7c3aed"
TEXT_LIGHT = "#e2e8f0"
TEXT_DIM   = "#94a3b8"
SUCCESS    = "#10b981"
WARNING    = "#f59e0b"
DANGER     = "#ef4444"
TEAL       = "#0f766e"
BLUE       = "#1d4ed8"
GOLD       = "#b45309"
SPEAK_CLR  = "#0e7490"   # cyan for Speak button

# Canvas size
CANVAS_W = 510
CANVAS_H = 400
DISPLAY_MAX = 380


# ──────────────────────────────────────────────────────────────
#  TRANSLATION TABLE
#  Built-in offline dictionary: digit 0-9 → word in 10 languages
#  Add or edit any language here — no internet required.
# ──────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "English":    ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"],
    "Hindi":      ["Shunya","Ek","Do","Teen","Chaar","Paanch","Chhah","Saat","Aath","Nau"],
    "Spanish":    ["Cero","Uno","Dos","Tres","Cuatro","Cinco","Seis","Siete","Ocho","Nueve"],
    "French":     ["Zéro","Un","Deux","Trois","Quatre","Cinq","Six","Sept","Huit","Neuf"],
    "German":     ["Null","Eins","Zwei","Drei","Vier","Fünf","Sechs","Sieben","Acht","Neun"],
    "Arabic":     ["Sifr","Wahid","Ithnan","Thalatha","Arba'a","Khamsa","Sitta","Sab'a","Thamaniya","Tis'a"],
    "Japanese":   ["Rei","Ichi","Ni","San","Shi","Go","Roku","Nana","Hachi","Kyū"],
    "Chinese":    ["Líng","Yī","Èr","Sān","Sì","Wǔ","Liù","Qī","Bā","Jiǔ"],
    "Russian":    ["Nol'","Odin","Dva","Tri","Chetyre","Pyat'","Shest'","Sem'","Vosem'","Devyat'"],
    "Portuguese": ["Zero","Um","Dois","Três","Quatro","Cinco","Seis","Sete","Oito","Nove"],
}

LANGUAGE_LIST = list(TRANSLATIONS.keys())


APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_HEIGHT = 32
IMG_WIDTH = 256
MODEL_OPTIONS = {
    "Checkpoint v4": os.path.join(
        APP_DIR,
        "models",
        "checkpoints",
        "htr_ctc_words_multilang_best_v4_unpacked",
        "model.weights.h5",
    ),
    "Final v4": os.path.join(
        APP_DIR,
        "models",
        "final",
        "htr_ctc_words_multilang_best_v4_unpacked",
        "model.weights.h5",
    ),
}
DATA_PATH = os.path.join(
    APP_DIR,
    "data",
    "processed",
    "processed_data",
    "data2.npz",
)

_MODEL_CACHE = {}
_IDX_TO_CHAR_CACHE = None
_BLANK_IDX_CACHE = None
PREDICTION_MODES = ["Auto", "Word", "Sentence"]


class PositionalEncoding(layers.Layer):
    def build(self, input_shape):
        d_model = int(input_shape[-1])
        seq_len = int(input_shape[-2])
        positions = np.arange(seq_len)[:, np.newaxis]
        half_dim = max(d_model // 2, 1)
        div_term = np.exp(np.arange(half_dim) * -(math.log(10000.0) / half_dim))
        angles = positions * div_term[np.newaxis, :]
        pos_encoding = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
        pos_encoding = pos_encoding[:, :d_model].astype(np.float32)
        self.positional_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        pos_encoding = tf.cast(self.positional_encoding, dtype=inputs.dtype)
        return inputs + pos_encoding

    def compute_output_shape(self, input_shape):
        return input_shape


def _conv_block(x, filters, pool=True, dropout=0.0):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if pool:
        x = layers.MaxPooling2D((2, 2))(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def _transformer_block(x, d_model, num_heads, ff_dim, dropout):
    attn_input = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(attn_input, attn_input)
    x = layers.Add()([x, attn_output])

    ffn_input = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = layers.Dense(ff_dim, activation="gelu")(ffn_input)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    x = layers.Add()([x, ffn])
    return x


def _build_v4_inference_model(num_output_tokens):
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    x = _conv_block(image_input, 64, pool=True, dropout=0.05)
    x = _conv_block(x, 128, pool=True, dropout=0.08)
    x = _conv_block(x, 256, pool=False, dropout=0.10)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((IMG_WIDTH // 4, 8 * 256))(x)
    x = layers.Dense(256)(x)
    x = PositionalEncoding(name="positional_encoding")(x)

    for _ in range(4):
        x = _transformer_block(x, d_model=256, num_heads=8, ff_dim=768, dropout=0.10)

    x = layers.Bidirectional(layers.LSTM(192, return_sequences=True, dropout=0.15))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(256, activation="gelu")(x)
    y_pred = layers.Dense(
        num_output_tokens,
        activation="softmax",
        dtype="float32",
        name="y_pred",
    )(x)

    return models.Model(inputs=image_input, outputs=y_pred)


def _load_charset():
    global _IDX_TO_CHAR_CACHE, _BLANK_IDX_CACHE
    if _IDX_TO_CHAR_CACHE is not None and _BLANK_IDX_CACHE is not None:
        return _IDX_TO_CHAR_CACHE, _BLANK_IDX_CACHE

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset charset file not found: {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=True)
    texts = data["labels"].tolist()
    charset = sorted(set("".join(texts)))

    _IDX_TO_CHAR_CACHE = {i: c for i, c in enumerate(charset)}
    _BLANK_IDX_CACHE = len(charset)
    return _IDX_TO_CHAR_CACHE, _BLANK_IDX_CACHE


def _get_inference_model(model_name):
    global _MODEL_CACHE
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    idx_to_char, blank_idx = _load_charset()
    num_output_tokens = blank_idx + 1

    model_weights_path = MODEL_OPTIONS.get(model_name)
    if model_weights_path is None:
        raise ValueError(f"Unknown model selection: {model_name}")

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(
            "Model weights not found. Expected extracted weights at "
            f"{model_weights_path}"
        )

    model = _build_v4_inference_model(num_output_tokens)
    model.load_weights(model_weights_path)
    _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _preprocess_for_v4(pil_image):
    gray = pil_image.convert("L")
    img_np = np.array(gray, dtype=np.float32) / 255.0

    h, w = img_np.shape
    if h <= 0 or w <= 0:
        raise ValueError("Image is empty or invalid.")

    scale = IMG_HEIGHT / h
    new_w = max(1, min(int(w * scale), IMG_WIDTH))

    resized = gray.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    resized_np = np.array(resized, dtype=np.float32) / 255.0

    padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    padded[:, :new_w] = resized_np

    model_input = padded[np.newaxis, ..., np.newaxis]
    time_steps = max(1, new_w // 4)
    return model_input, time_steps


def _decode_prediction(preds, time_steps, idx_to_char, blank_idx):
    preds = preds[:, :time_steps, :]
    decoded, _ = tf.keras.backend.ctc_decode(
        preds,
        input_length=np.array([time_steps]),
        greedy=True,
    )
    sequence = decoded[0].numpy()[0]

    chars = []
    for idx in sequence:
        idx = int(idx)
        if idx == -1 or idx == blank_idx:
            continue
        chars.append(idx_to_char[idx])
    return "".join(chars)


def _predict_word_image(pil_image, model_name):
    idx_to_char, blank_idx = _load_charset()
    model = _get_inference_model(model_name)
    model_input, time_steps = _preprocess_for_v4(pil_image)
    preds = model.predict(model_input, verbose=0)
    text = _decode_prediction(preds, time_steps, idx_to_char, blank_idx)
    return unicodedata.normalize("NFC", text)


def _segment_sentence_words(pil_image):
    rgb = pil_image.convert("RGB")
    frame = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, bin_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    bin_inv = cv2.morphologyEx(
        bin_inv,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[2] >= 3]
    med_h = int(np.median(heights)) if heights else 20
    med_h = max(12, min(med_h, 80))

    kx = max(12, int(med_h * 0.8))
    ky = max(3, int(med_h * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    connected = cv2.dilate(bin_inv, kernel, iterations=1)

    word_contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    words = []
    for cnt in word_contours:
        x, y, w, h = cv2.boundingRect(cnt)
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
                line["cy"] = int(np.mean([b[1] + b[3] // 2 for b in line["items"]]))
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "items": [(x, y, w, h, crop)]})

    lines.sort(key=lambda line: line["cy"])
    for line in lines:
        line["items"].sort(key=lambda box: box[0])
        ordered_words.extend(line["items"])

    return ordered_words


def _predict_sentence_image(pil_image, model_name):
    words = _segment_sentence_words(pil_image)
    if not words:
        return "[no text detected]"

    predictions = []
    for _, _, _, _, word_img in words:
        if word_img.shape[1] < 15:
            continue
        word_pil = Image.fromarray(word_img)
        pred = _predict_word_image(word_pil, model_name)
        if pred:
            predictions.append(pred)

    if not predictions:
        return "[no text detected]"
    return " ".join(predictions)


def _infer_prediction_mode(pil_image):
    width, height = pil_image.size
    if width >= 500 or height >= 120 or (width >= 280 and width >= height * 2.8):
        return "Sentence"
    return "Word"


# ==============================================================
#  ████████████████████████████████████████████████████████████
#  YOUR MODEL GOES HERE
#  ────────────────────────────────────────────────────────────
#  Replace the body of  _run_my_model()  with your own code.
#  The function receives a PIL Image and must return an integer
#  digit (0-9), or a string label.
#
#  Example sketch:
#      import your_model_module
#      result = your_model_module.predict(pil_image)
#      return result
#
#  The return value is displayed as:
#      "Predicted Handwriting: <your_return_value>"
# ==============================================================
def _run_my_model(pil_image: Image.Image, model_name: str):
    """
    ┌─────────────────────────────────────────────────────────┐
    │  PLUG YOUR MODEL IN HERE                                │
    │                                                         │
    │  Parameters:                                            │
    │    pil_image (PIL.Image.Image) – the uploaded /         │
    │               cropped image ready for inference         │
    │                                                         │
    │  Return:                                                │
    │    int or str – the predicted digit / label             │
    │    e.g.  return 5   or   return "Five"                  │
    └─────────────────────────────────────────────────────────┘
    """

    # ══════════════════════════════════════════
    #   YOUR CODE STARTS HERE
    # ══════════════════════════════════════════

    text = _predict_word_image(pil_image, model_name)
    if not text:
        return "[no text detected]"
    return text

    # ══════════════════════════════════════════
    #   YOUR CODE ENDS HERE
    # ══════════════════════════════════════════


# ==============================================================
#  MAIN APPLICATION CLASS
# ==============================================================
class HandwritingApp:
    """Full Tkinter GUI for the Handwritten Recognition System."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("✍  Handwritten Recognition System")
        self.root.geometry("960x700")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_DARK)

        # ── State variables ────────────────────────────────────
        self.original_image  = None   # PIL Image – full-res upload
        self.display_image   = None   # PIL Image – scaled for canvas
        self.working_image   = None   # PIL Image – what gets predicted
        self.tk_image        = None   # Tkinter-compatible photo object
        self.crop_rect_id    = None   # Canvas rubber-band rectangle id
        self.crop_start_x    = 0
        self.crop_start_y    = 0
        self.crop_mode       = False
        self.img_offset_x    = 0      # centering offset inside canvas
        self.img_offset_y    = 0

        # ── Prediction result (needed by Translate & Speak) ────
        self.last_predicted_digit = None   # int 0-9 (or None if no prediction yet)
        self.last_result_text     = ""     # full result string
        self.last_prediction_mode = "Word"
        self.last_model_name = "Checkpoint v4"

        # ── Build the interface ────────────────────────────────
        self._build_ui()


    # ==========================================================
    #  BUILD UI
    # ==========================================================
    def _build_ui(self):
        """Construct all widgets and lay them out."""

        # ── Title bar ──────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=ACCENT, height=54)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="✍  Handwritten Recognition System",
            bg=ACCENT, fg="white",
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left", padx=20, pady=14)

        tk.Label(
            title_bar,
            text="v2  •  Translate & Speak edition",
            bg=ACCENT, fg="#c4b5fd",
            font=("Segoe UI", 9),
        ).pack(side="right", padx=20, pady=18)

        # ── Body ───────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=14, pady=12)

        # ── LEFT: Image canvas ─────────────────────────────────
        left = tk.Frame(body, bg=BG_PANEL,
                        highlightthickness=2, highlightbackground=ACCENT)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Image Preview",
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 9, "bold")).pack(pady=(8, 2))

        self.canvas = tk.Canvas(left,
                                width=CANVAS_W, height=CANVAS_H,
                                bg=BG_CANVAS, cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0, 10))

        # Mouse events for cropping
        self.canvas.bind("<ButtonPress-1>",   self._crop_start)
        self.canvas.bind("<B1-Motion>",       self._crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._crop_end)

        self._canvas_placeholder()

        tk.Label(left, text="Predicted Text",
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 9, "bold")).pack(pady=(0, 4))

        result_wrap = tk.Frame(left, bg=BG_PANEL)
        result_wrap.pack(fill="x", padx=10, pady=(0, 10))

        result_box = tk.Frame(result_wrap, bg="#0f172a",
                              highlightthickness=1, highlightbackground=ACCENT)
        result_box.pack(fill="x")

        self.result_text = tk.Text(
            result_box,
            height=7,
            wrap="word",
            bg="#0f172a",
            fg=TEXT_DIM,
            insertbackground=TEXT_LIGHT,
            relief="flat",
            bd=0,
            padx=10,
            pady=10,
            font=("Segoe UI", 11),
        )
        self.result_text.pack(side="left", fill="both", expand=True)

        result_scroll = ttk.Scrollbar(result_box, orient="vertical", command=self.result_text.yview)
        result_scroll.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=result_scroll.set)
        self.result_text.insert("1.0", "No prediction yet")
        self.result_text.configure(state="disabled")

        # ── RIGHT: Control panel ───────────────────────────────
        right = tk.Frame(body, bg=BG_PANEL, width=270,
                         highlightthickness=2, highlightbackground=ACCENT)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        # ── Section: Image ─────────────────────────────────────
        self._sep(right)
        self._section(right, "📁  Image")
        self._btn(right, "⬆   Upload Image",   self._upload_image,  ACCENT)
        self._btn(right, "✂   Crop Image",     self._activate_crop, TEAL)

        # ── Section: Recognition ───────────────────────────────
        self._sep(right)
        self._section(right, "🔍  Recognition")

        mode_row = tk.Frame(right, bg=BG_PANEL)
        mode_row.pack(fill="x", padx=16, pady=(0, 4))

        tk.Label(mode_row, text="Mode:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        self.prediction_mode_var = tk.StringVar(value="Auto")
        mode_menu = ttk.Combobox(
            mode_row,
            textvariable=self.prediction_mode_var,
            values=PREDICTION_MODES,
            state="readonly",
            width=14,
            font=("Segoe UI", 9),
        )
        mode_menu.pack(side="right")

        model_row = tk.Frame(right, bg=BG_PANEL)
        model_row.pack(fill="x", padx=16, pady=(0, 4))

        tk.Label(model_row, text="Model:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        self.model_var = tk.StringVar(value="Checkpoint v4")
        model_menu = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly",
            width=14,
            font=("Segoe UI", 9),
        )
        model_menu.pack(side="right")

        self._btn(right, "▶   Predict Handwriting", self._predict, BLUE)

        # ── Section: NEW — Translate & Speak ───────────────────
        self._sep(right)
        self._section(right, "🌐  Language & Voice")

        # Language selector (dropdown)
        lang_row = tk.Frame(right, bg=BG_PANEL)
        lang_row.pack(fill="x", padx=16, pady=(0, 4))

        tk.Label(lang_row, text="Language:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        self.lang_var = tk.StringVar(value="English")
        lang_menu = ttk.Combobox(
            lang_row,
            textvariable=self.lang_var,
            values=LANGUAGE_LIST,
            state="readonly",
            width=14,
            font=("Segoe UI", 9),
        )
        lang_menu.pack(side="right")

        # Translate button
        self._btn(right, "🌐  Translate",  self._translate, GOLD)

        # Speak Aloud button
        self._btn(right, "🔊  Speak Aloud", self._speak,    SPEAK_CLR)

        # ── Section: Reset ─────────────────────────────────────
        self._sep(right)
        self._btn(right, "🗑   Clear / Reset", self._reset, DANGER)

        # ── Status label ───────────────────────────────────────
        self._sep(right)
        tk.Label(right, text="Status", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 8, "bold")).pack(pady=(6, 2))

        self.status_var = tk.StringVar(value="Upload an image to begin.")
        self.status_lbl = tk.Label(right, textvariable=self.status_var,
                                   bg=BG_PANEL, fg=TEXT_DIM,
                                   font=("Segoe UI", 8),
                                   wraplength=230, justify="center")
        self.status_lbl.pack(padx=8)

        # ── Translation card ───────────────────────────────────
        self._sep(right)
        tk.Label(right, text="Translation", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 8, "bold")).pack(pady=(6, 4))

        tcard = tk.Frame(right, bg="#0f172a",
                         highlightthickness=1, highlightbackground=GOLD)
        tcard.pack(padx=12, pady=2, fill="x")

        self.trans_var = tk.StringVar(value="—")
        self.trans_lbl = tk.Label(tcard, textvariable=self.trans_var,
                                  bg="#0f172a", fg=TEXT_DIM,
                                  font=("Segoe UI", 13, "bold"),
                                  wraplength=228, justify="center",
                                  pady=8, padx=6)
        self.trans_lbl.pack(fill="x")

        # ── Quick-tip ──────────────────────────────────────────
        self._sep(right)
        tk.Label(right,
                 text="① Upload  ② Pick mode  ③ Crop (opt)\n④ Predict  ⑤ Translate / Speak",
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 7), pady=6).pack()


    # ==========================================================
    #  UI HELPERS
    # ==========================================================
    def _sep(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=3)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=BG_PANEL, fg=TEXT_LIGHT,
                 font=("Segoe UI", 9, "bold"), pady=6
                 ).pack(anchor="w", padx=16)

    def _btn(self, parent, text, cmd, colour):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=colour, fg="white",
                      font=("Segoe UI", 9, "bold"),
                      relief="flat", cursor="hand2",
                      padx=10, pady=8,
                      activebackground=colour, activeforeground="white", bd=0)
        b.pack(fill="x", padx=16, pady=3)
        b.bind("<Enter>", lambda e: b.configure(bg=self._darken(colour)))
        b.bind("<Leave>", lambda e: b.configure(bg=colour))
        return b

    @staticmethod
    def _darken(hex_c, f=0.82):
        hex_c = hex_c.lstrip("#")
        r, g, b = (int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        return "#{:02x}{:02x}{:02x}".format(int(r*f), int(g*f), int(b*f))

    def _canvas_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(CANVAS_W//2, CANVAS_H//2,
                                text="Upload an image to get started",
                                fill=TEXT_DIM, font=("Segoe UI", 13),
                                tags="placeholder")

    def _set_status(self, text, colour=TEXT_DIM):
        self.status_var.set(text)
        self.status_lbl.configure(fg=colour)

    def _set_result(self, text, colour=TEXT_DIM):
        self.result_text.configure(state="normal", fg=colour)
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _set_trans(self, text, colour=TEXT_DIM):
        self.trans_var.set(text)
        self.trans_lbl.configure(fg=colour)


    # ==========================================================
    #  ACTION – Upload Image
    # ==========================================================
    def _upload_image(self):
        """Open file-chooser, load image, display on canvas."""
        path = filedialog.askopenfilename(
            title="Select a handwritten word or sentence image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not open image:\n{e}")
            return

        self.original_image = img
        self.working_image  = img
        self.display_image  = self._fit(img, DISPLAY_MAX)
        self._draw(self.display_image)

        self.last_predicted_digit = None
        self.last_result_text     = ""
        self.last_prediction_mode = "Word"
        self.last_model_name = self.model_var.get()

        w, h = img.size
        self._set_status(f"Image loaded  ({w}×{h} px)", TEXT_DIM)
        self._set_result("No prediction yet", TEXT_DIM)
        self._set_trans("—", TEXT_DIM)
        self.crop_mode = False


    # ==========================================================
    #  ACTION – Activate Crop Mode
    # ==========================================================
    def _activate_crop(self):
        """Turn on crop mode so the next mouse drag selects a region."""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        self.crop_mode = True
        self._set_status("Crop mode ON\nDrag on the image to select a region.", WARNING)


    # ==========================================================
    #  CROP – mouse events
    # ==========================================================
    def _crop_start(self, event):
        if not self.crop_mode:
            return
        self.crop_start_x, self.crop_start_y = event.x, event.y
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None

    def _crop_drag(self, event):
        if not self.crop_mode:
            return
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
        self.crop_rect_id = self.canvas.create_rectangle(
            self.crop_start_x, self.crop_start_y, event.x, event.y,
            outline="#facc15", width=2, dash=(6, 3)
        )

    def _crop_end(self, event):
        if not self.crop_mode:
            return
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
        self.crop_mode = False

        x1 = min(self.crop_start_x, event.x)
        y1 = min(self.crop_start_y, event.y)
        x2 = max(self.crop_start_x, event.x)
        y2 = max(self.crop_start_y, event.y)

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            self._set_status("Crop cancelled (selection too small).", WARNING)
            return

        # Map canvas coordinates → original image coordinates
        disp_w, disp_h = self.display_image.size
        orig_w, orig_h = self.original_image.size
        sx, sy = orig_w / disp_w, orig_h / disp_h

        rx1 = max(0, int((x1 - self.img_offset_x) * sx))
        ry1 = max(0, int((y1 - self.img_offset_y) * sy))
        rx2 = min(orig_w, int((x2 - self.img_offset_x) * sx))
        ry2 = min(orig_h, int((y2 - self.img_offset_y) * sy))

        if rx2 <= rx1 or ry2 <= ry1:
            self._set_status("Crop region outside image. Try again.", WARNING)
            return

        cropped = self.original_image.crop((rx1, ry1, rx2, ry2))
        self.working_image = cropped
        self.display_image = self._fit(cropped, DISPLAY_MAX)
        self._draw(self.display_image)

        cw, ch = cropped.size
        self._set_status(f"Cropped to {cw}×{ch} px — click Predict.", SUCCESS)


    # ==========================================================
    #  ACTION – Predict
    #  Calls _run_my_model() which you fill in above.
    # ==========================================================
    def _predict(self):
        """
        Calls _run_my_model() with the current working image.
        Displays whatever that function returns.
        After prediction, Translate and Speak buttons become usable.
        """
        if self.working_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        self._set_status("Running model…", TEXT_DIM)
        self._set_trans("—", TEXT_DIM)
        self.root.update_idletasks()   # refresh UI before blocking

        try:
            # ── Call your model ────────────────────────────────
            selected_mode = self.prediction_mode_var.get()
            selected_model = self.model_var.get()
            if selected_mode == "Auto":
                resolved_mode = _infer_prediction_mode(self.working_image)
            else:
                resolved_mode = selected_mode

            if resolved_mode == "Sentence":
                raw = _predict_sentence_image(self.working_image, selected_model)
            else:
                raw = _run_my_model(self.working_image, selected_model)
            # ──────────────────────────────────────────────────

            # Handle the case where the model slot is still empty
            if raw is None:
                self._set_result("⚠  Model not connected yet", WARNING)
                self._set_status(
                    "Open app.py and add your\nmodel code inside _run_my_model().",
                    WARNING,
                )
                self.last_predicted_digit = None
                return

            # Store digit (int) for translate / speak
            try:
                self.last_predicted_digit = int(raw)
            except (ValueError, TypeError):
                self.last_predicted_digit = None  # string label — translate won't apply

            # Build and show result text
            self.last_prediction_mode = resolved_mode
            self.last_model_name = selected_model
            self.last_result_text = f"Predicted Handwriting ({selected_model}): {raw}"
            self._set_result(self.last_result_text, SUCCESS)
            self._set_status(
                f"Prediction complete! ({resolved_mode} mode, {selected_model})\nNow Translate or Speak.",
                SUCCESS,
            )

        except Exception as exc:
            self._set_result(f"Error: {exc}", DANGER)
            self._set_status("Prediction failed.", DANGER)
            self.last_predicted_digit = None


    # ==========================================================
    #  NEW ACTION – Translate  🌐
    # ==========================================================
    def _translate(self):
        """
        Translate the predicted digit into the selected language
        using the built-in offline TRANSLATIONS dictionary.
        No internet connection required.
        """
        # Guard: need a prediction first
        if self.last_predicted_digit is None:
            messagebox.showwarning(
                "No Prediction",
                "Translate works for digit predictions only.\nPlease run Predict on a digit image first."
            )
            return

        digit = self.last_predicted_digit

        # Guard: digit must be 0-9
        if not (0 <= digit <= 9):
            self._set_trans(f"Cannot translate '{digit}'", WARNING)
            return

        # Look up the selected language in the translation table
        language = self.lang_var.get()
        word = TRANSLATIONS[language][digit]

        # Display in the translation card
        display = f"{word}  ({language})"
        self._set_trans(display, GOLD)
        self._set_status(f"Translated to {language}.", GOLD)


    # ==========================================================
    #  NEW ACTION – Speak Aloud  🔊
    # ==========================================================
    def _speak(self):
        """
        Speak the result string (or translation if one is shown)
        using the operating system's built-in TTS engine.

        • macOS  → uses `say` command (built-in)
        • Linux  → uses `espeak` or `festival` (install one of them)
        • Windows → uses PowerShell + SAPI.SpVoice (built-in)

        The speech runs in a background thread so the GUI never freezes.
        """
        # Decide what text to speak
        trans_text = self.trans_var.get()
        result_text = self.result_text.get("1.0", "end-1c").strip()

        if trans_text and trans_text not in ("—", ""):
            # Speak the translated word if a translation is shown
            speak_text = trans_text
        elif result_text and result_text not in ("No prediction yet", ""):
            # Otherwise speak the raw prediction result
            speak_text = result_text
        else:
            messagebox.showwarning(
                "Nothing to Speak",
                "Run Predict (and optionally Translate) first."
            )
            return

        self._set_status(f'Speaking: "{speak_text}"…', SPEAK_CLR)

        # Run TTS in a background thread so the window stays responsive
        threading.Thread(
            target=self._tts_speak,
            args=(speak_text,),
            daemon=True,         # thread dies when app closes
        ).start()


    def _tts_speak(self, text: str):
        """
        Internal — called in a background thread.
        Detects the OS and uses the appropriate TTS command.
        """
        system = platform.system()

        try:
            if system == "Darwin":
                # macOS: `say` is built-in, no install needed
                subprocess.run(["say", text], check=True)

            elif system == "Windows":
                # Windows: use PowerShell + built-in SAPI speech engine
                ps_cmd = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Speak('{text}');"
                )
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    check=True, creationflags=subprocess.CREATE_NO_WINDOW
                )

            else:
                # Linux: try espeak, fallback to festival
                if subprocess.run(["which", "espeak"],
                                  capture_output=True).returncode == 0:
                    subprocess.run(["espeak", text], check=True)
                elif subprocess.run(["which", "festival"],
                                    capture_output=True).returncode == 0:
                    proc = subprocess.Popen(["festival", "--tts"],
                                            stdin=subprocess.PIPE)
                    proc.communicate(input=text.encode())
                else:
                    # No TTS engine found on Linux
                    self.root.after(0, lambda: messagebox.showinfo(
                        "TTS not found",
                        "Install espeak:  sudo apt install espeak\n"
                        "or festival:     sudo apt install festival"
                    ))
                    return

            # Update status back on the main thread when done
            self.root.after(0, lambda: self._set_status("Speech complete.", SUCCESS))

        except Exception as exc:
            self.root.after(0, lambda: self._set_status(f"Speech error: {exc}", DANGER))


    # ==========================================================
    #  ACTION – Clear / Reset
    # ==========================================================
    def _reset(self):
        """Reset everything to the initial blank state."""
        self.original_image       = None
        self.display_image        = None
        self.working_image        = None
        self.tk_image             = None
        self.crop_mode            = False
        self.last_predicted_digit = None
        self.last_result_text     = ""
        self.last_prediction_mode = "Word"
        self.last_model_name      = self.model_var.get()

        self._canvas_placeholder()
        self._set_result("No prediction yet", TEXT_DIM)
        self._set_trans("—", TEXT_DIM)
        self._set_status("Cleared. Upload a new image.", TEXT_DIM)


    # ==========================================================
    #  HELPERS – image scaling and canvas drawing
    # ==========================================================
    @staticmethod
    def _fit(img: Image.Image, max_px: int) -> Image.Image:
        """Scale image proportionally so neither side exceeds max_px."""
        w, h = img.size
        scale = min(max_px / w, max_px / h, 1.0)
        return img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.LANCZOS)

    def _draw(self, img: Image.Image):
        """Convert PIL Image to Tkinter photo and centre it on the canvas."""
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_offset_x = (CANVAS_W - img.width)  // 2
        self.img_offset_y = (CANVAS_H - img.height) // 2
        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                 anchor="nw", image=self.tk_image)


# ==============================================================
#  ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = HandwritingApp(root)
    root.mainloop()
