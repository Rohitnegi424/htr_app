import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


SEED = 42
IMG_HEIGHT = 32
IMG_WIDTH = 256
VAL_SPLIT = 0.1
BATCH_SIZE = 16
EPOCHS = 80
WARMUP_EPOCHS = 5
BASE_LR = 3e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_SAMPLES = 0
ENABLE_MIXED_PRECISION = False

DATA_PATH = "/content/drive/MyDrive/data/data2.npz"
OUTPUT_DIR = "/content/drive/MyDrive/data"

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "htr_ctc_words_multilang_best_v4.keras")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "htr_ctc_words_multilang_final_v4.keras")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "htr_ctc_words_multilang_history_v4.csv")


tf.random.set_seed(SEED)
np.random.seed(SEED)


def configure_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        if ENABLE_MIXED_PRECISION:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled.")
            except ValueError:
                print("Mixed precision not available, continuing with float32.")
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
            print("Mixed precision disabled for stability.")
    else:
        print("GPU not found, training will run on CPU.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_runtime_config():
    batch_size = int(os.getenv("MODEL4_BATCH_SIZE", BATCH_SIZE))
    epochs = int(os.getenv("MODEL4_EPOCHS", EPOCHS))
    val_split = float(os.getenv("MODEL4_VAL_SPLIT", VAL_SPLIT))
    warmup_epochs = int(os.getenv("MODEL4_WARMUP_EPOCHS", WARMUP_EPOCHS))
    max_samples = int(os.getenv("MODEL4_MAX_SAMPLES", MAX_SAMPLES))
    base_lr = float(os.getenv("MODEL4_BASE_LR", 1e-4))
    return batch_size, epochs, val_split, warmup_epochs, max_samples, base_lr


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=True)
    images = data["images"].astype(np.float32)
    texts = data["labels"].tolist()

    print(f"Loaded {len(images)} samples from {DATA_PATH}")
    print(f"Image tensor shape: {images.shape}")

    charset = sorted(set("".join(texts)))
    char_to_idx = {c: i for i, c in enumerate(charset)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    num_chars = len(charset)
    blank_idx = num_chars
    max_label_len = max(len(t) for t in texts)

    labels = np.full((len(texts), max_label_len), blank_idx, dtype=np.int32)
    for i, text in enumerate(texts):
        for j, ch in enumerate(text):
            labels[i, j] = char_to_idx[ch]

    label_length = np.array([[len(t)] for t in texts], dtype=np.int32)

    return images, texts, labels, label_length, charset, idx_to_char, blank_idx


def build_datasets(images, labels, label_length, batch_size, val_split):
    num_samples = len(images)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    val_size = int(num_samples * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_images = images[train_idx]
    val_images = images[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    train_label_length = label_length[train_idx]
    val_label_length = label_length[val_idx]

    input_length = np.full((num_samples, 1), IMG_WIDTH // 4, dtype=np.int32)
    train_input_length = input_length[train_idx]
    val_input_length = input_length[val_idx]

    train_targets = np.zeros((len(train_idx),), dtype=np.float32)
    val_targets = np.zeros((len(val_idx),), dtype=np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((
        {
            "image": train_images,
            "labels": train_labels,
            "input_length": train_input_length,
            "label_length": train_label_length,
        },
        train_targets,
    ))
    train_ds = train_ds.shuffle(min(len(train_idx), 8192), seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        {
            "image": val_images,
            "labels": val_labels,
            "input_length": val_input_length,
            "label_length": val_label_length,
        },
        val_targets,
    ))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    return train_ds, val_ds


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


def conv_block(x, filters, pool=True, dropout=0.0):
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


def transformer_block(x, d_model, num_heads, ff_dim, dropout):
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


def build_model(num_chars, max_label_len):
    augment = tf.keras.Sequential(
        [
            layers.RandomRotation(0.02),
            layers.RandomTranslation(0.03, 0.03),
            layers.RandomZoom(height_factor=0.05, width_factor=0.08),
            layers.RandomContrast(0.15),
        ],
        name="augment",
    )

    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    labels_input = layers.Input(shape=(max_label_len,), dtype="int32", name="labels")
    input_len_input = layers.Input(shape=(1,), dtype="int32", name="input_length")
    label_len_input = layers.Input(shape=(1,), dtype="int32", name="label_length")

    x = augment(image_input)
    x = conv_block(x, 64, pool=True, dropout=0.05)
    x = conv_block(x, 128, pool=True, dropout=0.08)
    x = conv_block(x, 256, pool=False, dropout=0.10)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((IMG_WIDTH // 4, 8 * 256))(x)
    x = layers.Dense(256)(x)
    x = PositionalEncoding(name="positional_encoding")(x)

    for _ in range(4):
        x = transformer_block(x, d_model=256, num_heads=8, ff_dim=768, dropout=0.10)

    x = layers.Bidirectional(layers.LSTM(192, return_sequences=True, dropout=0.15))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(256, activation="gelu")(x)
    y_pred = layers.Dense(num_chars + 1, activation="softmax", dtype="float32", name="y_pred")(x)

    def ctc_loss(args):
        y_pred_arg, labels_arg, input_len_arg, label_len_arg = args
        return tf.keras.backend.ctc_batch_cost(
            labels_arg, y_pred_arg, input_len_arg, label_len_arg
        )

    def identity_loss(_y_true, y_pred_arg):
        return y_pred_arg

    loss = layers.Lambda(ctc_loss, name="ctc")(
        [y_pred, labels_input, input_len_input, label_len_input]
    )

    model = models.Model(
        inputs=[image_input, labels_input, input_len_input, label_len_input],
        outputs=loss,
        name="htr_transformer_ctc_v4",
    )
    model.identity_loss = identity_loss
    return model


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.base_lr * (step + 1.0) / tf.maximum(warmup_steps, 1.0)
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


def main():
    configure_runtime()
    batch_size, epochs, val_split, warmup_epochs, max_samples, base_lr = get_runtime_config()
    images, _, labels, label_length, charset, _, _ = load_data()
    if max_samples > 0:
        images = images[:max_samples]
        labels = labels[:max_samples]
        label_length = label_length[:max_samples]
        print(f"Using subset of {max_samples} samples for this run.")

    max_label_len = labels.shape[1]
    train_ds, val_ds = build_datasets(images, labels, label_length, batch_size, val_split)

    steps_per_epoch = max(1, math.ceil((len(images) * (1 - val_split)) / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    lr_schedule = WarmupCosineDecay(base_lr, warmup_steps, total_steps, MIN_LR)

    model = build_model(len(charset), max_label_len)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=model.identity_loss,
    )

    model.summary()

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.CSVLogger(HISTORY_PATH),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callback_list,
    )

    model.save(FINAL_MODEL_PATH)
    print("Training complete.")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"History log: {HISTORY_PATH}")
    return history


if __name__ == "__main__":
    main()
