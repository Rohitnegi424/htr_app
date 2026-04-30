import math
import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf

IMG_HEIGHT = 32
IMG_WIDTH = 256


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
        return inputs + tf.cast(self.positional_encoding, dtype=inputs.dtype)


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


def build_v4_inference_model(num_output_tokens):
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    x = conv_block(image_input, 64, pool=True, dropout=0.05)
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
    y_pred = layers.Dense(
        num_output_tokens,
        activation="softmax",
        dtype="float32",
        name="y_pred",
    )(x)

    return models.Model(inputs=image_input, outputs=y_pred)


def load_model(weights_path, num_chars):
    model = build_v4_inference_model(num_chars)
    model.load_weights(weights_path)
    return model
