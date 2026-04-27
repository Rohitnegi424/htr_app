import os
import cv2
import numpy as np
import unicodedata

# PATHS
images_dir = r"D:/project/iam_words/words"
labels_file = r"D:/project/iam_words/words.txt"
output_dir = r"D:/project/processed_data"
failed_images_log = r"D:/project/failed_images.txt"

os.makedirs(output_dir, exist_ok=True)

# CONFIG
IMG_HEIGHT = 32
MAX_WIDTH = 256   # allow longer words safely

print("Starting preprocessing (pre2)...")

# READ LABELS
labels = {}
with open(labels_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 9:
            continue

        image_id = parts[0]
        transcription = " ".join(parts[8:])

        # Unicode normalization (CRITICAL for multilingual)
        transcription = unicodedata.normalize("NFC", transcription)

        if transcription.strip():
            labels[image_id] = transcription

print(f"Total valid labels read: {len(labels)}")

# IMAGE PROCESSING
images = []
labels_list = []
widths = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def resize_and_pad(img, height, max_width):
    h, w = img.shape
    if h <= 0 or w <= 0:
        return None, 0

    scale = height / h
    new_w = int(w * scale)

    if new_w > max_width:
        new_w = max_width

    img = cv2.resize(img, (new_w, height))

    padded = np.zeros((height, max_width), dtype=np.float32)
    padded[:, :new_w] = img

    return padded, new_w


processed = 0

for subdir, _, files in os.walk(images_dir):
    for file in files:
        if not file.endswith(".png"):
            continue

        image_id = os.path.splitext(file)[0]
        if image_id not in labels:
            continue

        file_path = os.path.join(subdir, file)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            with open(failed_images_log, "a") as f:
                f.write(f"Failed to load {file_path}\n")
            continue

        # Contrast normalization (helps uneven illumination)
        img = clahe.apply(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        img = img.astype(np.float32) / 255.0
        img, valid_width = resize_and_pad(img, IMG_HEIGHT, MAX_WIDTH)
        if img is None or valid_width <= 1:
            with open(failed_images_log, "a") as f:
                f.write(f"Invalid image size {file_path}\n")
            continue

        img = np.expand_dims(img, axis=-1)

        images.append(img)
        labels_list.append(labels[image_id])
        widths.append(valid_width)

        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} images...")

print(f"Total processed images: {processed}")

# SAVE
images_np = np.array(images, dtype=np.float32)
labels_np = np.array(labels_list, dtype=object)
widths_np = np.array(widths, dtype=np.int32)

np.savez_compressed(
    os.path.join(output_dir, "data2.npz"),
    images=images_np,
    labels=labels_np,
    widths=widths_np
)

print("Data saved successfully")
print("Final image shape:", images_np.shape)
