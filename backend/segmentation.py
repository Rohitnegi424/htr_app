import cv2
import numpy as np


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
