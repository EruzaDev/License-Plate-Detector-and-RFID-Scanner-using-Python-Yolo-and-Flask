"""
ocr_processor.py — EasyOCR license plate recognition with multi-frame
consensus and advanced DIP preprocessing.

Pipeline:
  1. Multiple preprocessing variants are generated per crop to maximise OCR hits.
  2. For batch mode, ~5 crops of the same vehicle are collected over ~1 second.
  3. OCR runs on every preprocessed variant of every crop.
  4. All candidate plate strings are compared; the most frequent (consensus)
     result is chosen as the final plate.
"""

import re
from collections import Counter

import cv2
import numpy as np
import easyocr

# Initialise the EasyOCR reader once (model load is expensive).
# English only; GPU disabled for Raspberry Pi 5 (CPU-only).
_reader = easyocr.Reader(["en"], gpu=False)

# Regex: keep only alphanumeric characters typical of license plates
_PLATE_PATTERN = re.compile(r"[^A-Z0-9]")

# Minimum plate length to be considered valid
_MIN_PLATE_LEN = 4


# ---------------------------------------------------------------------------
# DIP preprocessing — multiple variants to maximise OCR accuracy
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess_clahe(image: np.ndarray) -> np.ndarray:
    """Grayscale → CLAHE → denoise."""
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    return denoised


def preprocess_adaptive_thresh(image: np.ndarray) -> np.ndarray:
    """Grayscale → bilateral filter → adaptive threshold (binarise)."""
    gray = _to_gray(image)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    return thresh


def preprocess_otsu(image: np.ndarray) -> np.ndarray:
    """Grayscale → Gaussian blur → Otsu binarisation."""
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_sharpen(image: np.ndarray) -> np.ndarray:
    """Grayscale → CLAHE → unsharp mask to sharpen text edges."""
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def preprocess_morph(image: np.ndarray) -> np.ndarray:
    """Grayscale → Otsu → morphological close to join broken characters."""
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


# All preprocessing pipelines to try on each crop
_PREPROCESS_PIPELINES = [
    preprocess_clahe,
    preprocess_adaptive_thresh,
    preprocess_otsu,
    preprocess_sharpen,
    preprocess_morph,
]


# ---------------------------------------------------------------------------
# Single-image OCR (returns all valid candidates from every pipeline)
# ---------------------------------------------------------------------------

def _ocr_candidates(image: np.ndarray) -> list[tuple[str, float]]:
    """
    Run OCR on multiple preprocessed variants of `image`.
    Returns a list of (plate_text, confidence) for every valid hit.
    """
    candidates: list[tuple[str, float]] = []
    for pipeline in _PREPROCESS_PIPELINES:
        processed = pipeline(image)
        results = _reader.readtext(processed, detail=1, paragraph=False)
        for _, text, conf in results:
            cleaned = _PLATE_PATTERN.sub("", text.upper())
            if len(cleaned) >= _MIN_PLATE_LEN:
                candidates.append((cleaned, round(float(conf), 4)))
    return candidates


def recognise_plate(image: np.ndarray) -> tuple[str, float]:
    """
    Run OCR on a single BGR image and return (plate_text, confidence).
    Uses multiple DIP pipelines and picks the best candidate.
    If nothing is detected, returns ("UNKNOWN", 0.0).
    """
    candidates = _ocr_candidates(image)
    if not candidates:
        return ("UNKNOWN", 0.0)
    # Pick the highest-confidence candidate
    best = max(candidates, key=lambda c: c[1])
    return best


# ---------------------------------------------------------------------------
# Batch / multi-frame consensus OCR
# ---------------------------------------------------------------------------

def recognise_plate_batch(images: list[np.ndarray]) -> tuple[str, float]:
    """
    Run OCR on a batch of BGR crops (multiple frames of the same vehicle).
    Each image is processed through every DIP pipeline.
    The plate text that appears most often across all frames wins (majority vote).
    Returns (plate_text, avg_confidence).  Falls back to ("UNKNOWN", 0.0).
    """
    all_candidates: list[tuple[str, float]] = []
    for img in images:
        all_candidates.extend(_ocr_candidates(img))

    if not all_candidates:
        return ("UNKNOWN", 0.0)

    # Group by plate text and count occurrences
    text_counts: Counter[str] = Counter()
    text_confs: dict[str, list[float]] = {}
    for plate, conf in all_candidates:
        text_counts[plate] += 1
        text_confs.setdefault(plate, []).append(conf)

    # Pick the plate with the highest vote count; break ties by avg confidence
    best_plate = max(
        text_counts,
        key=lambda p: (text_counts[p], sum(text_confs[p]) / len(text_confs[p])),
    )
    avg_conf = round(sum(text_confs[best_plate]) / len(text_confs[best_plate]), 4)

    return (best_plate, avg_conf)
