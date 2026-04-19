"""
ocr_processor.py — EasyOCR license plate recognition with multi-frame
consensus and advanced DIP preprocessing.

Pipeline:
  1. Multiple preprocessing variants are generated per crop to maximise OCR hits.
    2. For batch mode, ~5 crops of the same plate are collected over ~1 second.
  3. OCR runs on every preprocessed variant of every crop.
  4. All candidate plate strings are compared; the most frequent (consensus)
     result is chosen as the final plate.
"""

import re
from collections import Counter
from difflib import SequenceMatcher

import cv2
import numpy as np
import easyocr

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover - graceful fallback when rapidfuzz is missing
    fuzz = None
    process = None

# Initialise the EasyOCR reader once (model load is expensive).
# English only; GPU disabled for Raspberry Pi 5 (CPU-only).
_reader = easyocr.Reader(["en"], gpu=False)

# Regex: keep only alphanumeric characters typical of license plates
_PLATE_PATTERN = re.compile(r"[^A-Z0-9]")

# PH LTO plate patterns:
# - Modern: ABC1234
# - Classic: AB1234
_PH_MODERN_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{4}$")
_PH_CLASSIC_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{4}$")

# OCR disambiguation map for common confusions.
_CHAR_TO_INT = {"O": "0", "I": "1", "S": "5", "G": "6", "B": "8", "Z": "2"}
_INT_TO_CHAR = {v: k for k, v in _CHAR_TO_INT.items()}

# Minimum plate length to be considered valid
_MIN_PLATE_LEN = 4


def normalize_plate_text(text: str | None) -> str:
    """Return uppercase alphanumeric-only plate text."""
    if not text:
        return ""
    return _PLATE_PATTERN.sub("", str(text).upper())


def _to_letter(ch: str) -> str:
    """Coerce a character into a letter when OCR produced a lookalike digit."""
    c = ch.upper()
    if "A" <= c <= "Z":
        return c
    return _INT_TO_CHAR.get(c, c)


def _to_digit(ch: str) -> str:
    """Coerce a character into a digit when OCR produced a lookalike letter."""
    c = ch.upper()
    if "0" <= c <= "9":
        return c
    return _CHAR_TO_INT.get(c, c)


def _coerce_modern_plate(raw: str) -> str:
    """Convert text to PH modern plate layout (3 letters + 4 digits)."""
    letters = "".join(_to_letter(ch) for ch in raw[:3])
    digits = "".join(_to_digit(ch) for ch in raw[3:7])
    return letters + digits


def _coerce_classic_plate(raw: str) -> str:
    """Convert text to PH classic plate layout (2 letters + 4 digits)."""
    letters = "".join(_to_letter(ch) for ch in raw[:2])
    digits = "".join(_to_digit(ch) for ch in raw[2:6])
    return letters + digits


def correct_ph_plate(text: str | None) -> tuple[str, bool, str]:
    """
    Apply PH plate cleanup rules and position-aware OCR correction.

    Returns
    -------
    tuple[str, bool, str]
        (corrected_plate_no_space, is_valid, format_name)
        format_name is one of: MODERN, CLASSIC, INVALID.
    """
    cleaned = normalize_plate_text(text)
    if not cleaned or cleaned == "UNKNOWN":
        return ("UNKNOWN", False, "INVALID")

    attempts: list[tuple[str, str]] = []
    if len(cleaned) >= 7:
        attempts.append(("MODERN", _coerce_modern_plate(cleaned[:7])))
    if len(cleaned) >= 6:
        attempts.append(("CLASSIC", _coerce_classic_plate(cleaned[:6])))

    if not attempts:
        return (cleaned, False, "INVALID")

    for fmt, candidate in attempts:
        if fmt == "MODERN" and _PH_MODERN_PATTERN.fullmatch(candidate):
            return (candidate, True, fmt)
        if fmt == "CLASSIC" and _PH_CLASSIC_PATTERN.fullmatch(candidate):
            return (candidate, True, fmt)

    # Keep the highest-priority corrected candidate even when invalid.
    best_fmt, best_candidate = attempts[0]
    return (best_candidate, False, best_fmt)


def format_plate_for_display(plate: str | None) -> str:
    """Render normalized plate as human-friendly display text with spacing."""
    cleaned = normalize_plate_text(plate)
    if _PH_MODERN_PATTERN.fullmatch(cleaned):
        return f"{cleaned[:3]} {cleaned[3:]}"
    if _PH_CLASSIC_PATTERN.fullmatch(cleaned):
        return f"{cleaned[:2]} {cleaned[2:]}"
    return cleaned or "UNKNOWN"


def _fallback_extract_one(query: str, choices: list[str]) -> tuple[str, float] | None:
    """Fallback fuzzy matching when rapidfuzz is unavailable."""
    if not choices:
        return None
    best_choice = ""
    best_score = -1.0
    for choice in choices:
        score = SequenceMatcher(None, query, choice).ratio() * 100.0
        if score > best_score:
            best_choice = choice
            best_score = score
    if best_score < 0:
        return None
    return (best_choice, round(best_score, 2))


def match_plate(
    ocr_result: str | None,
    registered_plates: list[str],
    threshold: float = 85.0,
    review_threshold: float = 60.0,
) -> tuple[str | None, float, str]:
    """
    Match OCR result against registered plates using Levenshtein similarity.

    Returns
    -------
    tuple[str | None, float, str]
        (best_match, score, status)
        status: AUTO_MATCHED | NEEDS_REVIEW | NO_MATCH | NO_REGISTRY
    """
    query = normalize_plate_text(ocr_result)
    cleaned_choices = [normalize_plate_text(p) for p in registered_plates if p]
    cleaned_choices = [p for p in cleaned_choices if p]

    if not cleaned_choices:
        return (None, 0.0, "NO_REGISTRY")
    if not query or query == "UNKNOWN":
        return (None, 0.0, "NO_MATCH")

    best_match: str | None = None
    score = 0.0

    if process is not None and fuzz is not None:
        result = process.extractOne(query, cleaned_choices, scorer=fuzz.ratio)
        if result is not None:
            best_match, score, _ = result
            score = float(score)
    else:
        fallback = _fallback_extract_one(query, cleaned_choices)
        if fallback is not None:
            best_match, score = fallback

    if best_match is None:
        return (None, 0.0, "NO_MATCH")

    if score >= threshold:
        return (best_match, round(score, 2), "AUTO_MATCHED")
    if score >= review_threshold:
        return (best_match, round(score, 2), "NEEDS_REVIEW")
    return (None, round(score, 2), "NO_MATCH")


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


def _plate_bbox_from_contours(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Estimate a license-plate bounding box from a vehicle crop using contour
    geometry (aspect ratio + area heuristics).
    """
    gray = _to_gray(image)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 50, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    frame_area = float(max(h * w, 1))
    best_box = None
    best_score = 0.0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) < 4 or len(approx) > 10:
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        if bw < 20 or bh < 8:
            continue

        area = float(bw * bh)
        area_ratio = area / frame_area
        aspect = bw / float(max(bh, 1))

        if not (2.0 <= aspect <= 6.8):
            continue
        if not (0.006 <= area_ratio <= 0.35):
            continue

        # Score plate-likeness: typical aspect ratio and plausible size.
        aspect_score = max(0.0, 1.0 - abs(aspect - 4.0) / 4.0)
        area_score = max(0.0, 1.0 - abs(area_ratio - 0.06) / 0.06)
        fill_ratio = min(1.0, cv2.contourArea(cnt) / max(area, 1.0))
        score = (0.5 * aspect_score) + (0.35 * area_score) + (0.15 * fill_ratio)

        if score > best_score:
            best_score = score
            best_box = (x, y, x + bw, y + bh)

    return best_box


def _expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    """Expand a bounding box by a ratio and clamp to image bounds."""
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return x1, y1, x2, y2


def extract_plate_crop(
    vehicle_image: np.ndarray,
    pad_ratio: float = 0.08,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """
    Detect a plate-like region inside a vehicle image and return
    (plate_crop, plate_bbox). If no plate region is found, returns (None, None).
    """
    if vehicle_image is None or vehicle_image.size == 0:
        return None, None

    h, w = vehicle_image.shape[:2]
    box = _plate_bbox_from_contours(vehicle_image)
    if box is None:
        return None, None

    x1, y1, x2, y2 = _expand_box(box, w, h, pad_ratio)
    crop = vehicle_image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


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
    Run OCR on a batch of BGR crops (multiple frames of the same plate).
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
