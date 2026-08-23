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
try:
    import easyocr
    _reader = easyocr.Reader(["en"], gpu=False)
except Exception as _easyocr_err:
    easyocr = None
    _reader = None

_fast_alpr_instance = None
_fast_alpr_failed = False


def get_fast_alpr():
    """Return singleton ALPR instance if fast_alpr is installed."""
    global _fast_alpr_instance, _fast_alpr_failed
    if _fast_alpr_failed:
        return None
    if _fast_alpr_instance is None:
        try:
            from fast_alpr import ALPR
            _fast_alpr_instance = ALPR(
                detector_model="yolo-v9-s-608-license-plate-end2end",
                ocr_model="cct-s-v2-global-model",
            )
            print("[ocr_processor] Fast-ALPR engine (yolo-v9-s-608 + cct-s-v2-global) loaded successfully.")
        except Exception as err:
            print(f"[ocr_processor] Fast-ALPR unavailable ({err}); using EasyOCR fallback.")
            _fast_alpr_failed = True
            return None
    return _fast_alpr_instance

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
    Normalize plate text output from Fast-ALPR without coercing characters.
    Preserves exact OCR text output matching Fast-ALPR / HuggingFace demo.
    """
    cleaned = normalize_plate_text(text)
    if not cleaned or cleaned == "UNKNOWN":
        return ("UNKNOWN", False, "INVALID")

    if _PH_MODERN_PATTERN.fullmatch(cleaned):
        return (cleaned, True, "MODERN")
    if _PH_CLASSIC_PATTERN.fullmatch(cleaned):
        return (cleaned, True, "CLASSIC")

    return (cleaned, True, "FAST_ALPR")


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
# OCR character confusion clusters for similarity scoring
_OCR_CONFUSION_PAIRS = {
    frozenset(("O", "0")), frozenset(("O", "Q")), frozenset(("0", "Q")),
    frozenset(("O", "D")), frozenset(("0", "D")), frozenset(("I", "1")),
    frozenset(("I", "L")), frozenset(("1", "L")), frozenset(("I", "|")),
    frozenset(("S", "5")), frozenset(("B", "8")), frozenset(("Z", "2")),
    frozenset(("G", "6")), frozenset(("C", "G")), frozenset(("D", "0")),
    frozenset(("T", "7")), frozenset(("A", "4")), frozenset(("U", "V")),
    frozenset(("E", "F")), frozenset(("K", "X")), frozenset(("M", "N")),
    frozenset(("P", "R")), frozenset(("H", "N")), frozenset(("Y", "V")),
}


def are_ocr_confusable(c1: str, c2: str) -> bool:
    """Return True if two characters are visually confusable in typical OCR."""
    if c1.upper() == c2.upper():
        return True
    return frozenset((c1.upper(), c2.upper())) in _OCR_CONFUSION_PAIRS


def ocr_levenshtein_distance(s1: str, s2: str) -> float:
    """Compute weighted Levenshtein distance accounting for common OCR confusions."""
    norm1 = normalize_plate_text(s1)
    norm2 = normalize_plate_text(s2)
    if norm1 == norm2:
        return 0.0
    if not norm1:
        return float(len(norm2))
    if not norm2:
        return float(len(norm1))

    m, n = len(norm1), len(norm2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1, c2 = norm1[i - 1], norm2[j - 1]
            if c1 == c2:
                cost = 0.0
            elif are_ocr_confusable(c1, c2):
                cost = 0.25  # Small penalty for common OCR confusion
            else:
                cost = 1.0

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,        # deletion
                dp[i][j - 1] + 1.0,        # insertion
                dp[i - 1][j - 1] + cost,    # substitution
            )

    return dp[m][n]


def calculate_ocr_similarity(s1: str | None, s2: str | None) -> float:
    """
    Return OCR similarity score (0.0 to 100.0) with tolerance for OCR character misreads.
    """
    norm1 = normalize_plate_text(s1)
    norm2 = normalize_plate_text(s2)
    if not norm1 or not norm2:
        return 0.0
    if norm1 == norm2:
        return 100.0

    max_len = max(len(norm1), len(norm2))
    dist = ocr_levenshtein_distance(norm1, norm2)
    sim = max(0.0, (1.0 - (dist / max_len)) * 100.0)
    return round(sim, 2)


def find_best_fuzzy_match(query: str | None, candidates: list[str], min_score: float = 65.0) -> tuple[str | None, float]:
    """
    Find the best matching plate string from candidates using OCR-tolerant similarity.
    """
    query_norm = normalize_plate_text(query)
    if not query_norm or query_norm == "UNKNOWN" or not candidates:
        return (None, 0.0)

    best_match = None
    best_score = 0.0

    for cand in candidates:
        cand_norm = normalize_plate_text(cand)
        if not cand_norm:
            continue
        # Exact match check
        if query_norm == cand_norm:
            return (cand, 100.0)

        score = calculate_ocr_similarity(query_norm, cand_norm)
        if score > best_score:
            best_score = score
            best_match = cand

    if best_score >= min_score:
        return (best_match, round(best_score, 2))
    return (None, round(best_score, 2))


def match_plate(
    ocr_result: str | None,
    registered_plates: list[str],
    threshold: float = 80.0,
    review_threshold: float = 60.0,
) -> tuple[str | None, float, str]:
    """
    Match OCR result against registered plates using OCR-tolerant similarity.

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

    best_match, score = find_best_fuzzy_match(query, cleaned_choices, min_score=0.0)

    if best_match is None:
        return (None, 0.0, "NO_MATCH")

    if score >= threshold:
        return (best_match, score, "AUTO_MATCHED")
    if score >= review_threshold:
        return (best_match, score, "NEEDS_REVIEW")
    return (None, score, "NO_MATCH")


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
    Run OCR on image variants using Fast-ALPR if available, falling back to EasyOCR.
    Returns a list of (plate_text, confidence) for valid hits.
    """
    candidates: list[tuple[str, float]] = []
    alpr = get_fast_alpr()

    if alpr is not None:
        try:
            results = alpr.predict(image)
            if results:
                for res in results:
                    if res.ocr and res.ocr.text:
                        cleaned = normalize_plate_text(res.ocr.text)
                        if len(cleaned) >= _MIN_PLATE_LEN:
                            conf = (
                                sum(res.ocr.confidence) / len(res.ocr.confidence)
                                if isinstance(res.ocr.confidence, list) and res.ocr.confidence
                                else (res.ocr.confidence if isinstance(res.ocr.confidence, (int, float)) else res.detection.confidence)
                            )
                            candidates.append((cleaned, round(float(conf), 4)))
            if not candidates:
                ocr_res = alpr.ocr.predict(image)
                if ocr_res and ocr_res.text:
                    cleaned = normalize_plate_text(ocr_res.text)
                    if len(cleaned) >= _MIN_PLATE_LEN:
                        conf = (
                            sum(ocr_res.confidence) / len(ocr_res.confidence)
                            if isinstance(ocr_res.confidence, list) and ocr_res.confidence
                            else (ocr_res.confidence if isinstance(ocr_res.confidence, (int, float)) else 0.0)
                        )
                        candidates.append((cleaned, round(float(conf), 4)))
        except Exception:
            pass

        if candidates:
            return candidates

    if _reader is not None:
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
