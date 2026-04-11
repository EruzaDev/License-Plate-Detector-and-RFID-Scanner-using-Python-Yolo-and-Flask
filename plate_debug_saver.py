"""
Optional debug helper for bug testing.

This file writes extra license-plate crops to captures/debug_plates/.
You can delete this file when you no longer need debug plate image copies.
"""

import os
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_PLATE_DIR = os.path.join(BASE_DIR, "captures", "debug_plates")
os.makedirs(DEBUG_PLATE_DIR, exist_ok=True)


def _safe_plate_text(text: str) -> str:
    cleaned = "".join(ch for ch in text if ch.isalnum())
    return cleaned or "UNKNOWN"


def save_debug_plate_image(
    plate_image: np.ndarray,
    camera_name: str,
    plate_text: str,
    confidence: float,
) -> str | None:
    """
    Save a debug image copy of a detected license plate crop.

    Returns the saved file path on success, or None on failure.
    """
    if plate_image is None or plate_image.size == 0:
        return None

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    plate = _safe_plate_text(plate_text)
    conf = f"{confidence:.2f}".replace(".", "p")
    filename = f"{ts}_{camera_name}_{plate}_c{conf}.jpg"
    path = os.path.join(DEBUG_PLATE_DIR, filename)

    ok = cv2.imwrite(path, plate_image)
    return path if ok else None
