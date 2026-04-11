"""
download_models.py - One-time script to download plate model + EasyOCR assets.
Run once so the app can start without downloading models at runtime.
"""

import os
import shutil
import urllib.error
import urllib.request

import easyocr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = "license_plate_detector.pt"
MODEL_DEST = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_URLS = [
    "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt",
    "https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/main/license_plate_detector.pt",
]

os.makedirs(MODELS_DIR, exist_ok=True)


def _is_valid_weights_file(path: str) -> bool:
    """Basic validation to reject tiny files, HTML, and git-lfs pointers."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 200 * 1024:
        return False

    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return False

    head_lower = head.lower()
    if b"git-lfs" in head_lower or b"<html" in head_lower:
        return False
    return True


def _try_move_local_copy() -> None:
    """Move a manually-downloaded model from project root into models/."""
    src = os.path.join(BASE_DIR, MODEL_NAME)
    if os.path.exists(src) and not os.path.exists(MODEL_DEST):
        shutil.move(src, MODEL_DEST)
        print(f"  Moved existing local model to {MODEL_DEST}")


def _download_model() -> bool:
    """Try multiple download sources for the requested model file."""
    tmp_path = MODEL_DEST + ".tmp"
    for url in MODEL_URLS:
        print(f"  Trying {url}")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            if _is_valid_weights_file(tmp_path):
                os.replace(tmp_path, MODEL_DEST)
                return True
            print("    Downloaded file is not a valid weights file. Trying next source...")
        except urllib.error.URLError as exc:
            print(f"    Download failed: {exc}")
        except Exception as exc:
            print(f"    Download failed: {exc}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return False


print("Preparing license plate detector model...")
if _is_valid_weights_file(MODEL_DEST):
    print(f"  Already exists: {MODEL_DEST}")
else:
    _try_move_local_copy()
    if not _is_valid_weights_file(MODEL_DEST):
        if not _download_model():
            print("  ERROR: Failed to download license_plate_detector.pt automatically.")
            print("  Download it manually from the upstream repository and place it at:")
            print(f"  {MODEL_DEST}")
            raise SystemExit(1)
    print(f"  Ready: {MODEL_DEST}")

print("Downloading EasyOCR English model...")
easyocr.Reader(["en"], gpu=False)
print("  EasyOCR models cached.")

print("\nAll models ready. You can now run: python app.py")
