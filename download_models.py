"""
download_models.py — One-time script to download YOLOv8s + EasyOCR models.
Run this once after install so the main system starts without network access.
"""

import os
import shutil
from ultralytics import YOLO
import easyocr

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- YOLOv8s ----
print("Downloading YOLOv8s model...")
model = YOLO("yolov8s.pt")  # auto-downloads if not cached

# Move the weights into our models/ folder if not already there
dest = os.path.join(MODELS_DIR, "yolov8s.pt")
if not os.path.exists(dest):
    src = "yolov8s.pt"
    if os.path.exists(src):
        shutil.move(src, dest)
        print(f"  Moved to {dest}")
    else:
        print(f"  Model cached by ultralytics (will auto-resolve at runtime)")
else:
    print(f"  Already exists: {dest}")

# ---- EasyOCR ----
print("Downloading EasyOCR English model...")
easyocr.Reader(["en"], gpu=False)  # pre-caches OCR model files
print("  EasyOCR models cached.")

print("\nAll models ready. You can now run:  python app.py")
