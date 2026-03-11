"""
camera_system.py — Motion detection + YOLOv8s vehicle detection + tracking + capture.

Pipeline per camera (runs in its own thread):
  1. MOG2 background subtraction detects motion in the frame.
  2. If motion area exceeds threshold → run YOLOv8s to detect vehicles.
  3. Euclidean-distance tracker assigns stable IDs to each vehicle.
  4. Each tracked vehicle is captured only ONCE (per-ID cooldown).
  5. Burst-crop + batch consensus OCR reads the plate.
  6. Result is saved to SQLite.
"""

import os
import time
import threading
import glob
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from ocr_processor import recognise_plate_batch
from database import insert_detection
from tracker import Tracker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s.pt")

# If model isn't in models/ yet, ultralytics will auto-download to cwd
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8s.pt"

# YOLO class IDs that count as "vehicle"
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Motion detection: minimum contour area (pixels) to count as real motion
MOTION_THRESHOLD = 8000

# Cooldown per tracked vehicle ID (seconds) — once a vehicle ID is captured,
# it won't be captured again for this duration
CAPTURE_COOLDOWN = 10.0

# YOLO confidence threshold
YOLO_CONF = 0.50

# Minimum bounding box area (pixels) to accept a vehicle detection
# Filters out tiny false-positive boxes
MIN_VEHICLE_AREA = 8000

# Burst capture: how many crops to collect and over how long
BURST_FRAMES = 5
BURST_INTERVAL = 0.2  # seconds between crops (~5 per second)

# Ensure captures directory exists
os.makedirs(CAPTURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared state — accessed by Flask for MJPEG streaming
# ---------------------------------------------------------------------------
# Latest annotated frame per camera (with YOLO bounding boxes drawn)
latest_frames = {"entrance": None, "exit": None}
frame_locks = {"entrance": threading.Lock(), "exit": threading.Lock()}


def _draw_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw YOLO bounding boxes and labels on the frame.
    `detections` is a list of dicts with keys: box, conf, cls_name.
    Returns a copy with overlays.
    """
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        track_id = det.get("track_id", "")
        tid_str = f" #{track_id}" if track_id != "" else ""
        label = f"{det['cls_name']}{tid_str} {det['conf']:.2f}"
        # Green box with label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


class CameraProcessor:
    """
    Processes a single camera feed using two threads:
      - Capture thread: reads frames as fast as possible, updates the live feed.
      - Processing thread: picks up frames and runs motion/YOLO/OCR without
        blocking the live stream.
    """

    def __init__(self, device_index: int, camera_name: str, yolo_model: YOLO):
        self.device_index = device_index
        self.camera_name = camera_name  # "entrance" or "exit"
        self.model = yolo_model
        self.running = False
        self._capture_thread = None
        self._process_thread = None

        # The latest raw frame grabbed by the capture thread
        self._raw_frame = None
        self._raw_lock = threading.Lock()

        # MOG2 background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=True
        )

        # Per-class trackers for stable vehicle IDs
        self._trackers = {
            "car": Tracker(),
            "motorcycle": Tracker(),
            "bus": Tracker(),
            "truck": Tracker(),
        }

        # Per-vehicle-ID cooldown: {(cls_name, track_id): last_capture_time}
        self._captured_ids: dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        """Start the capture and processing threads."""
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self):
        """Signal all threads to stop."""
        self.running = False

    # ------------------------------------------------------------------
    # Capture loop — fast, never blocked by YOLO / OCR
    # ------------------------------------------------------------------
    def _capture_loop(self):
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[{self.camera_name}] ERROR: cannot open /dev/video{self.device_index}")
            return

        # Use MJPEG format — critical for running two USB cams simultaneously
        # (raw YUV streams exceed USB 2.0 bandwidth on a shared controller)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Lower resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"[{self.camera_name}] Camera started on /dev/video{self.device_index}")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Always update the raw frame for the processing thread
            with self._raw_lock:
                self._raw_frame = frame

            # Immediately publish the frame for the live MJPEG stream
            with frame_locks[self.camera_name]:
                latest_frames[self.camera_name] = frame

        cap.release()
        print(f"[{self.camera_name}] Camera stopped.")

    # ------------------------------------------------------------------
    # Processing loop — runs YOLO / OCR at its own pace
    # ------------------------------------------------------------------
    def _process_loop(self):
        while self.running:
            # Grab the latest frame from the capture thread
            with self._raw_lock:
                frame = self._raw_frame
            if frame is None:
                time.sleep(0.1)
                continue

            # --- Step 1: Motion detection via MOG2 ---
            fg_mask = self.bg_subtractor.apply(frame)
            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(
                cv2.contourArea(c) > MOTION_THRESHOLD for c in contours
            )

            # --- Step 2: If motion → run YOLOv8s ---
            vehicle_detections = []
            if motion_detected:
                results = self.model(frame, conf=YOLO_CONF, verbose=False)[0]

                # Sort detections by class for the per-class trackers
                class_boxes: dict[str, list[list[int]]] = {
                    "car": [], "motorcycle": [], "bus": [], "truck": [],
                }
                class_confs: dict[str, dict[tuple[int, ...], float]] = {
                    "car": {}, "motorcycle": {}, "bus": {}, "truck": {},
                }

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    cls_name = results.names[cls_id]
                    if cls_name not in class_boxes:
                        continue
                    coords = box.xyxy[0].tolist()
                    box_area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                    if box_area < MIN_VEHICLE_AREA:
                        continue
                    int_coords = [int(c) for c in coords]
                    class_boxes[cls_name].append(int_coords)
                    class_confs[cls_name][tuple(int_coords)] = float(box.conf[0])

                # Update per-class trackers and build detection list
                for cls_name, boxes in class_boxes.items():
                    if not boxes:
                        continue
                    tracked = self._trackers[cls_name].update(boxes)
                    for x1, y1, x2, y2, track_id in tracked:
                        conf = class_confs[cls_name].get((x1, y1, x2, y2), 0.5)
                        vehicle_detections.append({
                            "box": [x1, y1, x2, y2],
                            "conf": conf,
                            "cls_name": cls_name,
                            "track_id": track_id,
                        })

            # --- Draw bounding boxes and update the live stream ---
            if vehicle_detections:
                annotated = _draw_boxes(frame, vehicle_detections)
                with frame_locks[self.camera_name]:
                    latest_frames[self.camera_name] = annotated

            # --- Step 3: For each tracked vehicle, capture once per ID ---
            now = time.time()
            for det in vehicle_detections:
                key = (det["cls_name"], det["track_id"])
                last_time = self._captured_ids.get(key, 0.0)
                if (now - last_time) >= CAPTURE_COOLDOWN:
                    self._captured_ids[key] = now
                    frame_snapshot = frame.copy()
                    threading.Thread(
                        target=self._process_vehicle,
                        args=(frame_snapshot, det),
                        daemon=True,
                    ).start()

            # Prune old IDs from the cooldown dict (older than 60 s)
            self._captured_ids = {
                k: t for k, t in self._captured_ids.items() if (now - t) < 60
            }

            # Small sleep to limit CPU usage
            time.sleep(0.03)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _crop_vehicle(self, frame: np.ndarray, detection: dict) -> np.ndarray | None:
        """Crop a vehicle region from the frame. Returns None if empty."""
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _process_vehicle(self, first_frame: np.ndarray, detection: dict):
        """
        Burst-capture ~5 crops of the same vehicle over ~1 s, then run
        batch consensus OCR across all crops for maximum accuracy.
        """
        # First crop from the triggering frame
        crops: list[np.ndarray] = []
        crop = self._crop_vehicle(first_frame, detection)
        if crop is not None:
            crops.append(crop)

        # Collect more crops from subsequent live frames
        for _ in range(BURST_FRAMES - 1):
            time.sleep(BURST_INTERVAL)
            with self._raw_lock:
                frame = self._raw_frame
            if frame is None:
                continue
            # Re-run YOLO quickly to get updated bounding box
            results = self.model(frame, conf=YOLO_CONF, verbose=False)[0]
            best_det = None
            best_area = 0
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    coords = box.xyxy[0].tolist()
                    area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                    if area > best_area:
                        best_area = area
                        best_det = {"box": coords}
            if best_det is not None:
                c = self._crop_vehicle(frame, best_det)
                if c is not None:
                    crops.append(c)

        if not crops:
            return

        # --- Batch consensus OCR across all collected crops ---
        plate, confidence = recognise_plate_batch(crops)

        # Build filename and save
        now = datetime.now()
        ts_file = now.strftime("%Y-%m-%d_%H-%M-%S")
        ts_db = now.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{ts_file}_{self.camera_name}_{plate}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)

        cv2.imwrite(filepath, first_frame)

        rel_path = os.path.join("captures", filename)

        insert_detection(
            plate_number=plate,
            camera=self.camera_name,
            timestamp=ts_db,
            image_path=rel_path,
            confidence=confidence,
        )
        print(f"[{self.camera_name}] Detected: {plate} (conf={confidence:.2f}, "
              f"{len(crops)} crops) → {filename}")


# ---------------------------------------------------------------------------
# Module-level helpers to start / stop the whole camera system
# ---------------------------------------------------------------------------
_processors: list[CameraProcessor] = []

# Load YOLO model once, shared across both camera threads (thread-safe for inference)
_yolo_model = None


def _get_model():
    global _yolo_model
    if _yolo_model is None:
        print("[system] Loading YOLOv8s model …")
        _yolo_model = YOLO(MODEL_PATH)
        print("[system] YOLOv8s ready.")
    return _yolo_model


def start_cameras():
    """
    Load the YOLO model but do NOT start any cameras automatically.
    Users assign cameras from the dashboard.
    """
    _get_model()
    print("[system] Ready — assign cameras from the dashboard.")


def stop_cameras():
    """Stop all running camera threads."""
    for p in _processors:
        p.stop()
    _processors.clear()


def _get_device_name(idx: int) -> str:
    """Read the human-readable device name from sysfs, or fall back."""
    name_path = f"/sys/class/video4linux/video{idx}/name"
    try:
        with open(name_path) as f:
            return f.read().strip()
    except OSError:
        return f"Camera {idx}"


def _is_capture_device(idx: int) -> bool:
    """
    Check if /dev/videoN is a real video-capture device by looking for
    'Video Capture' in its V4L2 capabilities via sysfs device uevent,
    or by checking the sysfs index file. Metadata-only nodes are filtered out.
    """
    # Method 1: check for 'capture' in the device_caps via uevent
    uevent_path = f"/sys/class/video4linux/video{idx}/uevent"
    try:
        with open(uevent_path) as f:
            uevent = f.read()
            # Only capture nodes have DEVNAME; metadata ones are different
    except OSError:
        pass

    # Method 2: sysfs index — capture nodes have index 0
    index_path = f"/sys/class/video4linux/video{idx}/index"
    try:
        with open(index_path) as f:
            return f.read().strip() == "0"
    except OSError:
        pass

    return True  # can't determine — assume valid


def _find_video_indices() -> list[int]:
    """
    Scan /sys/class/video4linux/ to find which video device indices exist.
    Only returns indices that exist on disk — no blind probing.
    """
    indices: list[int] = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        name = os.path.basename(path)
        try:
            idx = int(name.replace("video", ""))
            indices.append(idx)
        except ValueError:
            continue
    return indices


def list_video_devices() -> list[dict]:
    """
    List available video-capture devices. Only probes devices that actually
    exist in /sys/class/video4linux/, and filters out metadata-only nodes.
    Devices currently in use by our own processors are included without
    re-opening them.
    Each entry: {"index": int, "name": str, "in_use_by": str|None}.
    """
    # Build a map of indices already held by our processors
    in_use: dict[int, str] = {}
    for p in _processors:
        in_use[p.device_index] = p.camera_name

    # Find which indices truly exist on this system
    existing = _find_video_indices()

    devices: list[dict] = []
    for idx in existing:
        if idx in in_use:
            devices.append({
                "index": idx,
                "name": _get_device_name(idx),
                "in_use_by": in_use[idx],
            })
            continue

        # Skip metadata-only nodes
        if not _is_capture_device(idx):
            continue

        # Open with V4L2 backend specifically to avoid FFMPEG fallback noise
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                devices.append({
                    "index": idx,
                    "name": _get_device_name(idx),
                    "in_use_by": None,
                })
    return devices


def get_camera_assignments() -> dict:
    """Return current device index for each camera name."""
    assignments = {}
    for p in _processors:
        assignments[p.camera_name] = p.device_index
    return assignments


def reassign_camera(camera_name: str, new_device_index: int):
    """
    Stop a running camera processor (if any) and start it with a new device index.
    `camera_name` must be 'entrance' or 'exit'.
    """
    model = _get_model()
    # Find and stop the existing processor
    for i, p in enumerate(_processors):
        if p.camera_name == camera_name:
            p.stop()
            time.sleep(0.5)  # give capture thread time to release the device
            _processors.pop(i)
            break

    # Reset the shared frame so the dashboard doesn't show a stale image
    with frame_locks[camera_name]:
        latest_frames[camera_name] = None

    # Start a new processor with the new device
    new_proc = CameraProcessor(
        device_index=new_device_index,
        camera_name=camera_name,
        yolo_model=model,
    )
    new_proc.start()
    _processors.append(new_proc)
    print(f"[system] {camera_name} camera started on /dev/video{new_device_index}")


def stop_camera(camera_name: str):
    """Stop a single camera by name."""
    for i, p in enumerate(_processors):
        if p.camera_name == camera_name:
            p.stop()
            time.sleep(0.3)
            _processors.pop(i)
            with frame_locks[camera_name]:
                latest_frames[camera_name] = None
            print(f"[system] {camera_name} camera stopped.")
            return True
    return False
