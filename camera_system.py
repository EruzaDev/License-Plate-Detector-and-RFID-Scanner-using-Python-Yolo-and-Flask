"""
camera_system.py - Motion detection + 2-stage YOLO + OCR pipeline.

Pipeline per camera (runs in its own thread):
    1. MOG2 background subtraction detects motion in the frame.
    2. Stage 1 YOLOv8n detects vehicles on the full frame.
    3. Stage 2 YOLO plate model runs inside each vehicle ROI.
    4. Euclidean tracker keeps stable IDs for detected plates.
    5. Burst-crop + consensus OCR + PH format correction.
    6. Optional fuzzy match against registered plates.
    7. Result is saved to SQLite.
"""

import os
import time
import threading
import glob
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from ocr_processor import recognise_plate_batch, correct_ph_plate, match_plate
from database import insert_detection, get_registered_plates, enqueue_manual_input
from tracker import Tracker

try:
    from plate_debug_saver import save_debug_plate_image
except Exception:
    # Optional helper for bug testing. Safe no-op when file is removed.
    def save_debug_plate_image(*_args, **_kwargs):
        return None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
MODEL_PATH = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
VEHICLE_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "yolov8n.pt"),
    os.path.join(BASE_DIR, "yolov8s.pt"),
    "yolov8n.pt",
]

# Allow a project-root fallback for manual placement of the model file.
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "license_plate_detector.pt")


def _env_float(
    name: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read a float from env with optional bounds and safe fallback."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)

    try:
        value = float(raw)
    except ValueError:
        print(f"[config] Invalid {name}={raw!r}; using default {default}")
        return float(default)

    if min_value is not None and value < min_value:
        print(f"[config] {name} below minimum ({min_value}); clamping")
        value = min_value
    if max_value is not None and value > max_value:
        print(f"[config] {name} above maximum ({max_value}); clamping")
        value = max_value

    return float(value)

# Motion detection: minimum contour area (pixels) to count as real motion
MOTION_THRESHOLD = 8000

# Stage 1 vehicle detection settings (COCO classes)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
VEHICLE_CONF = 0.35
MIN_VEHICLE_AREA = 2500

# Vehicle ROI expansion to avoid clipping plates near bumper edges
VEHICLE_ROI_PAD = 0.08

# Cooldown per tracked plate ID (seconds) - once a plate ID is captured,
# it won't be captured again for this duration
CAPTURE_COOLDOWN = 10.0

# YOLO confidence threshold
YOLO_CONF = 0.50

# Minimum bounding box area (pixels) to accept a plate detection
# Filters out tiny false-positive boxes
MIN_PLATE_AREA = 700

# Burst capture: how many crops to collect and over how long
BURST_FRAMES = 5
BURST_INTERVAL = 0.2  # seconds between crops (~5 per second)

# Batch merge controls: detections that overlap/are very close are treated
# as the same input event, so burst frames count as one DB entry.
BATCH_MERGE_IOU = _env_float("BATCH_MERGE_IOU", 0.20, min_value=0.0, max_value=1.0)
BATCH_CENTER_DISTANCE = _env_float("BATCH_CENTER_DISTANCE", 80.0, min_value=0.0)

# Ensure captures directory exists
os.makedirs(CAPTURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared state — accessed by Flask for MJPEG streaming
# ---------------------------------------------------------------------------
# Latest annotated frame per camera (with YOLO bounding boxes drawn)
latest_frames = {"entrance": None, "exit": None}
frame_locks = {"entrance": threading.Lock(), "exit": threading.Lock()}
HAS_V4L2_SYSFS = os.path.isdir("/sys/class/video4linux")
GENERIC_PROBE_COUNT = 10


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

    def __init__(
        self,
        device_index: int,
        camera_name: str,
        plate_model: YOLO,
        vehicle_model: YOLO,
    ):
        self.device_index = device_index
        self.camera_name = camera_name  # "entrance" or "exit"
        self.plate_model = plate_model
        self.vehicle_model = vehicle_model
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

        # Single tracker for plate boxes
        self._plate_tracker = Tracker()

        # Per-track-ID cooldown: {track_id: last_capture_time}
        self._captured_ids: dict[int, float] = {}

        # Recent capture regions used to merge duplicate triggers into one batch.
        # Each item: {"box": [x1, y1, x2, y2], "time": float}
        self._recent_batch_regions: list[dict] = []

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
        cap, backend_name = _open_video_capture(self.device_index)
        if cap is None:
            print(f"[{self.camera_name}] ERROR: cannot open {_device_label(self.device_index)}")
            return

        # Use MJPEG format on Linux/V4L2 to reduce USB bandwidth usage.
        if backend_name == "CAP_V4L2":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Lower resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"[{self.camera_name}] Camera started on {_device_label(self.device_index)} "
              f"({backend_name})")

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

            # --- Step 2: If motion -> run 2-stage vehicle/plate detection ---
            plate_detections = []
            if motion_detected:
                detected_boxes: list[list[int]] = []
                box_meta: dict[tuple[int, int, int, int], dict[str, float | str]] = {}

                for det in self._detect_plate_candidates(frame):
                    int_coords = [int(c) for c in det["box"]]
                    detected_boxes.append(int_coords)
                    box_meta[tuple(int_coords)] = {
                        "conf": float(det["conf"]),
                        "cls_name": str(det["cls_name"]),
                    }

                tracked = self._plate_tracker.update(detected_boxes)
                for x1, y1, x2, y2, track_id in tracked:
                    meta = box_meta.get(
                        (x1, y1, x2, y2),
                        {"conf": 0.5, "cls_name": "license_plate"},
                    )
                    plate_detections.append({
                        "box": [x1, y1, x2, y2],
                        "conf": float(meta["conf"]),
                        "cls_name": str(meta["cls_name"]),
                        "track_id": track_id,
                    })

            # --- Draw bounding boxes and update the live stream ---
            if plate_detections:
                annotated = _draw_boxes(frame, plate_detections)
                with frame_locks[self.camera_name]:
                    latest_frames[self.camera_name] = annotated

            # --- Step 3: For each tracked plate, capture once per ID ---
            now = time.time()
            for det in plate_detections:
                track_id = int(det["track_id"])
                last_time = self._captured_ids.get(track_id, 0.0)
                if (now - last_time) >= CAPTURE_COOLDOWN:
                    # Merge nearby/overlapping detections into one input batch.
                    if not self._claim_batch_input(det["box"], now):
                        self._captured_ids[track_id] = now
                        continue

                    self._captured_ids[track_id] = now
                    frame_snapshot = frame.copy()
                    threading.Thread(
                        target=self._process_plate,
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
    def _crop_detection(self, frame: np.ndarray, detection: dict) -> np.ndarray | None:
        """Crop a detected region from the frame. Returns None if empty."""
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    @staticmethod
    def _box_iou(box_a: list[int] | tuple[int, int, int, int],
                 box_b: list[int] | tuple[int, int, int, int]) -> float:
        """Return IoU overlap between two boxes."""
        ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
        bx1, by1, bx2, by2 = [int(v) for v in box_b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / float(union)

    @staticmethod
    def _box_center_distance(box_a: list[int] | tuple[int, int, int, int],
                             box_b: list[int] | tuple[int, int, int, int]) -> float:
        """Return Euclidean distance between box centers."""
        ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
        bx1, by1, bx2, by2 = [int(v) for v in box_b]
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        return float(np.hypot(acx - bcx, acy - bcy))

    def _claim_batch_input(self, box: list[int], now: float) -> bool:
        """
        Register a detection as a new input batch if it's not a recent duplicate.
        Returns True when this detection should be processed, else False.
        """
        self._recent_batch_regions = [
            item for item in self._recent_batch_regions
            if (now - float(item["time"])) < CAPTURE_COOLDOWN
        ]

        for item in self._recent_batch_regions:
            prev_box = item["box"]
            iou = self._box_iou(box, prev_box)
            dist = self._box_center_distance(box, prev_box)
            if iou >= BATCH_MERGE_IOU or dist <= BATCH_CENTER_DISTANCE:
                return False

        self._recent_batch_regions.append({"box": [int(v) for v in box], "time": now})
        return True

    @staticmethod
    def _expand_box_coords(
        box: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        pad_ratio: float,
    ) -> tuple[int, int, int, int]:
        """Expand a box by ratio and clamp it to frame bounds."""
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_y)
        return x1, y1, x2, y2

    def _detect_vehicle_rois(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect vehicles in the frame and return expanded ROIs."""
        h, w = frame.shape[:2]
        results = self.vehicle_model(
            frame,
            classes=VEHICLE_CLASSES,
            conf=VEHICLE_CONF,
            verbose=False,
        )[0]

        rois: list[tuple[int, int, int, int]] = []
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_VEHICLE_AREA:
                continue
            rois.append(self._expand_box_coords((x1, y1, x2, y2), w, h, VEHICLE_ROI_PAD))

        return rois

    def _detect_plate_candidates(
        self,
        frame: np.ndarray,
        roi_hint: tuple[int, int, int, int] | None = None,
    ) -> list[dict]:
        """Detect plate candidates using vehicle-guided ROIs."""
        h, w = frame.shape[:2]
        if roi_hint is not None:
            rois = [self._expand_box_coords(roi_hint, w, h, 0.2)]
        else:
            rois = self._detect_vehicle_rois(frame)
            if not rois:
                # Fallback keeps capture working when Stage 1 misses a vehicle.
                rois = [(0, 0, w, h)]

        detections: list[dict] = []
        for rx1, ry1, rx2, ry2 in rois:
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            result = self.plate_model(roi, conf=YOLO_CONF, verbose=False)[0]
            names = result.names

            for box in result.boxes:
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                gx1 = rx1 + bx1
                gy1 = ry1 + by1
                gx2 = rx1 + bx2
                gy2 = ry1 + by2
                area = (gx2 - gx1) * (gy2 - gy1)
                if area < MIN_PLATE_AREA:
                    continue

                cls_id = int(box.cls[0]) if box.cls is not None else -1
                if isinstance(names, dict):
                    cls_name = names.get(cls_id, "license_plate")
                elif isinstance(names, list) and 0 <= cls_id < len(names):
                    cls_name = names[cls_id]
                else:
                    cls_name = "license_plate"

                detections.append({
                    "box": [gx1, gy1, gx2, gy2],
                    "conf": float(box.conf[0]),
                    "cls_name": str(cls_name),
                })

        return detections

    def _process_plate(self, first_frame: np.ndarray, detection: dict):
        """
        Burst-capture ~5 crops of the same plate over ~1 s, then run
        batch consensus OCR across all crops for maximum accuracy.
        """
        # Collect plate crops and run consensus OCR over the burst.
        plate_crops: list[np.ndarray] = []

        # First crop from the triggering frame
        crop = self._crop_detection(first_frame, detection)
        if crop is not None:
            plate_crops.append(crop)

        base_box = tuple(int(v) for v in detection["box"])

        # Collect more crops from subsequent live frames
        for _ in range(BURST_FRAMES - 1):
            time.sleep(BURST_INTERVAL)
            with self._raw_lock:
                frame = self._raw_frame
            if frame is None:
                continue
            # Re-run 2-stage detection focused around the original plate region.
            candidates = self._detect_plate_candidates(frame, roi_hint=base_box)
            best_det = max(candidates, key=lambda d: d["conf"], default=None)
            if best_det is not None:
                c = self._crop_detection(frame, best_det)
                if c is not None:
                    plate_crops.append(c)

        if not plate_crops:
            return

        ocr_raw, confidence = recognise_plate_batch(plate_crops)
        ocr_corrected, plate_valid, _ = correct_ph_plate(ocr_raw)

        registered = get_registered_plates()
        matched_plate, match_score, match_status = match_plate(ocr_corrected, registered)

        if match_status == "AUTO_MATCHED" and matched_plate:
            final_plate = matched_plate
        elif ocr_corrected and ocr_corrected != "UNKNOWN":
            final_plate = ocr_corrected
        else:
            final_plate = ocr_raw if ocr_raw else "UNKNOWN"

        safe_plate = "".join(ch for ch in final_plate if ch.isalnum()) or "manual"

        # Build filename and save
        now = datetime.now()
        ts_file = now.strftime("%Y-%m-%d_%H-%M-%S")
        ts_db = now.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{ts_file}_{self.camera_name}_{safe_plate}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)

        image_to_save = plate_crops[0]
        cv2.imwrite(filepath, image_to_save)

        # Optional debug copy for bug testing of plate crops.
        save_debug_plate_image(
            plate_image=plate_crops[0],
            camera_name=self.camera_name,
            plate_text=final_plate,
            confidence=confidence,
        )

        rel_path = f"captures/{filename}"

        if final_plate == "UNKNOWN":
            manual_id = enqueue_manual_input(
                camera=self.camera_name,
                timestamp=ts_db,
                image_path=rel_path,
                confidence=confidence,
                ocr_raw=ocr_raw,
                ocr_corrected=ocr_corrected,
                match_status=match_status,
            )
            print(
                f"[{self.camera_name}] Manual entry required "
                f"(queue_id={manual_id}, raw={ocr_raw}, conf={confidence:.2f})"
            )
            return

        insert_detection(
            plate_number=final_plate,
            camera=self.camera_name,
            timestamp=ts_db,
            image_path=rel_path,
            confidence=confidence,
            ocr_raw=ocr_raw,
            ocr_corrected=ocr_corrected,
            plate_valid=plate_valid,
            matched_plate=matched_plate,
            match_score=match_score,
            match_status=match_status,
        )
        print(
            f"[{self.camera_name}] Detected: {final_plate} "
            f"(raw={ocr_raw}, conf={confidence:.2f}, match={match_status}, "
            f"score={match_score:.2f}, {len(plate_crops)} plate crops) -> {filename}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers to start / stop the whole camera system
# ---------------------------------------------------------------------------
_processors: list[CameraProcessor] = []

# Load YOLO models once, shared across camera threads.
_plate_model = None
_vehicle_model = None


def _resolve_vehicle_model_path() -> str:
    """Pick the first available vehicle model path, else default to yolov8n.pt."""
    for candidate in VEHICLE_MODEL_CANDIDATES:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        if not os.path.isabs(candidate):
            return candidate
    return "yolov8n.pt"


def _get_plate_model():
    global _plate_model
    if _plate_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "license_plate_detector.pt not found. "
                "Run 'python download_models.py' first."
            )
        print(f"[system] Loading plate detector from {MODEL_PATH} ...")
        _plate_model = YOLO(MODEL_PATH)
        print("[system] Plate detector ready.")
    return _plate_model


def _get_vehicle_model():
    global _vehicle_model
    if _vehicle_model is None:
        vehicle_path = _resolve_vehicle_model_path()
        print(f"[system] Loading vehicle detector from {vehicle_path} ...")
        _vehicle_model = YOLO(vehicle_path)
        print("[system] Vehicle detector ready.")
    return _vehicle_model


def start_cameras():
    """
    Load YOLO models but do NOT start any cameras automatically.
    Users assign cameras from the dashboard.
    """
    _get_plate_model()
    _get_vehicle_model()
    print(
        "[system] Batch merge config: "
        f"BATCH_MERGE_IOU={BATCH_MERGE_IOU:.2f}, "
        f"BATCH_CENTER_DISTANCE={BATCH_CENTER_DISTANCE:.1f}px"
    )
    print("[system] Ready — assign cameras from the dashboard.")


def stop_cameras():
    """Stop all running camera threads."""
    for p in _processors:
        p.stop()
    _processors.clear()


def _device_label(idx: int) -> str:
    """Return a user-facing label for a camera device index."""
    if HAS_V4L2_SYSFS:
        return f"/dev/video{idx}"
    return f"camera index {idx}"


def _capture_backend_candidates() -> list[tuple[int | None, str]]:
    """Return preferred OpenCV backend candidates per OS."""
    candidates: list[tuple[int | None, str]] = []
    if os.name == "nt":
        candidates.extend([
            (getattr(cv2, "CAP_DSHOW", None), "CAP_DSHOW"),
            (getattr(cv2, "CAP_MSMF", None), "CAP_MSMF"),
        ])
    elif HAS_V4L2_SYSFS:
        candidates.append((getattr(cv2, "CAP_V4L2", None), "CAP_V4L2"))

    # Final fallback lets OpenCV auto-select a backend.
    candidates.append((None, "default"))

    # De-duplicate when a backend constant is unavailable.
    unique: list[tuple[int | None, str]] = []
    seen: set[int | None] = set()
    for backend, name in candidates:
        if backend in seen:
            continue
        seen.add(backend)
        unique.append((backend, name))
    return unique


def _open_video_capture(device_index: int) -> tuple[cv2.VideoCapture | None, str]:
    """Try opening a camera index with OS-appropriate OpenCV backends."""
    for backend, name in _capture_backend_candidates():
        cap = cv2.VideoCapture(device_index) if backend is None \
            else cv2.VideoCapture(device_index, backend)
        if cap.isOpened():
            return cap, name
        cap.release()
    return None, "unavailable"


def _can_read_frame(cap: cv2.VideoCapture, attempts: int = 5) -> bool:
    """Warm up and verify the capture can return at least one frame."""
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True
        time.sleep(0.05)
    return False


def _get_device_name(idx: int) -> str:
    """Read the human-readable device name from sysfs, or fall back."""
    if not HAS_V4L2_SYSFS:
        return f"Camera {idx}"

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
    if not HAS_V4L2_SYSFS:
        return True

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
    if HAS_V4L2_SYSFS:
        indices: list[int] = []
        for path in sorted(glob.glob("/sys/class/video4linux/video*")):
            name = os.path.basename(path)
            try:
                idx = int(name.replace("video", ""))
                indices.append(idx)
            except ValueError:
                continue
        return indices

    # Non-Linux fallback: probe a small index range (0..9).
    return list(range(GENERIC_PROBE_COUNT))


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

        cap, _ = _open_video_capture(idx)
        if cap is not None:
            ret = _can_read_frame(cap)
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
    plate_model = _get_plate_model()
    vehicle_model = _get_vehicle_model()
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
        plate_model=plate_model,
        vehicle_model=vehicle_model,
    )
    new_proc.start()
    _processors.append(new_proc)
    print(f"[system] {camera_name} camera started on {_device_label(new_device_index)}")


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
