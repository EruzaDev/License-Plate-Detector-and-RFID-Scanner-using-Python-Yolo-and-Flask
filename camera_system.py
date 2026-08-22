"""
camera_system.py - Fast-ALPR sole plate detection & RFID tap-only capture pipeline.

Architecture:
    1. CameraProcessor handles live camera frame grabbing for stream previews (no background detection loop).
    2. Plate detection & OCR relies solely on Fast-ALPR (ankandrew/fast-alpr ONNX Runtime engine).
    3. Detection and capture trigger exclusively on RFID card taps or manual dashboard scans.
"""

import os
import time
import threading
import glob
from datetime import datetime

import cv2
import numpy as np

from ocr_processor import recognise_plate, recognise_plate_batch, correct_ph_plate, match_plate, get_fast_alpr, normalize_plate_text
from database import (
    insert_detection,
    get_registered_plates,
    get_registered_plate_record,
    suggest_plate_from_feedback,
    enqueue_manual_input,
    find_matching_entrance_detection,
    is_recent_duplicate_exit,
    is_recent_duplicate_entrance,
    mark_entrance_departed,
)
from sound_system import play_sound, play_gate_success

try:
    from plate_debug_saver import save_debug_plate_image
except Exception:
    def save_debug_plate_image(*_args, **_kwargs):
        return None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared state — accessed by Flask for MJPEG streaming
# ---------------------------------------------------------------------------
latest_frames = {"entrance": None, "exit": None}
frame_locks = {"entrance": threading.Lock(), "exit": threading.Lock()}
HAS_V4L2_SYSFS = os.path.isdir("/sys/class/video4linux")
GENERIC_PROBE_COUNT = 10


class CameraProcessor:
    """
    Captures live camera frames from a device index for stream preview and on-demand tap processing.
    Runs a lightweight thread that grabs frames without continuous background inference.
    """

    def __init__(self, device_index: int, camera_name: str):
        self.device_index = device_index
        self.camera_name = camera_name  # "entrance" or "exit"
        self.running = False
        self._capture_thread = None

    def start(self):
        """Start the live capture thread."""
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """Signal thread to stop."""
        self.running = False

    def _capture_loop(self):
        cap, backend_name = _open_video_capture(self.device_index)
        if cap is None:
            print(f"[{self.camera_name}] ERROR: cannot open {_device_label(self.device_index)}")
            return

        if backend_name == "CAP_V4L2":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"[{self.camera_name}] Camera started on {_device_label(self.device_index)} ({backend_name})")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            with frame_locks[self.camera_name]:
                latest_frames[self.camera_name] = frame

        cap.release()
        print(f"[{self.camera_name}] Camera stopped.")


# ---------------------------------------------------------------------------
# Module-level helpers to start / stop the whole camera system
# ---------------------------------------------------------------------------
_processors: list[CameraProcessor] = []


def start_cameras():
    """
    Initialize Fast-ALPR system on application startup.
    Cameras are assigned and started on demand from the dashboard.
    """
    get_fast_alpr()
    print("[system] Fast-ALPR sole detection engine ready. Waiting for RFID card taps.")


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

    candidates.append((None, "default"))

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
    """Read human-readable device name from sysfs, or fall back."""
    if not HAS_V4L2_SYSFS:
        return f"Camera {idx}"

    name_path = f"/sys/class/video4linux/video{idx}/name"
    try:
        with open(name_path) as f:
            return f.read().strip()
    except OSError:
        return f"Camera {idx}"


def _is_capture_device(idx: int) -> bool:
    """Check if /dev/videoN is a real video-capture device."""
    if not HAS_V4L2_SYSFS:
        return True

    index_path = f"/sys/class/video4linux/video{idx}/index"
    try:
        with open(index_path) as f:
            return f.read().strip() == "0"
    except OSError:
        pass

    return True


def _find_video_indices() -> list[int]:
    """Scan /sys/class/video4linux/ to find existing video device indices."""
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

    return list(range(GENERIC_PROBE_COUNT))


def list_video_devices() -> list[dict]:
    """List available video-capture devices."""
    in_use: dict[int, str] = {}
    for p in _processors:
        in_use[p.device_index] = p.camera_name

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
    for i, p in enumerate(_processors):
        if p.camera_name == camera_name:
            p.stop()
            time.sleep(0.5)
            _processors.pop(i)
            break

    with frame_locks[camera_name]:
        latest_frames[camera_name] = None

    new_proc = CameraProcessor(
        device_index=new_device_index,
        camera_name=camera_name,
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


def _scan_plate_from_frame(frame: np.ndarray) -> dict:
    """Run Fast-ALPR plate detection and OCR on a raw frame."""
    if frame is None or frame.size == 0:
        raise RuntimeError("Unable to read a valid frame from camera.")

    alpr = get_fast_alpr()
    if alpr is None:
        raise RuntimeError("Fast-ALPR engine is unavailable.")

    results = alpr.predict(frame)
    if not results:
        raise ValueError("No license plate detected by Fast-ALPR. Retry scan or reposition vehicle.")

    def get_conf(r):
        if r.ocr and isinstance(r.ocr.confidence, list) and r.ocr.confidence:
            return sum(r.ocr.confidence) / len(r.ocr.confidence)
        if r.ocr and isinstance(r.ocr.confidence, (int, float)):
            return float(r.ocr.confidence)
        return float(r.detection.confidence)

    best_res = max(results, key=get_conf)
    ocr_raw = "UNKNOWN"
    confidence = float(best_res.detection.confidence)

    if best_res.ocr and best_res.ocr.text:
        ocr_raw = normalize_plate_text(best_res.ocr.text)
        if isinstance(best_res.ocr.confidence, list) and best_res.ocr.confidence:
            confidence = float(sum(best_res.ocr.confidence) / len(best_res.ocr.confidence))
        elif isinstance(best_res.ocr.confidence, (int, float)):
            confidence = float(best_res.ocr.confidence)

    b = best_res.detection.bounding_box
    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)

    ocr_corrected, plate_valid, fmt = correct_ph_plate(ocr_raw)
    feedback_plate, feedback_score, feedback_uses = suggest_plate_from_feedback(ocr_raw)

    if feedback_plate:
        final_plate = feedback_plate
        plate_valid = True
    elif ocr_corrected and ocr_corrected != "UNKNOWN":
        final_plate = ocr_corrected
    else:
        final_plate = ocr_raw if ocr_raw else "UNKNOWN"

    return {
        "raw_plate": ocr_raw,
        "corrected_plate": ocr_corrected,
        "final_plate": final_plate,
        "confidence": confidence,
        "plate_valid": bool(plate_valid),
        "format": fmt,
        "feedback_score": float(feedback_score),
        "feedback_uses": int(feedback_uses),
        "bbox": [x1, y1, x2, y2],
    }


def scan_plate_once(camera_name: str) -> dict:
    """
    Perform a single Fast-ALPR OCR plate scan from the latest frame for a camera.
    Returns a result dict with raw/corrected/final plate and confidence.
    """
    camera = str(camera_name or "").lower()
    if camera not in {"entrance", "exit"}:
        raise ValueError("Invalid camera name.")

    with frame_locks[camera]:
        frame = latest_frames[camera].copy() if latest_frames[camera] is not None else None

    if frame is None:
        raise RuntimeError("No frame available. Start and warm up the selected camera first.")

    res = _scan_plate_from_frame(frame)
    res["camera"] = camera
    return res


def trigger_instant_capture(camera_name: str, rfid_uid: str | None = None) -> dict | None:
    """
    Instantly capture a live frame from the specified camera gate ('entrance' or 'exit')
    on an RFID tap, detect license plate and OCR via Fast-ALPR, store capture image and
    database record, and verify against the scanned RFID UID.
    """
    clean_cam = str(camera_name or "").strip().lower()
    if clean_cam not in ("entrance", "exit"):
        return None

    with frame_locks[clean_cam]:
        frame = latest_frames[clean_cam].copy() if latest_frames[clean_cam] is not None else None

    if frame is None:
        print(f"[{clean_cam}] Instant capture requested, but no active live frame available.")
        return None

    ocr_raw = "UNKNOWN"
    confidence = 0.0
    crop = None
    bbox = None

    alpr = get_fast_alpr()
    if alpr is not None:
        try:
            results = alpr.predict(frame)
            if results:
                def get_conf(r):
                    if r.ocr and isinstance(r.ocr.confidence, list) and r.ocr.confidence:
                        return sum(r.ocr.confidence) / len(r.ocr.confidence)
                    if r.ocr and isinstance(r.ocr.confidence, (int, float)):
                        return float(r.ocr.confidence)
                    return float(r.detection.confidence)

                best_res = max(results, key=get_conf)
                if best_res.ocr and best_res.ocr.text:
                    ocr_raw = normalize_plate_text(best_res.ocr.text)
                    if isinstance(best_res.ocr.confidence, list) and best_res.ocr.confidence:
                        confidence = float(sum(best_res.ocr.confidence) / len(best_res.ocr.confidence))
                    elif isinstance(best_res.ocr.confidence, (int, float)):
                        confidence = float(best_res.ocr.confidence)
                    else:
                        confidence = float(best_res.detection.confidence)

                b = best_res.detection.bounding_box
                bbox = (int(b.x1), int(b.y1), int(b.x2), int(b.y2))
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    crop = frame[y1:y2, x1:x2]
        except Exception as err:
            print(f"[{clean_cam}] Fast-ALPR error on instant capture: {err}")

    if crop is None or crop.size == 0:
        crop = frame

    ocr_corrected, plate_valid, _ = correct_ph_plate(ocr_raw)
    feedback_plate, feedback_score, feedback_uses = suggest_plate_from_feedback(ocr_raw)
    if feedback_plate:
        ocr_corrected = feedback_plate
        plate_valid = True

    entry_detection_id = None
    matched_entrance = None
    entrance_score = 0.0

    if clean_cam == "exit":
        matched_entrance, entrance_score = find_matching_entrance_detection(
            exit_candidate=ocr_corrected or ocr_raw,
            max_age_minutes=1440,
            min_similarity=65.0,
        )

    if matched_entrance is not None:
        final_plate = matched_entrance["plate_number"]
        matched_plate = matched_entrance["plate_number"]
        match_score = entrance_score
        match_status = "EXIT_MATCHED"
        entry_detection_id = int(matched_entrance["id"])
        plate_valid = True
        mark_entrance_departed(entry_detection_id)
    else:
        registered = get_registered_plates()
        matched_plate, match_score, match_status = match_plate(ocr_corrected, registered)
        if match_status == "AUTO_MATCHED" and matched_plate:
            final_plate = matched_plate
        elif ocr_corrected and ocr_corrected != "UNKNOWN":
            final_plate = ocr_corrected
        else:
            final_plate = ocr_raw if ocr_raw else "UNKNOWN"

    safe_plate = "".join(ch for ch in final_plate if ch.isalnum()) or "manual"
    now = datetime.now()
    ts_file = now.strftime("%Y-%m-%d_%H-%M-%S")
    ts_db = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"{ts_file}_{clean_cam}_rfid_{safe_plate}.jpg"
    filepath = os.path.join(CAPTURES_DIR, filename)
    cv2.imwrite(filepath, crop)
    rel_path = f"captures/{filename}"

    registration = get_registered_plate_record(final_plate)
    expected_rfid_uid = registration.get("rfid_uid") if registration else None
    access_granted = False

    if rfid_uid:
        scanned_clean = str(rfid_uid).strip().upper()
        from database import get_plates_by_rfid_uid, verify_rfid_plate_match

        registered_vehicles = get_plates_by_rfid_uid(scanned_clean)
        if registered_vehicles:
            expected_rfid_uid = scanned_clean
            is_match, matched_reg_plate = verify_rfid_plate_match(scanned_clean, final_plate)
            if not is_match and ocr_raw != final_plate:
                is_match, matched_reg_plate = verify_rfid_plate_match(scanned_clean, ocr_raw)

            if is_match:
                rfid_status = "MATCH"
                access_granted = True
                if matched_reg_plate:
                    matched_plate = matched_reg_plate
                    match_status = "EXACT_MATCH"
            else:
                rfid_status = "MISMATCH"
                access_granted = False
                match_status = "MISMATCH"
        else:
            rfid_status = "UNREGISTERED_TAG"
            access_granted = False
            match_status = "NO_MATCH"
    else:
        rfid_status = "NOT_SCANNED" if expected_rfid_uid else "NOT_REQUIRED"
        if match_status in ("EXACT_MATCH", "FUZZY_MATCH", "EXIT_MATCHED", "AUTO_MATCHED") or registration is not None:
            access_granted = True

    detection_id = insert_detection(
        plate_number=final_plate,
        camera=clean_cam,
        timestamp=ts_db,
        image_path=rel_path,
        confidence=confidence,
        ocr_raw=ocr_raw,
        ocr_corrected=ocr_corrected,
        plate_valid=plate_valid,
        matched_plate=matched_plate,
        match_score=match_score,
        match_status=match_status,
        rfid_status=rfid_status,
        expected_rfid_uid=expected_rfid_uid,
        scanned_rfid_uid=rfid_uid,
        rfid_verified_at=ts_db if rfid_uid else None,
        trip_status="ACTIVE" if clean_cam == "entrance" else "EXITED",
        entry_detection_id=entry_detection_id,
    )

    if access_granted:
        play_gate_success(clean_cam)
    else:
        play_sound("denied", gate=clean_cam)

    print(
        f"[{clean_cam}] RFID tap instant capture #{detection_id}: plate={final_plate} "
        f"(raw={ocr_raw}, rfid={rfid_uid}, status={rfid_status}, granted={access_granted})"
    )
    return {
        "detection_id": detection_id,
        "camera": clean_cam,
        "plate_number": final_plate,
        "raw_plate": ocr_raw,
        "confidence": confidence,
        "image_path": rel_path,
        "rfid_status": rfid_status,
        "rfid_uid": rfid_uid,
        "access_granted": access_granted,
    }


def scan_plate_once_from_device(device_index: int) -> dict:
    """One-shot plate scan using a physical camera index."""
    if not isinstance(device_index, int) or device_index < 0:
        raise ValueError("Invalid device index.")

    cap, backend_name = _open_video_capture(device_index)
    if cap is None:
        raise RuntimeError(f"Cannot open {_device_label(device_index)}")

    try:
        if backend_name == "CAP_V4L2":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame = None
        for _ in range(8):
            ret, candidate = cap.read()
            if ret and candidate is not None and candidate.size > 0:
                frame = candidate
            time.sleep(0.03)

        res = _scan_plate_from_frame(frame)
        res.update({
            "device_index": int(device_index),
            "backend": backend_name,
        })
        return res
    finally:
        cap.release()


def start_device_mjpeg_stream(device_index: int):
    """Start an MJPEG stream directly from a physical camera index for setup preview."""
    if not isinstance(device_index, int) or device_index < 0:
        raise ValueError("Invalid device index.")

    cap, backend_name = _open_video_capture(device_index)
    if cap is None:
        raise RuntimeError(f"Cannot open {_device_label(device_index)}")

    if backend_name == "CAP_V4L2":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _generator():
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.05)
                    continue

                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    time.sleep(0.03)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
                time.sleep(0.066)
        finally:
            cap.release()

    return _generator()
