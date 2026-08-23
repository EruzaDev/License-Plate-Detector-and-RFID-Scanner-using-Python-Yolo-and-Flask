"""
rfid_system.py — Pure Python USB HID RFID scanner discovery, assignment, and reading.

Works on standard Linux / Raspberry Pi without requiring C-compiled external libraries.
Discovers USB HID RFID scanners (e.g. "IC Reader IC Reader", Vendor ffff:0035) via /sys/class/input,
reads keystroke input packets directly via /dev/input/event*, and triggers auto-verification
specifically separated for Entrance vs Exit gates.
"""

from __future__ import annotations

import glob
import os
import re
import struct
import threading
import time
from datetime import datetime
from typing import Any, Callable

from database import (
    verify_detection_rfid,
    get_pending_rfid_verifications,
    is_rfid_registered,
)
from sound_system import play_sound, play_gate_success

# ---------------------------------------------------------------------------
# Linux Input Event Definitions (linux/input.h & linux/input-event-codes.h)
# ---------------------------------------------------------------------------
EV_KEY = 1

# Standard Linux keycodes to ASCII character mapping
KEY_MAP: dict[int, str] = {
    # Top number row
    2: "1", 3: "2", 4: "3", 5: "4", 6: "5", 7: "6", 8: "7", 9: "8", 10: "9", 11: "0",
    12: "-", 13: "=",
    # Top alphabet row
    16: "Q", 17: "W", 18: "E", 19: "R", 20: "T", 21: "Y", 22: "U", 23: "I", 24: "O", 25: "P",
    # Home alphabet row
    30: "A", 31: "S", 32: "D", 33: "F", 34: "G", 35: "H", 36: "J", 37: "K", 38: "L",
    # Bottom alphabet row
    44: "Z", 45: "X", 46: "C", 47: "V", 48: "B", 49: "N", 50: "M",
    # Numpad digits
    79: "1", 80: "2", 81: "3", 75: "4", 76: "5", 77: "6", 71: "7", 72: "8", 73: "9", 82: "0",
    # Punctuation & Space
    57: " ", 51: ",", 52: ".", 53: "/", 74: "-", 78: "+", 55: "*",
}

ENTER_CODES = {28, 96}  # KEY_ENTER (28) and KEY_KPENTER (96)

# Names of internal/system keyboard hardware to ignore
SYSTEM_KEYBOARD_PATTERNS = [
    re.compile(r"AT Translated Set", re.I),
    re.compile(r"Power Button", re.I),
    re.compile(r"Lid Switch", re.I),
    re.compile(r"Video Bus", re.I),
    re.compile(r"Sleep Button", re.I),
    re.compile(r"PC Speaker", re.I),
    re.compile(r"gpio", re.I),
    re.compile(r"Wireless Radio", re.I),
    re.compile(r"Touchpad", re.I),
    re.compile(r"HD-Audio|HDA NVidia|Headphone|soundcore|ALSA", re.I),
]


def _is_system_keyboard(name: str, phys: str) -> bool:
    """Check if device is an internal motherboard/system peripheral."""
    for pat in SYSTEM_KEYBOARD_PATTERNS:
        if pat.search(name):
            return True
    if phys and ("isa0060" in phys or "i2c-" in phys):
        return True
    return False


def _read_sysfs_attr(path: str) -> str:
    """Helper to safely read a single sysfs string file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Public API: Discover RFID devices from sysfs
# ---------------------------------------------------------------------------
def list_rfid_devices() -> list[dict[str, Any]]:
    """
    List USB HID devices that look like RFID scanners.
    Discovers via /sys/class/input (no root required for discovery).
    """
    in_use: dict[str, str] = {}
    for camera_name, reader in _readers.items():
        if reader and reader.event_path:
            in_use[reader.event_path] = camera_name

    devices: list[dict[str, Any]] = []

    for event_dir in sorted(glob.glob("/sys/class/input/event*")):
        ev_name = os.path.basename(event_dir)
        dev_path = f"/dev/input/{ev_name}"

        name = _read_sysfs_attr(os.path.join(event_dir, "device", "name"))
        phys = _read_sysfs_attr(os.path.join(event_dir, "device", "phys"))
        uniq = _read_sysfs_attr(os.path.join(event_dir, "device", "uniq"))
        modalias = _read_sysfs_attr(os.path.join(event_dir, "device", "modalias"))

        if not name:
            continue

        if _is_system_keyboard(name, phys):
            continue

        vendor = ""
        product = ""
        if modalias:
            mv = re.search(r"v([0-9A-Fa-f]{4})", modalias)
            mp = re.search(r"p([0-9A-Fa-f]{4})", modalias)
            if mv:
                vendor = f"0x{mv.group(1).lower()}"
            if mp:
                product = f"0x{mp.group(1).lower()}"

        # Test if user has read permission on /dev/input/event*
        accessible = os.access(dev_path, os.R_OK)

        # Build readable label
        display_label = name
        if phys:
            usb_match = re.search(r"usb-[^/]+", phys)
            if usb_match:
                display_label += f" [{usb_match.group(0)}]"

        devices.append({
            "event_path": dev_path,
            "name": display_label,
            "raw_name": name,
            "phys": phys,
            "uniq": uniq,
            "vendor": vendor,
            "product": product,
            "accessible": accessible,
            "in_use_by": in_use.get(dev_path),
        })

    return devices


# ---------------------------------------------------------------------------
# RFIDReader — Pure Python background thread reading /dev/input/event*
# ---------------------------------------------------------------------------
class RFIDReader:
    """
    Pure Python reader that reads Linux input events from /dev/input/event*.
    Extracts keystroke codes, buffers characters, and emits complete UIDs on Enter.
    """

    def __init__(
        self,
        event_path: str,
        camera_name: str,
        on_scan: Callable[[str, str], None] | None = None,
    ):
        self.event_path = event_path
        self.camera_name = camera_name  # "entrance" or "exit"
        self._on_scan = on_scan
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_uid: str | None = None
        self._last_scan_at: str | None = None
        self._last_result: str | None = None
        self._error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"rfid-reader-{self.camera_name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def last_scan(self) -> dict[str, Any]:
        with self._lock:
            return {
                "camera": self.camera_name,
                "event_path": self.event_path,
                "running": self.running,
                "last_uid": self._last_uid,
                "last_scan_at": self._last_scan_at,
                "last_result": self._last_result,
                "error": self._error,
            }

    def _read_loop(self) -> None:
        is_64bit = struct.calcsize("P") == 8
        event_struct_fmt = "qqHHi" if is_64bit else "iiHHi"
        event_size = struct.calcsize(event_struct_fmt)
        EVIOCGRAB = 0x40044590  # Linux exclusive grab ioctl

        fd = None
        while not self._stop_event.is_set():
            try:
                fd = os.open(self.event_path, os.O_RDONLY | os.O_NONBLOCK)
                # Try exclusive grab so keystrokes don't leak to desktop / other windows
                try:
                    import fcntl
                    fcntl.ioctl(fd, EVIOCGRAB, 1)
                    print(f"[rfid-{self.camera_name}] Exclusively grabbed {self.event_path}")
                except Exception as grab_err:
                    print(f"[rfid-{self.camera_name}] Note: Non-exclusive grab ({grab_err})")

                with self._lock:
                    self._error = None
                print(f"[rfid-{self.camera_name}] Opened {self.event_path} for reading.")
                break
            except PermissionError:
                with self._lock:
                    self._error = f"Permission denied for {self.event_path}. Run: sudo chmod 666 {self.event_path} or add user to 'input' group."
                print(f"[rfid-{self.camera_name}] {self._error}")
                if self._stop_event.wait(5.0):
                    return
            except OSError as e:
                with self._lock:
                    self._error = f"Device error: {e}"
                print(f"[rfid-{self.camera_name}] Cannot open {self.event_path}: {e}")
                if self._stop_event.wait(5.0):
                    return

        if fd is None:
            return

        buffer: list[str] = []
        try:
            import select
            while not self._stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.25)
                if not r:
                    continue

                try:
                    raw_data = os.read(fd, event_size * 16)
                except BlockingIOError:
                    continue
                except OSError as err:
                    if self._stop_event.is_set():
                        break
                    raise err

                if not raw_data:
                    time.sleep(0.05)
                    continue

                # Process chunk in units of event_size
                for offset in range(0, len(raw_data) - event_size + 1, event_size):
                    chunk = raw_data[offset : offset + event_size]
                    _, _, ev_type, ev_code, ev_val = struct.unpack(event_struct_fmt, chunk)

                    # Process key-down events (ev_val == 1)
                    if ev_type == EV_KEY and ev_val == 1:
                        if ev_code in ENTER_CODES:
                            uid = "".join(buffer).strip().upper()
                            buffer.clear()
                            if uid:
                                self._handle_scan(uid)
                        elif ev_code in KEY_MAP:
                            buffer.append(KEY_MAP[ev_code])
        except Exception as exc:
            with self._lock:
                self._error = f"Read error: {exc}"
            print(f"[rfid-{self.camera_name}] Reading error on {self.event_path}: {exc}")
        finally:
            try:
                if fd is not None:
                    try:
                        import fcntl
                        fcntl.ioctl(fd, EVIOCGRAB, 0)
                    except Exception:
                        pass
                    os.close(fd)
            except Exception:
                pass
            print(f"[rfid-{self.camera_name}] Reader loop ended.")

    def _handle_scan(self, uid: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._last_uid = uid
            self._last_scan_at = now
            self._last_result = "SCANNED"

        _update_global_scan(self.camera_name, uid)
        print(f"[rfid-{self.camera_name}] Scanned UID '{uid}' from {self.event_path}")

        if self._on_scan:
            try:
                self._on_scan(self.camera_name, uid)
            except Exception as exc:
                with self._lock:
                    self._last_result = f"CALLBACK_ERROR: {exc}"
                print(f"[rfid-{self.camera_name}] Verification callback error: {exc}")


# ---------------------------------------------------------------------------
# Module-level State: Active readers per camera gate & latest global scan
# ---------------------------------------------------------------------------
_readers: dict[str, RFIDReader | None] = {"entrance": None, "exit": None}
_latest_global_scan: dict[str, Any] = {
    "uid": None,
    "camera": None,
    "timestamp": None,
    "time_epoch": 0.0,
}


def _update_global_scan(camera_name: str, uid: str) -> None:
    global _latest_global_scan
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _latest_global_scan = {
        "uid": uid,
        "camera": camera_name,
        "timestamp": now,
        "time_epoch": time.time(),
    }


def _default_on_scan(camera_name: str, scanned_uid: str) -> None:
    """
    On RFID card scan:
    1. Instantly trigger a camera frame capture and plate OCR for this gate.
    2. Fall back to auto-verifying pending DB detections if camera snapshot is unavailable.
    """
    clean_gate = str(camera_name or "").strip().lower()

    # Step 1: Try instant live camera capture
    try:
        from camera_system import trigger_instant_capture
        res = trigger_instant_capture(camera_name=clean_gate, rfid_uid=scanned_uid)
        if res is not None:
            reader = _readers.get(clean_gate)
            if reader:
                with reader._lock:
                    reader._last_result = "ACCESS_GRANTED" if res.get("access_granted") else "DENIED"
            print(f"[rfid-{clean_gate}] Instant camera capture executed for UID {scanned_uid}: plate={res.get('plate_number')}")
            return
    except Exception as exc:
        print(f"[rfid-{clean_gate}] Instant capture error ({exc}); falling back to pending DB verification.")

    # Step 2: Fallback to pending DB detections
    try:
        pending = get_pending_rfid_verifications(limit=50)
    except Exception as exc:
        print(f"[rfid-{clean_gate}] Database query error: {exc}")
        return

    # Find the most recent unverified detection specifically for this gate
    target = None
    for item in pending:
        if str(item.get("camera", "")).lower() == clean_gate:
            target = item
            break

    if target is None:
        print(f"[rfid-{clean_gate}] Scanned UID {scanned_uid}, but no active camera frame or pending detection found for '{clean_gate}'.")
        reader = _readers.get(clean_gate)
        if reader:
            with reader._lock:
                reader._last_result = "NO_PENDING"
        return

    detection_id = int(target["id"])
    try:
        result = verify_detection_rfid(
            detection_id=detection_id,
            scanned_uid=scanned_uid,
        )
    except Exception as exc:
        print(f"[rfid-{clean_gate}] Error verifying detection #{detection_id}: {exc}")
        play_sound("denied", gate=clean_gate)
        reader = _readers.get(clean_gate)
        if reader:
            with reader._lock:
                reader._last_result = f"ERROR: {exc}"
        return

    if result is None:
        print(f"[rfid-{clean_gate}] Detection #{detection_id} not found.")
        play_sound("denied", gate=clean_gate)
        return

    decision = result.get("decision", "UNKNOWN")
    plate = result.get("plate_number", "?")
    status = result.get("rfid_status", "?")
    print(f"[rfid-{clean_gate}] Auto-verified #{detection_id} [{plate}] at {clean_gate} gate: {status} ({decision})")

    if decision == "ACCESS_GRANTED":
        # Both plate and RFID are verified for this gate
        play_gate_success(clean_gate)
    else:
        # RFID does not match expected UID
        play_sound("denied", gate=clean_gate)

    reader = _readers.get(clean_gate)
    if reader:
        with reader._lock:
            reader._last_result = decision


# ---------------------------------------------------------------------------
# Public API: Assign, Stop, and Query Scanners
# ---------------------------------------------------------------------------
def assign_rfid(camera_name: str, event_path: str) -> None:
    """
    Assign a physical /dev/input/event* device to a specific gate ('entrance' or 'exit').
    """
    clean_cam = str(camera_name or "").strip().lower()
    if clean_cam not in ("entrance", "exit"):
        raise ValueError("camera_name must be 'entrance' or 'exit'.")
    if not event_path or not event_path.startswith("/dev/input/"):
        raise ValueError("event_path must be a valid /dev/input/event* path.")

    # Stop any existing reader on this camera
    stop_rfid(clean_cam)

    reader = RFIDReader(
        event_path=event_path,
        camera_name=clean_cam,
        on_scan=_default_on_scan,
    )
    reader.start()
    _readers[clean_cam] = reader
    print(f"[system] Assigned {clean_cam} gate to RFID device {event_path}")


def stop_rfid(camera_name: str) -> bool:
    """Stop the RFID reader for a camera gate."""
    clean_cam = str(camera_name or "").strip().lower()
    reader = _readers.get(clean_cam)
    if reader is not None:
        reader.stop()
        _readers[clean_cam] = None
        print(f"[system] Stopped RFID reader for {clean_cam} gate.")
        return True
    return False


def get_rfid_assignments() -> dict[str, str | None]:
    """Return active device assignments per gate."""
    result: dict[str, str | None] = {}
    for camera_name, reader in _readers.items():
        if reader and reader.running:
            result[camera_name] = reader.event_path
        else:
            result[camera_name] = None
    return result


def get_last_rfid_scan(camera_name: str | None = None) -> dict[str, Any]:
    """Return status and latest scan details for all or a specific gate."""
    if camera_name:
        clean_cam = str(camera_name or "").strip().lower()
        reader = _readers.get(clean_cam)
        if reader:
            return reader.last_scan()
        return {
            "camera": clean_cam,
            "event_path": None,
            "running": False,
            "last_uid": None,
            "last_scan_at": None,
            "last_result": None,
            "error": None,
        }

    result: dict[str, Any] = {
        "latest_scanned_uid": _latest_global_scan.get("uid"),
        "latest_scanned_camera": _latest_global_scan.get("camera"),
        "latest_scanned_at": _latest_global_scan.get("timestamp"),
        "latest_scanned_epoch": _latest_global_scan.get("time_epoch", 0.0),
    }
    for cam in ("entrance", "exit"):
        reader = _readers.get(cam)
        if reader:
            result[cam] = reader.last_scan()
        else:
            result[cam] = {
                "camera": cam,
                "event_path": None,
                "running": False,
                "last_uid": None,
                "last_scan_at": None,
                "last_result": None,
                "error": None,
            }
    return result


def trigger_manual_rfid_scan(camera_name: str, scanned_uid: str) -> dict[str, Any]:
    """
    Direct software scan trigger (for browser / keyboard barcode scanner mode).
    Allows scanning into the web dashboard while specifying Entrance vs Exit gate.
    """
    clean_cam = str(camera_name or "").strip().lower()
    if clean_cam not in ("entrance", "exit"):
        raise ValueError("camera_name must be 'entrance' or 'exit'.")

    uid = str(scanned_uid or "").strip().upper()
    if not uid:
        raise ValueError("scanned_uid is required.")

    # Record scan info in reader state if exists
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _update_global_scan(clean_cam, uid)
    reader = _readers.get(clean_cam)
    if reader:
        with reader._lock:
            reader._last_uid = uid
            reader._last_scan_at = now
            reader._last_result = "WEB_SCANNED"

    _default_on_scan(clean_cam, uid)

    return {
        "ok": True,
        "camera": clean_cam,
        "scanned_uid": uid,
        "timestamp": now,
    }


def start_rfid_from_config(config: dict[str, str]) -> None:
    """Start readers from saved config on app startup."""
    for camera_name, event_path in config.items():
        if camera_name not in ("entrance", "exit"):
            continue
        if not event_path:
            continue

        if not os.path.exists(event_path):
            print(f"[rfid] Device {event_path} for {camera_name} not found — skipping.")
            continue

        try:
            assign_rfid(camera_name, event_path)
        except Exception as exc:
            print(f"[rfid] Failed to start {camera_name} RFID reader: {exc}")
