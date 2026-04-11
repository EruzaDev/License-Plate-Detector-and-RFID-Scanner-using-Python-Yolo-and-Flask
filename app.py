"""
app.py - Flask web server + dashboard for the LPR system.

Endpoints:
    /               -> dashboard UI
    /video/entrance -> MJPEG stream for entrance camera (with YOLO boxes)
    /video/exit     -> MJPEG stream for exit camera (with YOLO boxes)
    /api/detections -> JSON: recent detections (query ?date=YYYY-MM-DD to filter)
    /api/stats      -> JSON: summary statistics
    /captures/<fn>  -> serve saved plate images
"""

import os
import time
import threading

import cv2
from flask import Flask, Response, render_template, jsonify, request, send_from_directory

from database import get_recent_detections, get_detections_by_date, get_stats
from camera_system import (start_cameras, latest_frames, frame_locks,
                           list_video_devices, get_camera_assignments,
                           reassign_camera, stop_camera)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))


def _normalize_capture_filename(image_path: str | None) -> str | None:
    """Normalize DB/UI capture paths to a filename under captures/."""
    if not image_path:
        return None

    normalized = str(image_path).replace("\\", "/").lstrip("/")
    if normalized.startswith("captures/"):
        normalized = normalized[len("captures/"):]

    # Only allow filename lookups directly under captures/.
    normalized = os.path.basename(normalized)
    if normalized in ("", ".", ".."):
        return None

    return normalized or None


def _build_capture_url(image_path: str | None) -> str | None:
    """Return a web URL for existing capture files, else None."""
    filename = _normalize_capture_filename(image_path)
    if not filename:
        return None

    abs_path = os.path.join(CAPTURES_DIR, filename)
    if not os.path.isfile(abs_path):
        return None

    return f"/captures/{filename}"


# ---------------------------------------------------------------------------
# MJPEG streaming
# ---------------------------------------------------------------------------
def _generate_mjpeg(camera_name: str):
    """
    Generator that yields JPEG frames as an MJPEG stream.
    Reads the latest annotated frame (with YOLO bounding boxes)
    from shared memory populated by camera_system.
    """
    while True:
        frame = None
        with frame_locks[camera_name]:
            if latest_frames[camera_name] is not None:
                frame = latest_frames[camera_name].copy()

        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
        else:
            # No frame yet - send a tiny pause
            time.sleep(0.1)

        # ~15 FPS cap to save Pi 5 bandwidth
        time.sleep(0.066)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    """Render the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/video/entrance")
def video_entrance():
    """MJPEG stream for the entrance camera."""
    return Response(
        _generate_mjpeg("entrance"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video/exit")
def video_exit():
    """MJPEG stream for the exit camera."""
    return Response(
        _generate_mjpeg("exit"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/detections")
def api_detections():
    """
    Return recent detections as JSON.
    Optional query param: ?date=YYYY-MM-DD to filter by date.
    """
    date_filter = request.args.get("date", "").strip()
    # Basic validation: only allow YYYY-MM-DD format
    if date_filter:
        if len(date_filter) != 10 or date_filter[4] != "-" or date_filter[7] != "-":
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
        detections = get_detections_by_date(date_filter)
    else:
        detections = get_recent_detections(limit=100)

    # Attach a safe thumbnail URL only when the file still exists.
    for row in detections:
        row["image_url"] = _build_capture_url(row.get("image_path"))

    return jsonify(detections)


@app.route("/api/stats")
def api_stats():
    """Return summary statistics as JSON."""
    return jsonify(get_stats())


@app.route("/captures/<path:filename>")
def serve_capture(filename: str):
    """Serve a saved plate image from the captures folder."""
    normalized = _normalize_capture_filename(filename)
    if not normalized:
        return ("", 404)
    return send_from_directory(CAPTURES_DIR, normalized)


@app.route("/api/devices")
def api_devices():
    """Return available video devices and current camera assignments."""
    return jsonify({
        "devices": list_video_devices(),
        "assignments": get_camera_assignments(),
    })


@app.route("/api/assign_camera", methods=["POST"])
def api_assign_camera():
    """
    Reassign a camera to a different video device.
    JSON body: {"camera": "entrance"|"exit", "device_index": int}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    camera_name = data.get("camera", "").strip()
    device_index = data.get("device_index")

    if camera_name not in ("entrance", "exit"):
        return jsonify({"error": "camera must be 'entrance' or 'exit'."}), 400
    if not isinstance(device_index, int) or device_index < 0:
        return jsonify({"error": "device_index must be a non-negative integer."}), 400

    reassign_camera(camera_name, device_index)
    return jsonify({"ok": True, "camera": camera_name, "device_index": device_index})


@app.route("/api/stop_camera", methods=["POST"])
def api_stop_camera():
    """
    Stop a running camera.
    JSON body: {"camera": "entrance"|"exit"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    camera_name = data.get("camera", "").strip()
    if camera_name not in ("entrance", "exit"):
        return jsonify({"error": "camera must be 'entrance' or 'exit'."}), 400

    stopped = stop_camera(camera_name)
    return jsonify({"ok": stopped, "camera": camera_name})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  LPR System - Starting up")
    print("=" * 60)

    # Load YOLO model; cameras are assigned from the dashboard
    start_cameras()

    # Run Flask (accessible on the local network)
    print("[web] Dashboard -> http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
