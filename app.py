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

from database import (
    get_recent_detections,
    get_detections_by_date,
    get_stats,
    get_registered_plates,
    register_plate,
    get_pending_manual_inputs,
    resolve_manual_input,
    discard_manual_input,
)
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


@app.route("/api/registered_plates", methods=["GET"])
def api_registered_plates():
    """Return all registered plates used by fuzzy matching."""
    return jsonify({"plates": get_registered_plates()})


@app.route("/api/registered_plates", methods=["POST"])
def api_register_plate():
    """
    Register a plate for fuzzy matching.
    JSON body: {"plate_number": str, "owner_name": str|optional}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    plate_number = str(data.get("plate_number", "")).strip()
    owner_name = data.get("owner_name")
    if not plate_number:
        return jsonify({"error": "plate_number is required."}), 400

    ok = register_plate(plate_number=plate_number, owner_name=owner_name)
    if not ok:
        return jsonify({"error": "plate_number is invalid."}), 400

    return jsonify({"ok": True, "plate_number": plate_number})


@app.route("/api/manual_inputs/pending", methods=["GET"])
def api_pending_manual_inputs():
    """Return unresolved manual plate-entry items."""
    items = get_pending_manual_inputs(limit=20)
    for item in items:
        item["image_url"] = _build_capture_url(item.get("image_path"))
    return jsonify({"items": items})


@app.route("/api/manual_inputs/<int:item_id>/resolve", methods=["POST"])
def api_resolve_manual_input(item_id: int):
    """
    Resolve a pending manual-input item by submitting a plate number.
    JSON body: {"plate_number": "ABC1234"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    plate_number = str(data.get("plate_number", "")).strip()
    if not plate_number:
        return jsonify({"error": "plate_number is required."}), 400

    try:
        result = resolve_manual_input(item_id, plate_number)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if result is None:
        return jsonify({"error": "Manual input item not found."}), 404

    # Store accepted manual entries in the fuzzy-match registry for future runs.
    register_plate(result["plate_number"], owner_name=None)
    return jsonify({"ok": True, **result})


@app.route("/api/manual_inputs/<int:item_id>/discard", methods=["POST"])
def api_discard_manual_input(item_id: int):
    """
    Discard a pending manual-input item.
    Optional JSON body: {"delete_image": true|false} (default true)
    """
    data = request.get_json(silent=True) or {}
    delete_image = bool(data.get("delete_image", True))

    result = discard_manual_input(item_id)
    if result is None:
        return jsonify({"error": "Manual input item not found."}), 404

    image_deleted = False
    if delete_image:
        filename = _normalize_capture_filename(result.get("image_path"))
        if filename:
            abs_path = os.path.join(CAPTURES_DIR, filename)
            if os.path.isfile(abs_path):
                try:
                    os.remove(abs_path)
                    image_deleted = True
                except OSError:
                    image_deleted = False

    return jsonify({"ok": True, "image_deleted": image_deleted, **result})


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
