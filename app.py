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
import sqlite3
import re
from datetime import timedelta
from typing import Any

import cv2
from flask import (
    Flask,
    Response,
    flash,
    render_template,
    jsonify,
    request,
    send_from_directory,
    redirect,
    session,
    url_for,
)
from flask_bcrypt import Bcrypt

from database import (
    get_recent_detections,
    get_detections_by_date,
    get_stats,
    correct_detection_plate,
    record_ocr_feedback,
    get_registered_plates,
    get_registered_plate_records,
    register_plate,
    get_pending_manual_inputs,
    resolve_manual_input,
    discard_manual_input,
    get_pending_rfid_verifications,
    verify_detection_rfid,
)
from camera_system import (start_cameras, latest_frames, frame_locks,
                           list_video_devices, get_camera_assignments,
                           reassign_camera, stop_camera,
                           scan_plate_once_from_device, start_device_mjpeg_stream)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-change-this-secret")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)
bcrypt = Bcrypt(app)

AUTH_DB_PATH = os.path.join(BASE_DIR, "lpr_system.db")
PLATE_SANITIZER = re.compile(r"[^A-Z0-9]")
RFID_SANITIZER = re.compile(r"[^A-Z0-9]")


def _normalize_plate(value: str | None) -> str:
    if not value:
        return ""
    return PLATE_SANITIZER.sub("", str(value).upper())


def _normalize_rfid(value: str | None) -> str:
    if not value:
        return ""
    return RFID_SANITIZER.sub("", str(value).upper())


def _get_auth_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_auth_schema() -> None:
    """Ensure auth tables exist and seed a default superadmin when empty."""
    conn = _get_auth_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            rfid_uid TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plate_number TEXT NOT NULL,
            registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    user_count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    if user_count == 0:
        password_hash = bcrypt.generate_password_hash("changeme123").decode("utf-8")
        conn.execute(
            """
            INSERT INTO users (name, email, password_hash, role, is_active)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Admin", "admin@campus.local", password_hash, "superadmin", 1),
        )

    conn.commit()
    conn.close()


_ensure_auth_schema()


def _get_user_by_email(email: str) -> dict | None:
    conn = _get_auth_connection()
    row = conn.execute(
        """
        SELECT id, name, email, password_hash, role, is_active
        FROM users
        WHERE lower(email) = lower(?)
        LIMIT 1
        """,
        (email,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_auth_users() -> list[dict[str, Any]]:
    """Return all auth users for management screen."""
    conn = _get_auth_connection()
    rows = conn.execute(
        """
        SELECT
            u.id,
            u.name,
            u.email,
            u.role,
            u.rfid_uid,
            u.is_active,
            u.created_at,
            COALESCE(GROUP_CONCAT(v.plate_number, ', '), '') AS plate_numbers
        FROM users u
        LEFT JOIN vehicles v ON v.user_id = u.id
        GROUP BY u.id, u.name, u.email, u.role, u.rfid_uid, u.is_active, u.created_at
        ORDER BY datetime(u.created_at) DESC, u.id DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _is_email_in_use(email: str) -> bool:
    conn = _get_auth_connection()
    row = conn.execute(
        "SELECT 1 FROM users WHERE lower(email) = lower(?) LIMIT 1",
        (email,),
    ).fetchone()
    conn.close()
    return row is not None


def _create_auth_user(
    name: str,
    email: str,
    password: str,
    role: str,
    rfid_uid: str | None = None,
    license_plate: str | None = None,
) -> int:
    conn = _get_auth_connection()
    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    cur = conn.execute(
        """
        INSERT INTO users (name, email, password_hash, role, rfid_uid, is_active)
        VALUES (?, ?, ?, ?, ?, 1)
        """,
        (name, email, password_hash, role, rfid_uid),
    )

    user_id = int(cur.lastrowid)
    if license_plate:
        conn.execute(
            """
            INSERT INTO vehicles (user_id, plate_number)
            VALUES (?, ?)
            """,
            (user_id, license_plate),
        )

    conn.commit()
    conn.close()
    return user_id


def _current_role() -> str:
    auth_user = session.get("auth_user") or {}
    return str(auth_user.get("role", "")).lower()


def _allowed_create_roles_for(creator_role: str) -> set[str]:
    if creator_role == "superadmin":
        return {"user", "guard"}
    if creator_role == "guard":
        return {"user"}
    return set()


def _is_authenticated() -> bool:
    auth_user = session.get("auth_user") or {}
    return bool(auth_user) and str(auth_user.get("role", "")).lower() in {"superadmin", "guard"}


@app.context_processor
def inject_auth_user():
    return {"auth_user": session.get("auth_user")}


@app.before_request
def require_login_for_dashboard():
    open_endpoints = {"login", "logout", "serve_capture"}
    if request.endpoint in open_endpoints:
        return None

    # Allow browser static handling when enabled.
    if request.endpoint == "static":
        return None

    if _is_authenticated():
        return None

    # Drop stale/forbidden sessions (e.g., role=user).
    session.pop("auth_user", None)

    if request.path.startswith("/api/"):
        return jsonify({"error": "Authentication required."}), 401

    return redirect(url_for("login", next=request.full_path.rstrip("?")))


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


@app.route("/users", methods=["GET"])
def users_management():
    """User and guard creation screen for superadmin/guard roles."""
    if not _is_authenticated():
        return redirect(url_for("login", next="/users"))

    role = _current_role()
    if role not in {"superadmin", "guard"}:
        return redirect(url_for("dashboard"))

    users = _get_auth_users()
    return render_template(
        "user_management.html",
        users=users,
        creator_role=role,
        can_create_guard=(role == "superadmin"),
    )


@app.route("/users/create", methods=["POST"])
def users_create():
    """Create user accounts according to creator role permissions."""
    if not _is_authenticated():
        return redirect(url_for("login", next="/users"))

    creator_role = _current_role()
    if creator_role not in {"superadmin", "guard"}:
        return redirect(url_for("dashboard"))

    name = request.form.get("name", "").strip()
    credential_raw = request.form.get("username", "") or request.form.get("email", "")
    email = str(credential_raw).strip().lower()
    password = request.form.get("password", "")
    new_role = request.form.get("role", "").strip().lower()
    scanned_plate = _normalize_plate(request.form.get("scanned_plate", ""))
    manual_plate = _normalize_plate(request.form.get("manual_plate", ""))
    scan_source = _normalize_plate(request.form.get("plate_scan_source", ""))
    rfid_uid = _normalize_rfid(request.form.get("rfid_uid", ""))
    final_plate = manual_plate or scanned_plate
    allowed_roles = _allowed_create_roles_for(creator_role)

    if not email or not password:
        flash("Username/Email and password are required.", "error")
        return redirect(url_for("users_management"))

    if new_role == "guard":
        # Guard creation is credentials-only.
        if not name:
            name = email
        scanned_plate = ""
        manual_plate = ""
        scan_source = ""
        rfid_uid = ""
        final_plate = ""
    elif not name:
        flash("Name is required for user accounts.", "error")
        return redirect(url_for("users_management"))

    if new_role not in allowed_roles:
        if creator_role == "guard":
            flash("Guards can only create regular user accounts.", "error")
        else:
            flash("Invalid role selected.", "error")
        return redirect(url_for("users_management"))

    # Defense-in-depth: guard role can never create another guard.
    if creator_role == "guard" and new_role == "guard":
        flash("Guards cannot create other guards.", "error")
        return redirect(url_for("users_management"))

    if _is_email_in_use(email):
        flash("That email is already registered.", "error")
        return redirect(url_for("users_management"))

    if new_role == "user":
        if not final_plate:
            flash("User account requires a license plate. Scan first or enter manually.", "error")
            return redirect(url_for("users_management"))
        if not rfid_uid:
            flash("User account requires an RFID UID scan/input.", "error")
            return redirect(url_for("users_management"))

    try:
        _create_auth_user(
            name=name,
            email=email,
            password=password,
            role=new_role,
            rfid_uid=rfid_uid or None,
            license_plate=final_plate or None,
        )
    except sqlite3.IntegrityError:
        flash("Unable to create account due to a database constraint.", "error")
        return redirect(url_for("users_management"))

    if new_role == "user" and final_plate:
        register_plate(plate_number=final_plate, owner_name=name, rfid_uid=rfid_uid or None)

    feedback_source = scan_source or scanned_plate
    if new_role == "user" and feedback_source and final_plate and feedback_source != final_plate:
        record_ocr_feedback(
            wrong_input=feedback_source,
            corrected_plate=final_plate,
            source="user_management_manual_override",
        )

    flash(f"{new_role.title()} account created successfully.", "success")
    return redirect(url_for("users_management"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if _is_authenticated():
        return redirect(url_for("dashboard"))

    error = None
    next_url = request.args.get("next", "").strip()

    if request.method == "POST":
        credential = request.form.get("credential", request.form.get("email", "")).strip().lower()
        password = request.form.get("password", "")

        if not credential or not password:
            error = "Username/Email and password are required."
        else:
            user = _get_user_by_email(credential)
            if user is None or not bcrypt.check_password_hash(user["password_hash"], password):
                error = "Invalid username/email or password."
            elif not bool(user.get("is_active", 0)):
                error = "This account is inactive. Contact the administrator."
            elif str(user.get("role", "")).lower() not in {"superadmin", "guard"}:
                error = "Your account role is not allowed to sign in here."
            else:
                session.permanent = True
                session["auth_user"] = {
                    "id": int(user["id"]),
                    "name": str(user["name"]),
                    "email": str(user["email"]),
                    "role": str(user["role"]),
                }

                if next_url.startswith("/") and not next_url.startswith("//"):
                    return redirect(next_url)
                return redirect(url_for("dashboard"))

    return render_template("login.html", error=error)


@app.route("/logout", methods=["GET"])
def logout():
    session.pop("auth_user", None)
    return redirect(url_for("login"))


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


@app.route("/video/account_creation")
def video_account_creation():
    """MJPEG preview stream for account-creation camera selection."""
    device_index = request.args.get("device_index", type=int)
    if device_index is None or device_index < 0:
        return ("", 400)

    try:
        stream = start_device_mjpeg_stream(device_index)
    except (ValueError, RuntimeError):
        return ("", 503)

    return Response(
        stream,
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


@app.route("/api/detections/<int:detection_id>/correct", methods=["POST"])
def api_correct_detection(detection_id: int):
    """
    Correct a wrong detection and feed the correction into OCR learning memory.
    JSON body: {"plate_number": "ABC1234"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    corrected_plate = str(data.get("plate_number", "")).strip()
    if not corrected_plate:
        return jsonify({"error": "plate_number is required."}), 400

    try:
        result = correct_detection_plate(detection_id=detection_id, corrected_plate=corrected_plate)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if result is None:
        return jsonify({"error": "Detection not found."}), 404

    # Keep corrected plates in registry to improve matching confidence downstream.
    register_plate(result["plate_number"], owner_name=None)
    return jsonify({"ok": True, **result})


@app.route("/api/scan_plate_once", methods=["POST"])
def api_scan_plate_once():
    """
    One-shot plate scan from selected physical camera device.
    JSON body: {"device_index": int}
    """
    data = request.get_json(silent=True) or {}
    device_index = data.get("device_index")
    if not isinstance(device_index, int) or device_index < 0:
        return jsonify({"error": "device_index must be a non-negative integer."}), 400

    try:
        result = scan_plate_once_from_device(device_index)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({"ok": True, **result})


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


@app.route("/api/registered_plate_records", methods=["GET"])
def api_registered_plate_records():
    """Return registered plate records including owner and RFID UID."""
    return jsonify({"records": get_registered_plate_records(limit=300)})


@app.route("/api/registered_plates", methods=["POST"])
def api_register_plate():
    """
    Register a plate for fuzzy matching and RFID verification.
    JSON body: {"plate_number": str, "owner_name": str|optional, "rfid_uid": str|optional}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    plate_number = str(data.get("plate_number", "")).strip()
    owner_name = data.get("owner_name")
    rfid_uid = data.get("rfid_uid")
    if not plate_number:
        return jsonify({"error": "plate_number is required."}), 400

    ok = register_plate(
        plate_number=plate_number,
        owner_name=owner_name,
        rfid_uid=rfid_uid,
    )
    if not ok:
        return jsonify({"error": "plate_number is invalid."}), 400

    return jsonify({
        "ok": True,
        "plate_number": plate_number,
        "owner_name": owner_name,
        "rfid_uid": rfid_uid,
    })


@app.route("/api/rfid/pending", methods=["GET"])
def api_pending_rfid_verifications():
    """Return pending RFID verification items for captured detections."""
    items = get_pending_rfid_verifications(limit=20)
    for item in items:
        item["image_url"] = _build_capture_url(item.get("image_path"))
    return jsonify({"items": items})


@app.route("/api/rfid/verify", methods=["POST"])
def api_verify_rfid():
    """
    Verify an RFID scan against an existing detection.
    JSON body: {"detection_id": int, "scanned_uid": str}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    detection_id = data.get("detection_id")
    scanned_uid = str(data.get("scanned_uid", "")).strip()

    if not isinstance(detection_id, int) or detection_id <= 0:
        return jsonify({"error": "detection_id must be a positive integer."}), 400
    if not scanned_uid:
        return jsonify({"error": "scanned_uid is required."}), 400

    try:
        result = verify_detection_rfid(detection_id=detection_id, scanned_uid=scanned_uid)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if result is None:
        return jsonify({"error": "Detection not found."}), 404

    return jsonify({"ok": True, **result})


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

    _ensure_auth_schema()
    print("[auth] Login required on dashboard routes")
    print("[auth] Default admin: admin@campus.local / changeme123 (first run only)")

    # Load YOLO model; cameras are assigned from the dashboard
    start_cameras()

    # Run Flask (accessible on the local network)
    print("[web] Dashboard -> http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
