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
import csv
import io
import importlib
from datetime import timedelta
from functools import wraps
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
    delete_registered_plate,
    get_pending_manual_inputs,
    resolve_manual_input,
    discard_manual_input,
    get_pending_rfid_verifications,
    verify_detection_rfid,
    get_flagged_detections,
    review_flagged_detection,
    get_logbook_entries,
    save_device_config,
    get_device_config,
    get_sync_status_counts,
    save_rfid_config,
    get_rfid_config,
    remove_rfid_config,
)
from camera_system import (start_cameras, latest_frames, frame_locks,
                           list_video_devices, get_camera_assignments,
                           reassign_camera, stop_camera,
                           scan_plate_once_from_device, start_device_mjpeg_stream)
from rfid_system import (
    list_rfid_devices,
    assign_rfid,
    stop_rfid,
    get_rfid_assignments,
    get_last_rfid_scan,
    start_rfid_from_config,
    trigger_manual_rfid_scan,
)
from sound_system import play_sound, play_gate_success, get_last_sound_event
from relay_system import (
    init_relays,
    trigger_gate_relay,
    get_relay_status,
    set_relay_state,
    cleanup_relays,
)
from sync_worker import CloudSyncWorker

import atexit
atexit.register(cleanup_relays)


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
SYNC_WORKER = CloudSyncWorker()


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
        password_hash = bcrypt.generate_password_hash("admin123").decode("utf-8")
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


def _get_user_by_rfid(rfid_uid: str) -> dict | None:
    """Find existing user account by assigned RFID UID."""
    norm_uid = _normalize_rfid(rfid_uid)
    if not norm_uid:
        return None
    conn = _get_auth_connection()
    row = conn.execute(
        """
        SELECT u.id, u.name, u.email, u.role, u.rfid_uid, u.is_active
        FROM users u
        WHERE UPPER(REPLACE(REPLACE(REPLACE(COALESCE(u.rfid_uid, ''), ' ', ''), '-', ''), '_', '')) = ?
        LIMIT 1
        """,
        (norm_uid,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_user_by_email(email: str) -> dict | None:
    """Find existing user account by email/username."""
    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return None
    conn = _get_auth_connection()
    row = conn.execute(
        """
        SELECT u.id, u.name, u.email, u.password_hash, u.role, u.rfid_uid, u.is_active
        FROM users u
        WHERE LOWER(u.email) = ?
        LIMIT 1
        """,
        (clean_email,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_owner_of_plate(plate_number: str) -> dict | None:
    """
    Check if a normalized plate number is already registered in vehicles or registered_plates.
    Returns dict with user_id, owner_name, email, plate_number if registered, else None.
    """
    norm_p = _normalize_plate(plate_number)
    if not norm_p or norm_p == "UNKNOWN":
        return None

    conn = _get_auth_connection()
    row = conn.execute(
        """
        SELECT u.id AS user_id, u.name AS owner_name, u.email, v.plate_number
        FROM vehicles v
        JOIN users u ON u.id = v.user_id
        WHERE UPPER(REPLACE(REPLACE(v.plate_number, ' ', ''), '-', '')) = ?
        LIMIT 1
        """,
        (norm_p,),
    ).fetchone()
    if row:
        conn.close()
        return dict(row)

    row_rp = conn.execute(
        """
        SELECT 0 AS user_id, owner_name, '' AS email, plate_number
        FROM registered_plates
        WHERE UPPER(REPLACE(REPLACE(plate_number, ' ', ''), '-', '')) = ?
        LIMIT 1
        """,
        (norm_p,),
    ).fetchone()
    conn.close()
    return dict(row_rp) if row_rp else None


def _add_vehicle_to_user(user_id: int, plate_number: str, owner_name: str, rfid_uid: str | None = None) -> bool:
    """Attach a new vehicle to an existing user account and sync registered_plates."""
    norm_p = _normalize_plate(plate_number)
    if not norm_p or norm_p == "UNKNOWN":
        return False

    conn = _get_auth_connection()
    exists = conn.execute(
        """
        SELECT 1 FROM vehicles
        WHERE user_id = ? AND UPPER(REPLACE(REPLACE(plate_number, ' ', ''), '-', '')) = ?
        """,
        (user_id, norm_p),
    ).fetchone()

    was_added = False
    if not exists:
        conn.execute(
            "INSERT INTO vehicles (user_id, plate_number) VALUES (?, ?)",
            (user_id, norm_p),
        )
        conn.commit()
        was_added = True

    conn.close()
    register_plate(plate_number=norm_p, owner_name=owner_name, rfid_uid=rfid_uid or None)
    return was_added


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
        plates = [p.strip() for p in str(license_plate).split(",") if p.strip()]
        for p in plates:
            conn.execute(
                """
                INSERT INTO vehicles (user_id, plate_number)
                VALUES (?, ?)
                """,
                (user_id, p),
            )

    conn.commit()
    conn.close()

    if license_plate:
        plates = [p.strip() for p in str(license_plate).split(",") if p.strip()]
        for p in plates:
            register_plate(plate_number=p, owner_name=name, rfid_uid=rfid_uid or None)

    return user_id


def _get_auth_user_by_id(user_id: int) -> dict | None:
    conn = _get_auth_connection()
    row = conn.execute(
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
        WHERE u.id = ?
        GROUP BY u.id, u.name, u.email, u.role, u.rfid_uid, u.is_active, u.created_at
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _update_auth_user(
    user_id: int,
    name: str,
    email: str,
    role: str,
    rfid_uid: str | None = None,
    license_plate: str | None = None,
    is_active: int = 1,
    password: str | None = None,
) -> bool:
    conn = _get_auth_connection()
    if password:
        password_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        conn.execute(
            """
            UPDATE users
            SET name = ?, email = ?, password_hash = ?, role = ?, rfid_uid = ?, is_active = ?
            WHERE id = ?
            """,
            (name, email, password_hash, role, rfid_uid, is_active, user_id),
        )
    else:
        conn.execute(
            """
            UPDATE users
            SET name = ?, email = ?, role = ?, rfid_uid = ?, is_active = ?
            WHERE id = ?
            """,
            (name, email, role, rfid_uid, is_active, user_id),
        )

    conn.execute("DELETE FROM vehicles WHERE user_id = ?", (user_id,))
    if license_plate:
        plates = [p.strip() for p in str(license_plate).split(",") if p.strip()]
        for p in plates:
            conn.execute(
                """
                INSERT INTO vehicles (user_id, plate_number)
                VALUES (?, ?)
                """,
                (user_id, p),
            )

    conn.commit()
    conn.close()

    if license_plate:
        plates = [p.strip() for p in str(license_plate).split(",") if p.strip()]
        for p in plates:
            register_plate(plate_number=p, owner_name=name, rfid_uid=rfid_uid or None)

    return True


def _delete_auth_user(user_id: int) -> bool:
    conn = _get_auth_connection()
    vehicles = conn.execute("SELECT plate_number FROM vehicles WHERE user_id = ?", (user_id,)).fetchall()
    for v in vehicles:
        delete_registered_plate(v["plate_number"])

    conn.execute("DELETE FROM vehicles WHERE user_id = ?", (user_id,))
    cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def _toggle_user_status(user_id: int) -> dict | None:
    conn = _get_auth_connection()
    row = conn.execute("SELECT id, is_active FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return None
    new_status = 0 if row["is_active"] else 1
    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    return {"id": user_id, "is_active": new_status}


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


def _login_required(view_func):
    """Require a logged-in superadmin/guard session for a route."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if _is_authenticated():
            return view_func(*args, **kwargs)

        if request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required."}), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))

    return wrapped


def _role_required(*allowed_roles: str):
    """Require at least one role for a route."""
    normalized = {str(role).strip().lower() for role in allowed_roles if role}

    def decorator(view_func):
        @wraps(view_func)
        @_login_required
        def wrapped(*args, **kwargs):
            role = _current_role()
            if normalized and role not in normalized:
                if request.path.startswith("/api/"):
                    return jsonify({"error": "Forbidden."}), 403
                flash("You do not have permission to access this page.", "error")
                return redirect(url_for("dashboard"))
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


@app.context_processor
def inject_auth_user():
    return {"auth_user": session.get("auth_user")}


@app.before_request
def require_login_for_dashboard():
    open_endpoints = {"login", "logout", "serve_capture", "serve_sound_file"}
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


def _format_confidence_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _export_logbook_csv(entries: list[dict[str, Any]]) -> Response:
    """Build a CSV response for filtered logbook rows."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Timestamp",
        "Vehicle Plate",
        "Owner Name",
        "Direction",
        "Camera",
        "RFID Status",
        "OCR Confidence",
        "Entry Source",
        "Status",
    ])

    for row in entries:
        writer.writerow([
            row.get("timestamp", ""),
            row.get("plate_number", ""),
            row.get("owner_name", ""),
            row.get("direction", ""),
            row.get("camera", ""),
            row.get("rfid_status", ""),
            _format_confidence_percent(row.get("confidence", 0.0)),
            row.get("entry_source", ""),
            row.get("status", ""),
        ])

    filename = f"logbook_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


def _clip_pdf_text(value: Any, max_len: int) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _export_logbook_pdf(entries: list[dict[str, Any]], filters: dict[str, str]) -> Response:
    """Build a simple PDF report for filtered logbook rows."""
    try:
        pagesizes_mod = importlib.import_module("reportlab.lib.pagesizes")
        units_mod = importlib.import_module("reportlab.lib.units")
        canvas_mod = importlib.import_module("reportlab.pdfgen.canvas")
        page_a4 = pagesizes_mod.A4
        page_landscape = pagesizes_mod.landscape
        mm = units_mod.mm
        canvas_cls = canvas_mod.Canvas
    except Exception:
        return Response(
            "PDF export requires reportlab. Install dependencies from requirements.txt.",
            status=503,
            mimetype="text/plain",
        )

    buffer = io.BytesIO()
    page_size = page_landscape(page_a4)
    pdf = canvas_cls(buffer, pagesize=page_size)
    width, height = page_size

    left = 10 * mm
    right = width - 10 * mm
    row_height = 6 * mm

    def draw_header(page_no: int):
        y = height - 12 * mm
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left, y, "LPR System Logbook Report")
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(right, y, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        filter_text = (
            f"Filters: from={filters.get('date_from') or 'any'} | "
            f"to={filters.get('date_to') or 'any'} | "
            f"direction={filters.get('direction') or 'any'} | "
            f"status={filters.get('status') or 'any'} | "
            f"user={filters.get('user') or 'any'}"
        )
        pdf.drawString(left, y - 5 * mm, _clip_pdf_text(filter_text, 140))
        pdf.drawRightString(right, y - 5 * mm, f"Page {page_no}")

        header_y = y - 11 * mm
        pdf.setFont("Helvetica-Bold", 8)
        headers = [
            ("Timestamp", left),
            ("Plate", left + 34 * mm),
            ("Owner", left + 54 * mm),
            ("Dir", left + 92 * mm),
            ("Cam", left + 104 * mm),
            ("RFID", left + 118 * mm),
            ("Conf", left + 138 * mm),
            ("Source", left + 152 * mm),
            ("Status", left + 170 * mm),
        ]
        for title, x in headers:
            pdf.drawString(x, header_y, title)
        pdf.line(left, header_y - 1.5 * mm, right, header_y - 1.5 * mm)
        return header_y - 4 * mm

    page_no = 1
    y = draw_header(page_no)
    pdf.setFont("Helvetica", 7)

    for row in entries:
        if y <= 12 * mm:
            pdf.showPage()
            page_no += 1
            y = draw_header(page_no)
            pdf.setFont("Helvetica", 7)

        values = [
            _clip_pdf_text(row.get("timestamp", ""), 18),
            _clip_pdf_text(row.get("plate_number", ""), 10),
            _clip_pdf_text(row.get("owner_name", "-"), 22),
            _clip_pdf_text(row.get("direction", ""), 5),
            _clip_pdf_text(row.get("camera", ""), 7),
            _clip_pdf_text(row.get("rfid_status", ""), 10),
            _clip_pdf_text(_format_confidence_percent(row.get("confidence", 0.0)), 7),
            _clip_pdf_text(row.get("entry_source", ""), 8),
            _clip_pdf_text(row.get("status", ""), 10),
        ]

        x_positions = [
            left,
            left + 34 * mm,
            left + 54 * mm,
            left + 92 * mm,
            left + 104 * mm,
            left + 118 * mm,
            left + 138 * mm,
            left + 152 * mm,
            left + 170 * mm,
        ]

        for value, x in zip(values, x_positions):
            pdf.drawString(x, y, value)

        y -= row_height

    pdf.save()
    payload = buffer.getvalue()
    filename = f"logbook_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
    response = Response(payload, mimetype="application/pdf")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


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
@_login_required
def dashboard():
    """Render the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/users", methods=["GET"])
@_role_required("superadmin", "guard")
def users_management():
    """User and guard creation screen for superadmin/guard roles."""
    role = _current_role()

    users = _get_auth_users()
    return render_template(
        "user_management.html",
        users=users,
        creator_role=role,
        can_create_guard=(role == "superadmin"),
    )


@app.route("/users/create", methods=["POST"])
@_role_required("superadmin", "guard")
def users_create():
    """Create user accounts or attach new vehicles to existing accounts based on RFID/email."""
    creator_role = _current_role()

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

    if new_role == "user":
        if not final_plate:
            flash("User account requires a license plate. Scan first or enter manually.", "error")
            return redirect(url_for("users_management"))
        if not rfid_uid:
            flash("User account requires an RFID UID scan/input.", "error")
            return redirect(url_for("users_management"))

    # Parse individual requested plates
    requested_plates = [_normalize_plate(p) for p in final_plate.split(",") if _normalize_plate(p)] if final_plate else []

    # Check if RFID or Email matches an existing user account
    existing_rfid_user = _get_user_by_rfid(rfid_uid) if rfid_uid else None
    existing_email_user = _get_user_by_email(email) if email else None
    target_user = existing_rfid_user or existing_email_user
    target_user_id = target_user["id"] if target_user else None

    # ---------------------------------------------------------------------
    # RULE 1: License Plate Uniqueness
    # Reject creation if any requested plate is already registered to ANOTHER user
    # ---------------------------------------------------------------------
    for p in requested_plates:
        owner_info = _get_owner_of_plate(p)
        if owner_info:
            owner_user_id = owner_info.get("user_id")
            if target_user_id is None or owner_user_id != target_user_id:
                owner_name = owner_info.get("owner_name") or "another user"
                flash(f'Vehicle is owned by "{owner_name}"', "error")
                return redirect(url_for("users_management"))

    # ---------------------------------------------------------------------
    # RULE 2: Conflict Check (RFID vs Email mismatch between different users)
    # ---------------------------------------------------------------------
    if existing_rfid_user and existing_email_user and existing_rfid_user["id"] != existing_email_user["id"]:
        flash(f"RFID UID '{rfid_uid}' belongs to user '{existing_rfid_user['name']}', but email '{email}' belongs to user '{existing_email_user['name']}'. Credentials mismatch.", "error")
        return redirect(url_for("users_management"))

    # ---------------------------------------------------------------------
    # RULE 3: Same RFID / ID or Same Email Registered -> Attach Vehicle to Existing User
    # ---------------------------------------------------------------------
    if target_user:
        user_id = target_user["id"]
        user_name = target_user["name"]

        # Ensure RFID UID is updated on the existing user if missing
        if rfid_uid and not target_user.get("rfid_uid"):
            conn = _get_auth_connection()
            conn.execute("UPDATE users SET rfid_uid = ? WHERE id = ?", (rfid_uid, user_id))
            conn.commit()
            conn.close()

        added_plates = []
        already_plates = []
        for p in requested_plates:
            added = _add_vehicle_to_user(
                user_id=user_id,
                plate_number=p,
                owner_name=user_name,
                rfid_uid=rfid_uid or target_user.get("rfid_uid"),
            )
            if added:
                added_plates.append(p)
            else:
                already_plates.append(p)

        match_source = f"RFID UID '{rfid_uid}'" if existing_rfid_user else f"email '{email}'"
        if added_plates:
            flash(f"Recognized existing user '{user_name}' via {match_source}. Added new vehicle(s) '{', '.join(added_plates)}' to their account.", "success")
        else:
            flash(f"Vehicle(s) '{', '.join(already_plates)}' already registered to existing user '{user_name}' ({match_source}).", "info")

        return redirect(url_for("users_management"))

    # ---------------------------------------------------------------------
    # RULE 4: Completely New User Creation
    # ---------------------------------------------------------------------
    try:
        _create_auth_user(
            name=name,
            email=email,
            password=password,
            role=new_role,
            rfid_uid=rfid_uid or None,
            license_plate=final_plate or None,
        )
    except sqlite3.IntegrityError as err:
        flash(f"Unable to create account due to database constraint: {err}", "error")
        return redirect(url_for("users_management"))

    feedback_source = scan_source or scanned_plate
    if new_role == "user" and feedback_source and final_plate and feedback_source != final_plate:
        record_ocr_feedback(
            wrong_input=feedback_source,
            corrected_plate=final_plate,
            source="user_management_manual_override",
        )

    flash(f"{new_role.title()} account '{name}' created successfully with vehicle '{final_plate}'.", "success")
    return redirect(url_for("users_management"))


@app.route("/api/users/<int:user_id>", methods=["GET"])
@_role_required("superadmin")
def api_get_user(user_id: int):
    """Return user account details as JSON for editing."""
    u = _get_auth_user_by_id(user_id)
    if not u:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"user": u})


@app.route("/users/<int:user_id>/update", methods=["POST"])
@_role_required("superadmin")
def users_update(user_id: int):
    """Update user account details (Superadmin only)."""
    user = _get_auth_user_by_id(user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("users_management"))

    name = request.form.get("name", "").strip()
    email = request.form.get("username", "") or request.form.get("email", "")
    email = str(email).strip().lower()
    password = request.form.get("password", "").strip() or None
    role = request.form.get("role", "").strip().lower() or user.get("role")
    raw_manual_plate = request.form.get("manual_plate", "").strip()
    rfid_uid = _normalize_rfid(request.form.get("rfid_uid", "")) or None
    is_active = 1 if request.form.get("is_active") in ("1", "true", "on") else 0

    if raw_manual_plate:
        plates_list = [_normalize_plate(p) for p in raw_manual_plate.split(",") if _normalize_plate(p)]
        # Uniqueness check for each plate against OTHER users
        for p in plates_list:
            owner_info = _get_owner_of_plate(p)
            if owner_info and owner_info.get("user_id") != user_id:
                flash(f"License plate '{p}' is already registered to user '{owner_info.get('owner_name')}'. Each license plate must be unique.", "error")
                return redirect(url_for("users_management"))
        final_plate = ", ".join(plates_list)
    else:
        final_plate = ""

    if role == "user" and not name:
        flash("Driver name is required for user accounts.", "error")
        return redirect(url_for("users_management"))

    _update_auth_user(
        user_id=user_id,
        name=name or email,
        email=email,
        role=role,
        rfid_uid=rfid_uid,
        license_plate=final_plate or None,
        is_active=is_active,
        password=password,
    )
    flash(f"Account for '{name or email}' updated successfully.", "success")
    return redirect(url_for("users_management"))


@app.route("/users/<int:user_id>/delete", methods=["POST"])
@_role_required("superadmin")
def users_delete(user_id: int):
    """Delete a user account and associated vehicle entries (Superadmin only)."""
    current_auth = session.get("auth_user") or {}
    if current_auth.get("id") == user_id:
        flash("You cannot delete your own active superadmin account.", "error")
        return redirect(url_for("users_management"))

    ok = _delete_auth_user(user_id)
    if ok:
        flash("User account deleted successfully.", "success")
    else:
        flash("User not found or could not be deleted.", "error")
    return redirect(url_for("users_management"))


@app.route("/users/<int:user_id>/toggle_status", methods=["POST"])
@_role_required("superadmin")
def users_toggle_status(user_id: int):
    """Toggle user active/inactive status (Superadmin only)."""
    res = _toggle_user_status(user_id)
    if not res:
        flash("User not found.", "error")
    else:
        st = "Active" if res["is_active"] else "Inactive"
        flash(f"User account status updated to {st}.", "success")
    return redirect(url_for("users_management"))


@app.route("/guard/device", methods=["GET", "POST"])
@_role_required("superadmin", "guard")
def guard_device():
    """Dedicated guard page for entrance/exit camera and RFID assignment."""
    if request.method == "POST":
        form_type = request.form.get("form_type", "camera")

        if form_type == "rfid":
            return _handle_rfid_assignment()

        # --- Camera assignment (existing logic) ---
        devices = list_video_devices()
        available_indices = {int(d["index"]) for d in devices}

        entrance_device = request.form.get("entrance_device", type=int)
        exit_device = request.form.get("exit_device", type=int)

        if entrance_device is not None and entrance_device not in available_indices:
            flash("Selected entrance camera device is not available.", "error")
            return redirect(url_for("guard_device"))
        if exit_device is not None and exit_device not in available_indices:
            flash("Selected exit camera device is not available.", "error")
            return redirect(url_for("guard_device"))

        if entrance_device is not None and (exit_device is None or exit_device == entrance_device):
            remaining = [idx for idx in sorted(available_indices) if idx != entrance_device]
            if remaining:
                exit_device = remaining[0]

        pending_updates: dict[str, int] = {}
        if entrance_device is not None:
            pending_updates["entrance"] = int(entrance_device)
        if exit_device is not None:
            pending_updates["exit"] = int(exit_device)

        if not pending_updates:
            flash("Select at least one camera device to assign.", "error")
            return redirect(url_for("guard_device"))

        for camera_name, device_index in pending_updates.items():
            reassign_camera(camera_name, device_index)
            save_device_config(camera_name, device_index)

        flash("Camera assignments saved.", "success")
        return redirect(url_for("guard_device"))

    devices = list_video_devices()
    runtime_assignments = get_camera_assignments()
    saved_assignments = get_device_config()
    assignments = {
        "entrance": runtime_assignments.get("entrance", saved_assignments.get("entrance")),
        "exit": runtime_assignments.get("exit", saved_assignments.get("exit")),
    }

    rfid_devices = list_rfid_devices()
    rfid_runtime = get_rfid_assignments()
    rfid_saved = get_rfid_config()
    rfid_assignments = {
        "entrance": rfid_runtime.get("entrance") or rfid_saved.get("entrance"),
        "exit": rfid_runtime.get("exit") or rfid_saved.get("exit"),
    }

    return render_template(
        "guard_device.html",
        devices=devices,
        assignments=assignments,
        saved_assignments=saved_assignments,
        rfid_devices=rfid_devices,
        rfid_assignments=rfid_assignments,
        rfid_saved=rfid_saved,
    )


def _handle_rfid_assignment():
    """Process RFID scanner assignment form submission."""
    rfid_devices = list_rfid_devices()
    available_paths = {d["event_path"] for d in rfid_devices}

    entrance_rfid = request.form.get("entrance_rfid", "").strip()
    exit_rfid = request.form.get("exit_rfid", "").strip()

    assigned_any = False

    for camera_name, event_path in [("entrance", entrance_rfid), ("exit", exit_rfid)]:
        if not event_path:
            # "keep current" selected — skip
            continue

        if event_path == "__stop__":
            stop_rfid(camera_name)
            remove_rfid_config(camera_name)
            assigned_any = True
            continue

        if event_path not in available_paths:
            flash(f"Selected {camera_name} RFID device is not available.", "error")
            return redirect(url_for("guard_device"))

        try:
            assign_rfid(camera_name, event_path)
            save_rfid_config(camera_name, event_path)
            assigned_any = True
        except (ValueError, RuntimeError) as exc:
            flash(f"RFID {camera_name} error: {exc}", "error")
            return redirect(url_for("guard_device"))

    if assigned_any:
        flash("RFID scanner assignments saved.", "success")
    else:
        flash("No RFID changes were made.", "error")

    return redirect(url_for("guard_device"))


@app.route("/guard/review", methods=["GET"])
@_role_required("superadmin", "guard")
def guard_review():
    """Review detections that require manual guard action."""
    entries = get_flagged_detections(limit=250)
    for row in entries:
        row["image_url"] = _build_capture_url(row.get("image_path"))
        row["direction"] = "ENTRY" if str(row.get("camera", "")).lower() == "entrance" else "EXIT"

    return render_template("guard_review.html", entries=entries)


@app.route("/guard/review/<int:detection_id>/action", methods=["POST"])
@_role_required("superadmin", "guard")
def guard_review_action(detection_id: int):
    """Apply review actions: confirm, correct, or reject."""
    action = str(request.form.get("action", "")).strip().lower()
    corrected_plate = request.form.get("corrected_plate", "")

    try:
        result = review_flagged_detection(
            detection_id=detection_id,
            action=action,
            corrected_plate=corrected_plate,
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("guard_review"))

    if result is None:
        flash("Detection not found.", "error")
        return redirect(url_for("guard_review"))

    if action == "correct":
        register_plate(result["plate_number"], owner_name=None)

    flash("Flagged entry reviewed successfully.", "success")
    return redirect(url_for("guard_review"))


@app.route("/logbook", methods=["GET"])
@_role_required("superadmin", "guard")
def logbook():
    """Comprehensive logbook with filters and CSV/PDF export."""
    filters = {
        "date_from": request.args.get("date_from", "").strip(),
        "date_to": request.args.get("date_to", "").strip(),
        "direction": request.args.get("direction", "").strip().upper(),
        "status": request.args.get("status", "").strip().upper(),
        "user": request.args.get("user", "").strip(),
    }
    export_format = request.args.get("export", "").strip().lower()

    entries = get_logbook_entries(
        date_from=filters["date_from"] or None,
        date_to=filters["date_to"] or None,
        direction=filters["direction"] or None,
        status=filters["status"] or None,
        user=filters["user"] or None,
        limit=None if export_format in {"csv", "pdf"} else 500,
    )
    for row in entries:
        row["image_url"] = _build_capture_url(row.get("image_path"))
        row["confidence_percent"] = _format_confidence_percent(row.get("confidence", 0.0))

    if export_format == "csv":
        return _export_logbook_csv(entries)
    if export_format == "pdf":
        return _export_logbook_pdf(entries, filters)

    return render_template("logbook.html", entries=entries, filters=filters)


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


@app.route("/api/sync/status", methods=["GET"])
@_role_required("superadmin", "guard")
def api_sync_status():
    """Return Phase 4 cloud sync status and local queue counters."""
    status = SYNC_WORKER.status()
    status["local_counts"] = get_sync_status_counts()
    return jsonify(status)


@app.route("/api/sync/run", methods=["POST"])
@_role_required("superadmin", "guard")
def api_sync_run():
    """Trigger one on-demand sync cycle."""
    result = SYNC_WORKER.sync_once()
    payload = {
        "ok": bool(result.get("ok")),
        "result": result,
        "status": SYNC_WORKER.status(),
    }
    return jsonify(payload), (200 if result.get("ok") else 503)


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
    runtime_assignments = get_camera_assignments()
    saved_assignments = get_device_config()
    merged_assignments = {
        "entrance": runtime_assignments.get("entrance", saved_assignments.get("entrance")),
        "exit": runtime_assignments.get("exit", saved_assignments.get("exit")),
    }

    return jsonify({
        "devices": list_video_devices(),
        "assignments": merged_assignments,
        "saved_assignments": saved_assignments,
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
    save_device_config(camera_name, int(device_index))
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
# RFID scanner API endpoints
# ---------------------------------------------------------------------------
@app.route("/api/rfid_devices")
def api_rfid_devices():
    """Return available RFID HID devices and current assignments."""
    rfid_runtime = get_rfid_assignments()
    rfid_saved = get_rfid_config()
    merged = {
        "entrance": rfid_runtime.get("entrance") or rfid_saved.get("entrance"),
        "exit": rfid_runtime.get("exit") or rfid_saved.get("exit"),
    }
    return jsonify({
        "devices": list_rfid_devices(),
        "assignments": merged,
        "saved_assignments": rfid_saved,
    })


@app.route("/api/assign_rfid", methods=["POST"])
def api_assign_rfid():
    """
    Assign an RFID scanner to a camera.
    JSON body: {"camera": "entrance"|"exit", "event_path": "/dev/input/event5"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    camera_name = data.get("camera", "").strip()
    event_path = data.get("event_path", "").strip()

    if camera_name not in ("entrance", "exit"):
        return jsonify({"error": "camera must be 'entrance' or 'exit'."}), 400
    if not event_path:
        return jsonify({"error": "event_path is required."}), 400

    try:
        assign_rfid(camera_name, event_path)
        save_rfid_config(camera_name, event_path)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, "camera": camera_name, "event_path": event_path})


@app.route("/api/stop_rfid", methods=["POST"])
def api_stop_rfid():
    """
    Stop an RFID reader for a camera.
    JSON body: {"camera": "entrance"|"exit"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    camera_name = data.get("camera", "").strip()
    if camera_name not in ("entrance", "exit"):
        return jsonify({"error": "camera must be 'entrance' or 'exit'."}), 400

    stopped = stop_rfid(camera_name)
    remove_rfid_config(camera_name)
    return jsonify({"ok": stopped, "camera": camera_name})


@app.route("/api/rfid/last_scan")
def api_rfid_last_scan():
    """Return last RFID scan info per camera for dashboard polling."""
    return jsonify(get_last_rfid_scan())


@app.route("/api/rfid/lookup/<rfid_uid>")
def api_rfid_lookup(rfid_uid: str):
    """Check if an RFID UID belongs to an existing user and return user details for form auto-fill."""
    user = _get_user_by_rfid(rfid_uid)
    if user:
        return jsonify({
            "found": True,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
                "rfid_uid": user["rfid_uid"],
            }
        })
    return jsonify({"found": False, "user": None})


@app.route("/api/rfid/trigger_scan", methods=["POST"])
def api_rfid_trigger_scan():
    """
    Direct scan trigger endpoint for browser barcode / RFID keyboard input.
    JSON body: {"camera": "entrance"|"exit", "scanned_uid": "08FF20171101"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    camera_name = data.get("camera", "").strip().lower()
    scanned_uid = data.get("scanned_uid", "").strip().upper()

    if camera_name not in ("entrance", "exit"):
        return jsonify({"error": "camera must be 'entrance' or 'exit'."}), 400
    if not scanned_uid:
        return jsonify({"error": "scanned_uid is required."}), 400

    try:
        res = trigger_manual_rfid_scan(camera_name, scanned_uid)
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


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
        play_sound("denied")
        return jsonify({"error": str(exc)}), 400

    if result is None:
        play_sound("denied")
        return jsonify({"error": "Detection not found."}), 404

    decision = result.get("decision")
    if decision == "ACCESS_GRANTED":
        # Check camera from detection
        conn = _get_auth_connection()
        d_row = conn.execute("SELECT camera FROM detections WHERE id = ?", (detection_id,)).fetchone()
        conn.close()
        gate = d_row["camera"] if d_row else "entrance"
        play_gate_success(gate)
    else:
        play_sound("denied")

    return jsonify({"ok": True, **result})


@app.route("/sound/<path:filename>")
def serve_sound_file(filename):
    """Serve static MP3 sound files for web audio."""
    sound_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res", "sound")
    return send_from_directory(sound_dir, filename, mimetype="audio/mpeg")


@app.route("/api/sound/last_event", methods=["GET"])
def api_last_sound_event():
    """Return the most recent sound notification event for browser audio playback."""
    return jsonify(get_last_sound_event())


@app.route("/api/sound/play", methods=["POST"])
def api_play_sound():
    """Manually trigger a sound notification."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("sound", "")).strip().lower()
    gate = str(data.get("gate", "")).strip().lower() or None
    if not name:
        return jsonify({"error": "sound parameter required."}), 400
    if name == "gate_success":
        play_gate_success(gate or "entrance")
    else:
        play_sound(name, gate=gate)
    return jsonify({"ok": True, "sound": name, "gate": gate})


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
    play_sound("successful")
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
# Relay & Traffic Light API endpoints
# ---------------------------------------------------------------------------
@app.route("/api/relay/status", methods=["GET"])
def api_relay_status():
    """Return live status of both gate relays, active light colors, and pin mapping."""
    return jsonify(get_relay_status())


@app.route("/api/relay/trigger", methods=["POST"])
def api_relay_trigger():
    """
    Manually or programmatically trigger the green light for entrance or exit gate.
    JSON body: {"gate": "entrance"|"exit", "duration": float|optional}
    """
    data = request.get_json(silent=True) or {}
    gate = str(data.get("gate", "entrance")).strip().lower()
    duration = data.get("duration")
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            duration = None

    if gate not in ("entrance", "exit"):
        return jsonify({"error": "gate must be 'entrance' or 'exit'."}), 400

    try:
        res = trigger_gate_relay(gate=gate, duration=duration)
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/relay/set", methods=["POST"])
def api_relay_set():
    """
    Manually lock/override relay state.
    JSON body: {"gate": "entrance"|"exit", "active": true|false}
    """
    data = request.get_json(silent=True) or {}
    gate = str(data.get("gate", "entrance")).strip().lower()
    active = bool(data.get("active", False))

    if gate not in ("entrance", "exit"):
        return jsonify({"error": "gate must be 'entrance' or 'exit'."}), 400

    try:
        res = set_relay_state(gate=gate, active=active)
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  LPR System - Starting up")
    print("=" * 60)

    _ensure_auth_schema()
    print("[auth] Login required on dashboard routes")
    print("[auth] Default admin: admin@campus.local / admin123")

    # Initialize Relay & Light Hardware
    init_relays()

    # Load YOLO model; cameras are assigned from the dashboard
    start_cameras()

    # Start RFID readers from saved config
    rfid_config = get_rfid_config()
    if rfid_config:
        start_rfid_from_config(rfid_config)
        print(f"[rfid] Loaded saved RFID assignments: {rfid_config}")
    else:
        print("[rfid] No saved RFID assignments — assign scanners from the dashboard.")

    if SYNC_WORKER.enabled:
        SYNC_WORKER.start()
        print(
            "[sync] Cloud worker started -> "
            f"{SYNC_WORKER.cloud_api_base_url} "
            f"(interval={SYNC_WORKER.sync_interval_seconds}s, "
            f"batch={SYNC_WORKER.sync_batch_size})"
        )
    else:
        print("[sync] Cloud worker disabled (set CLOUD_API_BASE_URL to enable Phase 4 sync)")

    # Run Flask (accessible on the local network)
    port = int(os.getenv("PORT", "5000"))
    print(f"[web] Dashboard -> http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
