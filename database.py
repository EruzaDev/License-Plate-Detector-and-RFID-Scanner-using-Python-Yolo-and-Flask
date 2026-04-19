"""
database.py — SQLite setup and queries for the LPR system.
Stores every plate detection with metadata and provides query helpers.
"""

import sqlite3
import os
import threading
import re
from difflib import SequenceMatcher
from datetime import datetime
from typing import Any

# Path to the database file (sits next to this script)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lpr_system.db")

# Thread-local storage so each thread gets its own connection
_local = threading.local()
_PLATE_SANITIZER = re.compile(r"[^A-Z0-9]")
_RFID_SANITIZER = re.compile(r"[^A-Z0-9]")


def _normalize_plate(plate: str | None) -> str:
    """Normalize a plate string for consistent storage and matching."""
    if not plate:
        return ""
    return _PLATE_SANITIZER.sub("", str(plate).upper())


def _normalize_rfid_uid(uid: str | None) -> str:
    """Normalize RFID UID text to uppercase alphanumeric form."""
    if not uid:
        return ""
    return _RFID_SANITIZER.sub("", str(uid).upper())


def _get_connection():
    """Return a thread-local SQLite connection (created on first call)."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row  # dict-like access
        _local.conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    return _local.conn


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_def: str):
    """Add a table column if missing (lightweight migration helper)."""
    col_name = column_def.split()[0]
    existing_cols = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})")
    }
    if col_name not in existing_cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")


def init_db():
    """Create the detections table if it doesn't exist yet."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT    NOT NULL,
            camera       TEXT    NOT NULL,   -- 'entrance' or 'exit'
            timestamp    TEXT    NOT NULL,    -- ISO-8601
            image_path   TEXT    NOT NULL,
            confidence   REAL    NOT NULL,
            ocr_raw      TEXT,
            ocr_corrected TEXT,
            plate_valid  INTEGER DEFAULT 0,
            matched_plate TEXT,
            match_score  REAL,
            match_status TEXT DEFAULT 'NO_MATCH',
            rfid_status  TEXT DEFAULT 'NOT_REQUIRED',
            expected_rfid_uid TEXT,
            scanned_rfid_uid  TEXT,
            rfid_verified_at  TEXT
        )
    """)

    # Backward-compatible migration for older DB files that predate Phase 1.
    _ensure_column(conn, "detections", "ocr_raw TEXT")
    _ensure_column(conn, "detections", "ocr_corrected TEXT")
    _ensure_column(conn, "detections", "plate_valid INTEGER DEFAULT 0")
    _ensure_column(conn, "detections", "matched_plate TEXT")
    _ensure_column(conn, "detections", "match_score REAL")
    _ensure_column(conn, "detections", "match_status TEXT DEFAULT 'NO_MATCH'")
    _ensure_column(conn, "detections", "rfid_status TEXT DEFAULT 'NOT_REQUIRED'")
    _ensure_column(conn, "detections", "expected_rfid_uid TEXT")
    _ensure_column(conn, "detections", "scanned_rfid_uid TEXT")
    _ensure_column(conn, "detections", "rfid_verified_at TEXT")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS registered_plates (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL UNIQUE,
            owner_name   TEXT,
            rfid_uid     TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_column(conn, "registered_plates", "rfid_uid TEXT")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS manual_inputs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            camera        TEXT NOT NULL,
            timestamp     TEXT NOT NULL,
            image_path    TEXT NOT NULL,
            confidence    REAL NOT NULL,
            ocr_raw       TEXT,
            ocr_corrected TEXT,
            match_status  TEXT DEFAULT 'NO_MATCH',
            resolution_status TEXT DEFAULT 'PENDING',
            resolved_plate TEXT,
            resolved_at   TEXT,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    _ensure_column(conn, "manual_inputs", "resolution_status TEXT DEFAULT 'PENDING'")

    conn.execute(
        """
        UPDATE manual_inputs
        SET resolution_status = CASE
            WHEN resolved_at IS NULL THEN 'PENDING'
            WHEN resolved_plate IS NOT NULL THEN 'RESOLVED'
            ELSE 'DISCARDED'
        END
        WHERE resolution_status IS NULL OR TRIM(resolution_status) = ''
        """
    )

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_registered_plates_plate
        ON registered_plates (plate_number)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_manual_inputs_resolved
        ON manual_inputs (resolved_at, id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_timestamp
        ON detections (timestamp DESC)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ocr_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wrong_input TEXT NOT NULL,
            corrected_plate TEXT NOT NULL,
            source TEXT DEFAULT 'manual',
            usage_count INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(wrong_input, corrected_plate)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ocr_feedback_wrong
        ON ocr_feedback (wrong_input)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS device_config (
            camera TEXT PRIMARY KEY,
            device_index INTEGER NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_device_config_updated_at
        ON device_config (updated_at DESC)
    """)
    conn.commit()


def insert_detection(plate_number: str, camera: str, timestamp: str,
                     image_path: str, confidence: float,
                     ocr_raw: str | None = None,
                     ocr_corrected: str | None = None,
                     plate_valid: bool | None = None,
                     matched_plate: str | None = None,
                     match_score: float | None = None,
                     match_status: str | None = None,
                     rfid_status: str | None = None,
                     expected_rfid_uid: str | None = None,
                     scanned_rfid_uid: str | None = None,
                     rfid_verified_at: str | None = None) -> int:
    """
    Insert a new detection record.
    Returns the new row id.
    """
    conn = _get_connection()
    normalized_plate = _normalize_plate(plate_number) or "UNKNOWN"
    normalized_ocr_raw = _normalize_plate(ocr_raw) or None
    normalized_ocr_corrected = _normalize_plate(ocr_corrected) or None
    normalized_matched_plate = _normalize_plate(matched_plate) or None
    plate_valid_value = None if plate_valid is None else int(bool(plate_valid))
    status_value = (match_status or "NO_MATCH").upper()
    expected_uid_value = _normalize_rfid_uid(expected_rfid_uid) or None
    scanned_uid_value = _normalize_rfid_uid(scanned_rfid_uid) or None
    if rfid_status is None:
        rfid_status_value = "NOT_SCANNED" if expected_uid_value else "NOT_REQUIRED"
    else:
        rfid_status_value = str(rfid_status).upper()

    cur = conn.execute(
        """INSERT INTO detections (
               plate_number,
               camera,
               timestamp,
               image_path,
               confidence,
               ocr_raw,
               ocr_corrected,
               plate_valid,
               matched_plate,
               match_score,
               match_status,
               rfid_status,
               expected_rfid_uid,
               scanned_rfid_uid,
               rfid_verified_at
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            normalized_plate,
            camera,
            timestamp,
            image_path,
            confidence,
            normalized_ocr_raw,
            normalized_ocr_corrected,
            plate_valid_value,
            normalized_matched_plate,
            match_score,
            status_value,
            rfid_status_value,
            expected_uid_value,
            scanned_uid_value,
            rfid_verified_at,
        ),
    )
    conn.commit()
    return cur.lastrowid


def register_plate(
    plate_number: str,
    owner_name: str | None = None,
    rfid_uid: str | None = None,
) -> bool:
    """
    Register or update a known plate for fuzzy matching.
    Returns False when the plate is empty after normalization.
    """
    normalized = _normalize_plate(plate_number)
    if not normalized:
        return False
    normalized_uid = _normalize_rfid_uid(rfid_uid) or None

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO registered_plates (plate_number, owner_name, rfid_uid)
        VALUES (?, ?, ?)
        ON CONFLICT(plate_number) DO UPDATE SET
            owner_name = COALESCE(excluded.owner_name, registered_plates.owner_name),
            rfid_uid = COALESCE(excluded.rfid_uid, registered_plates.rfid_uid)
        """,
        (normalized, owner_name, normalized_uid),
    )
    conn.commit()
    return True


def get_registered_plates() -> list[str]:
    """Return all normalized registered plates for fuzzy matching."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT plate_number FROM registered_plates ORDER BY plate_number"
    ).fetchall()
    return [str(r["plate_number"]) for r in rows]


def get_registered_plate_record(plate_number: str) -> dict | None:
    """Return plate registration details including owner and RFID UID."""
    normalized = _normalize_plate(plate_number)
    if not normalized:
        return None

    conn = _get_connection()
    row = conn.execute(
        """
        SELECT plate_number, owner_name, rfid_uid
        FROM registered_plates
        WHERE plate_number = ?
        """,
        (normalized,),
    ).fetchone()
    return dict(row) if row else None


def get_registered_plate_records(limit: int = 200) -> list:
    """Return registered plate records with optional RFID UIDs."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT plate_number, owner_name, rfid_uid, created_at
        FROM registered_plates
        ORDER BY plate_number ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_pending_rfid_verifications(limit: int = 20) -> list:
    """Return detections that require RFID scan and are still pending."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT id, plate_number, camera, timestamp, image_path, confidence,
               rfid_status, expected_rfid_uid, scanned_rfid_uid, rfid_verified_at
        FROM detections
        WHERE rfid_status = 'NOT_SCANNED'
        ORDER BY id ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def verify_detection_rfid(detection_id: int, scanned_uid: str) -> dict | None:
    """
    Verify a pending detection against a scanned RFID UID.
    Returns verification summary or None when detection does not exist.
    Raises ValueError if UID is invalid or detection does not need RFID.
    """
    normalized_uid = _normalize_rfid_uid(scanned_uid)
    if not normalized_uid:
        raise ValueError("RFID UID is required.")

    conn = _get_connection()
    row = conn.execute(
        """
        SELECT id, plate_number, rfid_status, expected_rfid_uid
        FROM detections
        WHERE id = ?
        """,
        (detection_id,),
    ).fetchone()
    if row is None:
        return None

    expected_uid = _normalize_rfid_uid(row["expected_rfid_uid"])
    if not expected_uid:
        raise ValueError("This detection does not require RFID verification.")

    current_status = str(row["rfid_status"] or "NOT_REQUIRED").upper()
    if current_status not in ("NOT_SCANNED", "MISMATCH"):
        raise ValueError("RFID verification is already finalized for this detection.")

    new_status = "MATCH" if normalized_uid == expected_uid else "MISMATCH"
    verified_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        UPDATE detections
        SET rfid_status = ?,
            scanned_rfid_uid = ?,
            rfid_verified_at = ?
        WHERE id = ?
        """,
        (new_status, normalized_uid, verified_at, detection_id),
    )
    conn.commit()

    return {
        "detection_id": int(row["id"]),
        "plate_number": str(row["plate_number"]),
        "rfid_status": new_status,
        "expected_rfid_uid": expected_uid,
        "scanned_rfid_uid": normalized_uid,
        "decision": "ACCESS_GRANTED" if new_status == "MATCH" else "FLAGGED_MISMATCH",
    }


def enqueue_manual_input(
    camera: str,
    timestamp: str,
    image_path: str,
    confidence: float,
    ocr_raw: str | None = None,
    ocr_corrected: str | None = None,
    match_status: str | None = None,
) -> int:
    """
    Queue a capture for manual plate entry when OCR cannot produce a valid plate.
    Returns queued row id.
    """
    conn = _get_connection()
    cur = conn.execute(
        """INSERT INTO manual_inputs (
               camera,
               timestamp,
               image_path,
               confidence,
               ocr_raw,
               ocr_corrected,
               match_status,
               resolution_status
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            camera,
            timestamp,
            image_path,
            confidence,
            _normalize_plate(ocr_raw) or None,
            _normalize_plate(ocr_corrected) or None,
            (match_status or "NO_MATCH").upper(),
            "PENDING",
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_pending_manual_inputs(limit: int = 20) -> list:
    """Return unresolved manual-input items in FIFO order."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT *
        FROM manual_inputs
                WHERE resolved_at IS NULL
                    AND COALESCE(NULLIF(TRIM(resolution_status), ''), 'PENDING') = 'PENDING'
        ORDER BY id ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_manual_input(manual_input_id: int, plate_number: str) -> dict | None:
    """
    Resolve a queued manual input by inserting a finalized detection row.
    Returns a summary dict, or None when the queue item does not exist.
    Raises ValueError for invalid manual plates.
    """
    normalized_plate = _normalize_plate(plate_number)
    if not normalized_plate or normalized_plate == "UNKNOWN":
        raise ValueError("Manual plate entry is invalid.")

    conn = _get_connection()
    row = conn.execute(
        """
        SELECT *
        FROM manual_inputs
                WHERE id = ?
                    AND resolved_at IS NULL
                    AND COALESCE(NULLIF(TRIM(resolution_status), ''), 'PENDING') = 'PENDING'
        """,
        (manual_input_id,),
    ).fetchone()

    if row is None:
        return None

    detection_id = insert_detection(
        plate_number=normalized_plate,
        camera=str(row["camera"]),
        timestamp=str(row["timestamp"]),
        image_path=str(row["image_path"]),
        confidence=float(row["confidence"]),
        ocr_raw=row["ocr_raw"],
        ocr_corrected=row["ocr_corrected"],
        plate_valid=True,
        matched_plate=normalized_plate,
        match_score=100.0,
        match_status="MANUAL_ENTRY",
    )

    resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        UPDATE manual_inputs
        SET resolved_plate = ?,
            resolved_at = ?,
            resolution_status = 'RESOLVED'
        WHERE id = ?
        """,
        (normalized_plate, resolved_at, manual_input_id),
    )
    conn.commit()

    feedback_candidates = [row["ocr_corrected"], row["ocr_raw"]]
    for wrong_value in feedback_candidates:
        if record_ocr_feedback(wrong_value, normalized_plate, source="manual_queue"):
            break

    return {
        "manual_input_id": manual_input_id,
        "plate_number": normalized_plate,
        "detection_id": detection_id,
    }


def record_ocr_feedback(
    wrong_input: str | None,
    corrected_plate: str | None,
    source: str = "manual",
) -> bool:
    """
    Record a wrong->correct OCR mapping for future auto-corrections.
    Returns True when feedback is persisted.
    """
    wrong_norm = _normalize_plate(wrong_input)
    corrected_norm = _normalize_plate(corrected_plate)
    if not wrong_norm or wrong_norm == "UNKNOWN":
        return False
    if not corrected_norm or corrected_norm == "UNKNOWN":
        return False
    if wrong_norm == corrected_norm:
        return False

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO ocr_feedback (wrong_input, corrected_plate, source, usage_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(wrong_input, corrected_plate) DO UPDATE SET
            usage_count = usage_count + 1,
            updated_at = CURRENT_TIMESTAMP,
            source = COALESCE(excluded.source, ocr_feedback.source)
        """,
        (wrong_norm, corrected_norm, source),
    )
    conn.commit()
    return True


def suggest_plate_from_feedback(input_text: str | None) -> tuple[str | None, float, int]:
    """
    Suggest a corrected plate from learned OCR feedback.

    Returns
    -------
    tuple[str | None, float, int]
        (corrected_plate, confidence_score_0_to_100, usage_count)
    """
    normalized = _normalize_plate(input_text)
    if not normalized or normalized == "UNKNOWN":
        return (None, 0.0, 0)

    conn = _get_connection()

    # Exact match is strongest and applies immediately.
    exact = conn.execute(
        """
        SELECT corrected_plate, usage_count
        FROM ocr_feedback
        WHERE wrong_input = ?
        ORDER BY usage_count DESC, updated_at DESC
        LIMIT 1
        """,
        (normalized,),
    ).fetchone()
    if exact is not None:
        return (str(exact["corrected_plate"]), 100.0, int(exact["usage_count"]))

    # Fuzzy fallback for near-identical OCR mistakes.
    rows = conn.execute(
        """
        SELECT wrong_input, corrected_plate, usage_count
        FROM ocr_feedback
        """
    ).fetchall()
    if not rows:
        return (None, 0.0, 0)

    best_plate = None
    best_score = 0.0
    best_uses = 0

    for row in rows:
        wrong = str(row["wrong_input"])
        score = SequenceMatcher(None, normalized, wrong).ratio() * 100.0
        uses = int(row["usage_count"])
        if score > best_score or (abs(score - best_score) < 1e-6 and uses > best_uses):
            best_score = score
            best_uses = uses
            best_plate = str(row["corrected_plate"])

    # Conservative fuzzy rule to avoid over-correcting unseen strings.
    if best_plate is not None and best_score >= 92.0 and best_uses >= 2:
        return (best_plate, round(best_score, 2), best_uses)

    return (None, 0.0, 0)


def correct_detection_plate(detection_id: int, corrected_plate: str) -> dict | None:
    """
    Correct an existing detection plate and feed the correction back into OCR memory.
    Returns correction summary or None when detection does not exist.
    """
    normalized_plate = _normalize_plate(corrected_plate)
    if not normalized_plate or normalized_plate == "UNKNOWN":
        raise ValueError("Corrected plate is invalid.")

    conn = _get_connection()
    row = conn.execute(
        """
        SELECT id, plate_number, ocr_raw, ocr_corrected
        FROM detections
        WHERE id = ?
        """,
        (detection_id,),
    ).fetchone()

    if row is None:
        return None

    previous_plate = _normalize_plate(row["plate_number"])

    conn.execute(
        """
        UPDATE detections
        SET plate_number = ?,
            ocr_corrected = ?,
            plate_valid = 1,
            matched_plate = ?,
            match_score = 100.0,
            match_status = 'MANUAL_CORRECTION'
        WHERE id = ?
        """,
        (normalized_plate, normalized_plate, normalized_plate, detection_id),
    )
    conn.commit()

    for wrong_value in (previous_plate, row["ocr_corrected"], row["ocr_raw"]):
        if record_ocr_feedback(wrong_value, normalized_plate, source="dashboard_correction"):
            break

    return {
        "detection_id": int(row["id"]),
        "old_plate_number": previous_plate,
        "plate_number": normalized_plate,
    }


def get_flagged_detections(limit: int = 200) -> list:
    """Return detections that need guard review."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT
            d.id,
            d.timestamp,
            d.plate_number,
            d.ocr_raw,
            d.ocr_corrected,
            d.match_status,
            d.rfid_status,
            d.confidence,
            d.camera,
            d.image_path,
            COALESCE(rp.owner_name, '') AS owner_name
        FROM detections d
        LEFT JOIN registered_plates rp
            ON rp.plate_number = d.plate_number
        WHERE (
            UPPER(COALESCE(d.match_status, '')) IN ('NEEDS_REVIEW', 'NO_MATCH')
            OR UPPER(COALESCE(d.rfid_status, '')) = 'MISMATCH'
        )
          AND UPPER(COALESCE(d.match_status, '')) NOT LIKE 'REVIEWED_%'
        ORDER BY d.timestamp DESC, d.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def review_flagged_detection(
    detection_id: int,
    action: str,
    corrected_plate: str | None = None,
) -> dict | None:
    """
    Review a flagged detection.
    action: confirm | correct | reject
    """
    if not isinstance(detection_id, int) or detection_id <= 0:
        raise ValueError("detection_id must be a positive integer.")

    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"confirm", "correct", "reject"}:
        raise ValueError("action must be one of: confirm, correct, reject.")

    conn = _get_connection()
    row = conn.execute(
        """
        SELECT id, plate_number, ocr_raw, ocr_corrected
        FROM detections
        WHERE id = ?
        """,
        (detection_id,),
    ).fetchone()
    if row is None:
        return None

    previous_plate = _normalize_plate(row["plate_number"])

    if normalized_action == "confirm":
        conn.execute(
            """
            UPDATE detections
            SET match_status = 'REVIEWED_CONFIRMED'
            WHERE id = ?
            """,
            (detection_id,),
        )
        conn.commit()
        return {
            "detection_id": detection_id,
            "action": "confirm",
            "status": "REVIEWED",
            "plate_number": previous_plate,
        }

    if normalized_action == "reject":
        conn.execute(
            """
            UPDATE detections
            SET match_status = 'REVIEWED_REJECTED'
            WHERE id = ?
            """,
            (detection_id,),
        )
        conn.commit()
        return {
            "detection_id": detection_id,
            "action": "reject",
            "status": "REVIEWED",
            "plate_number": previous_plate,
        }

    normalized_plate = _normalize_plate(corrected_plate)
    if not normalized_plate or normalized_plate == "UNKNOWN":
        raise ValueError("corrected_plate is required for action=correct.")

    conn.execute(
        """
        UPDATE detections
        SET plate_number = ?,
            ocr_corrected = ?,
            plate_valid = 1,
            matched_plate = ?,
            match_score = 100.0,
            match_status = 'REVIEWED_CORRECTED'
        WHERE id = ?
        """,
        (normalized_plate, normalized_plate, normalized_plate, detection_id),
    )
    conn.commit()

    for wrong_value in (previous_plate, row["ocr_corrected"], row["ocr_raw"]):
        if record_ocr_feedback(wrong_value, normalized_plate, source="guard_review"):
            break

    return {
        "detection_id": detection_id,
        "action": "correct",
        "status": "REVIEWED",
        "old_plate_number": previous_plate,
        "plate_number": normalized_plate,
    }


def get_logbook_entries(
    date_from: str | None = None,
    date_to: str | None = None,
    direction: str | None = None,
    status: str | None = None,
    user: str | None = None,
    limit: int | None = 500,
) -> list:
    """Return logbook rows with Phase 3 fields and filters."""
    conn = _get_connection()

    status_case = (
        "CASE "
        "WHEN UPPER(COALESCE(d.match_status, '')) LIKE 'REVIEWED_%' THEN 'REVIEWED' "
        "WHEN UPPER(COALESCE(d.match_status, '')) IN ('NEEDS_REVIEW', 'NO_MATCH') "
        "     OR UPPER(COALESCE(d.rfid_status, '')) = 'MISMATCH' THEN 'FLAGGED' "
        "ELSE 'OK' END"
    )
    direction_case = (
        "CASE "
        "WHEN LOWER(COALESCE(d.camera, '')) = 'entrance' THEN 'ENTRY' "
        "WHEN LOWER(COALESCE(d.camera, '')) = 'exit' THEN 'EXIT' "
        "ELSE UPPER(COALESCE(d.camera, 'UNKNOWN')) END"
    )
    source_case = (
        "CASE "
        "WHEN UPPER(COALESCE(d.match_status, '')) IN "
        "('MANUAL_ENTRY', 'MANUAL_CORRECTION', 'REVIEWED_CONFIRMED', "
        " 'REVIEWED_CORRECTED', 'REVIEWED_REJECTED') "
        "THEN 'MANUAL' ELSE 'AUTO' END"
    )

    where_clauses: list[str] = []
    params: list[Any] = []

    if date_from:
        where_clauses.append("d.timestamp >= ?")
        params.append(f"{date_from} 00:00:00")

    if date_to:
        where_clauses.append("d.timestamp <= ?")
        params.append(f"{date_to} 23:59:59")

    direction_norm = str(direction or "").strip().upper()
    if direction_norm == "ENTRY":
        where_clauses.append("LOWER(COALESCE(d.camera, '')) = 'entrance'")
    elif direction_norm == "EXIT":
        where_clauses.append("LOWER(COALESCE(d.camera, '')) = 'exit'")

    status_norm = str(status or "").strip().upper()
    if status_norm in {"OK", "FLAGGED", "REVIEWED"}:
        where_clauses.append(f"{status_case} = ?")
        params.append(status_norm)

    user_query = str(user or "").strip()
    if user_query:
        where_clauses.append(
            "(COALESCE(rp.owner_name, '') LIKE ? OR d.plate_number LIKE ?)"
        )
        like = f"%{user_query}%"
        params.extend([like, like])

    sql = f"""
        SELECT
            d.id,
            d.timestamp,
            d.plate_number,
            d.ocr_raw,
            d.ocr_corrected,
            d.confidence,
            d.camera,
            d.rfid_status,
            d.match_status,
            d.image_path,
            COALESCE(rp.owner_name, '') AS owner_name,
            {direction_case} AS direction,
            {source_case} AS entry_source,
            {status_case} AS status
        FROM detections d
        LEFT JOIN registered_plates rp
            ON rp.plate_number = d.plate_number
    """

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += " ORDER BY d.timestamp DESC, d.id DESC"

    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def save_device_config(camera: str, device_index: int) -> None:
    """Persist camera -> device index assignment."""
    camera_name = str(camera or "").strip().lower()
    if camera_name not in {"entrance", "exit"}:
        raise ValueError("camera must be 'entrance' or 'exit'.")
    if not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer.")

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO device_config (camera, device_index, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(camera) DO UPDATE SET
            device_index = excluded.device_index,
            updated_at = CURRENT_TIMESTAMP
        """,
        (camera_name, int(device_index)),
    )
    conn.commit()


def get_device_config() -> dict[str, int]:
    """Return persisted camera assignments."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT camera, device_index
        FROM device_config
        WHERE camera IN ('entrance', 'exit')
        """
    ).fetchall()

    assignments: dict[str, int] = {}
    for row in rows:
        camera_name = str(row["camera"]).strip().lower()
        device_index = row["device_index"]
        if camera_name in {"entrance", "exit"} and isinstance(device_index, int):
            if device_index >= 0:
                assignments[camera_name] = int(device_index)
    return assignments


def discard_manual_input(manual_input_id: int) -> dict | None:
    """
    Discard a queued manual input so it no longer asks for manual entry.
    Returns summary with image_path for optional file cleanup, or None if missing.
    """
    conn = _get_connection()
    row = conn.execute(
        """
        SELECT id, image_path
        FROM manual_inputs
        WHERE id = ?
          AND resolved_at IS NULL
          AND COALESCE(NULLIF(TRIM(resolution_status), ''), 'PENDING') = 'PENDING'
        """,
        (manual_input_id,),
    ).fetchone()

    if row is None:
        return None

    resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        UPDATE manual_inputs
        SET resolved_at = ?,
            resolution_status = 'DISCARDED',
            resolved_plate = NULL
        WHERE id = ?
        """,
        (resolved_at, manual_input_id),
    )
    conn.commit()

    return {
        "manual_input_id": int(row["id"]),
        "image_path": str(row["image_path"]),
    }


def get_recent_detections(limit: int = 50) -> list:
    """Return the most recent detections (newest first)."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_detections_by_date(date_str: str) -> list:
    """
    Return all detections for a given date (YYYY-MM-DD).
    Uses a LIKE prefix match on the ISO timestamp.
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM detections WHERE timestamp LIKE ? ORDER BY timestamp DESC",
        (f"{date_str}%",),
    ).fetchall()
    return [dict(r) for r in rows]


def get_detections_by_camera(camera: str, limit: int = 100) -> list:
    """Return detections filtered by camera name."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM detections WHERE camera = ? ORDER BY timestamp DESC LIMIT ?",
        (camera, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    """Return quick summary statistics for the dashboard."""
    conn = _get_connection()
    today = datetime.now().strftime("%Y-%m-%d")
    total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    today_count = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (f"{today}%",)
    ).fetchone()[0]
    entrance_count = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE camera = 'entrance'"
    ).fetchone()[0]
    exit_count = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE camera = 'exit'"
    ).fetchone()[0]
    rfid_pending = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE rfid_status = 'NOT_SCANNED'"
    ).fetchone()[0]
    rfid_match = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE rfid_status = 'MATCH'"
    ).fetchone()[0]
    rfid_mismatch = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE rfid_status = 'MISMATCH'"
    ).fetchone()[0]
    return {
        "total": total,
        "today": today_count,
        "entrance": entrance_count,
        "exit": exit_count,
        "rfid_pending": rfid_pending,
        "rfid_match": rfid_match,
        "rfid_mismatch": rfid_mismatch,
    }


# Auto-initialise on import so the table is always ready
init_db()
