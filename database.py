"""
database.py — SQLite setup and queries for the LPR system.
Stores every plate detection with metadata and provides query helpers.
"""

import sqlite3
import os
import threading
import re
from datetime import datetime

# Path to the database file (sits next to this script)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lpr_system.db")

# Thread-local storage so each thread gets its own connection
_local = threading.local()
_PLATE_SANITIZER = re.compile(r"[^A-Z0-9]")


def _normalize_plate(plate: str | None) -> str:
    """Normalize a plate string for consistent storage and matching."""
    if not plate:
        return ""
    return _PLATE_SANITIZER.sub("", str(plate).upper())


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
            match_status TEXT DEFAULT 'NO_MATCH'
        )
    """)

    # Backward-compatible migration for older DB files that predate Phase 1.
    _ensure_column(conn, "detections", "ocr_raw TEXT")
    _ensure_column(conn, "detections", "ocr_corrected TEXT")
    _ensure_column(conn, "detections", "plate_valid INTEGER DEFAULT 0")
    _ensure_column(conn, "detections", "matched_plate TEXT")
    _ensure_column(conn, "detections", "match_score REAL")
    _ensure_column(conn, "detections", "match_status TEXT DEFAULT 'NO_MATCH'")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS registered_plates (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL UNIQUE,
            owner_name   TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

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
    conn.commit()


def insert_detection(plate_number: str, camera: str, timestamp: str,
                     image_path: str, confidence: float,
                     ocr_raw: str | None = None,
                     ocr_corrected: str | None = None,
                     plate_valid: bool | None = None,
                     matched_plate: str | None = None,
                     match_score: float | None = None,
                     match_status: str | None = None) -> int:
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
               match_status
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        ),
    )
    conn.commit()
    return cur.lastrowid


def register_plate(plate_number: str, owner_name: str | None = None) -> bool:
    """
    Register or update a known plate for fuzzy matching.
    Returns False when the plate is empty after normalization.
    """
    normalized = _normalize_plate(plate_number)
    if not normalized:
        return False

    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO registered_plates (plate_number, owner_name)
        VALUES (?, ?)
        ON CONFLICT(plate_number) DO UPDATE SET
            owner_name = COALESCE(excluded.owner_name, registered_plates.owner_name)
        """,
        (normalized, owner_name),
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

    return {
        "manual_input_id": manual_input_id,
        "plate_number": normalized_plate,
        "detection_id": detection_id,
    }


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
    return {
        "total": total,
        "today": today_count,
        "entrance": entrance_count,
        "exit": exit_count,
    }


# Auto-initialise on import so the table is always ready
init_db()
