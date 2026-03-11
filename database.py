"""
database.py — SQLite setup and queries for the LPR system.
Stores every plate detection with metadata and provides query helpers.
"""

import sqlite3
import os
import threading
from datetime import datetime

# Path to the database file (sits next to this script)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lpr_system.db")

# Thread-local storage so each thread gets its own connection
_local = threading.local()


def _get_connection():
    """Return a thread-local SQLite connection (created on first call)."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row  # dict-like access
        _local.conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    return _local.conn


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
            confidence   REAL    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_timestamp
        ON detections (timestamp DESC)
    """)
    conn.commit()


def insert_detection(plate_number: str, camera: str, timestamp: str,
                     image_path: str, confidence: float) -> int:
    """
    Insert a new detection record.
    Returns the new row id.
    """
    conn = _get_connection()
    cur = conn.execute(
        """INSERT INTO detections (plate_number, camera, timestamp, image_path, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (plate_number, camera, timestamp, image_path, confidence),
    )
    conn.commit()
    return cur.lastrowid


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
