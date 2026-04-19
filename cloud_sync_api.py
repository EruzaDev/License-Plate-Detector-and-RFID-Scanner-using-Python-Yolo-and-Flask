"""
cloud_sync_api.py - Thin Phase 4 cloud mirror API.

Endpoints:
- GET  /health
- POST /sync/logs
- GET  /logs

Run locally:
    uvicorn cloud_sync_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import sqlite3
import importlib
from typing import Any

_fastapi = importlib.import_module("fastapi")
_pydantic = importlib.import_module("pydantic")

FastAPI = getattr(_fastapi, "FastAPI")
Header = getattr(_fastapi, "Header")
HTTPException = getattr(_fastapi, "HTTPException")
Query = getattr(_fastapi, "Query")
BaseModel = getattr(_pydantic, "BaseModel")
Field = getattr(_pydantic, "Field")

CLOUD_SYNC_DB_PATH = os.getenv("CLOUD_SYNC_DB_PATH", "cloud_sync.db")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "").strip()

app = FastAPI(title="LPR Cloud Sync API", version="0.1.0")


class CloudLogEntry(BaseModel):
    device_log_id: str
    local_detection_id: int | None = None
    device_id: str | None = None
    timestamp: str | None = None
    plate_number: str | None = None
    camera: str | None = None
    image_path: str | None = None
    confidence: float | None = None
    ocr_raw: str | None = None
    ocr_corrected: str | None = None
    match_status: str | None = None
    rfid_status: str | None = None
    expected_rfid_uid: str | None = None
    scanned_rfid_uid: str | None = None
    rfid_verified_at: str | None = None


class SyncLogsRequest(BaseModel):
    device_id: str | None = None
    logs: list[CloudLogEntry] = Field(default_factory=list)


class SyncLogsResponse(BaseModel):
    ok: bool
    received: int
    upserted: int
    synced_device_log_ids: list[str]


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(CLOUD_SYNC_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cloud_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_log_id TEXT NOT NULL UNIQUE,
            device_id TEXT,
            local_detection_id INTEGER,
            timestamp TEXT,
            plate_number TEXT,
            camera TEXT,
            image_path TEXT,
            confidence REAL,
            ocr_raw TEXT,
            ocr_corrected TEXT,
            match_status TEXT,
            rfid_status TEXT,
            expected_rfid_uid TEXT,
            scanned_rfid_uid TEXT,
            rfid_verified_at TEXT,
            raw_payload TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cloud_logs_timestamp
        ON cloud_logs (timestamp DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cloud_logs_device
        ON cloud_logs (device_id)
        """
    )
    conn.commit()
    conn.close()


def _check_api_key(x_api_key: str | None) -> None:
    if CLOUD_API_KEY and x_api_key != CLOUD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


_init_db()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "lpr-cloud-sync-api",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "lpr-cloud-sync-api",
        "db": CLOUD_SYNC_DB_PATH,
    }


@app.post("/sync/logs", response_model=SyncLogsResponse)
def sync_logs(payload: SyncLogsRequest, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)

    logs = payload.logs or []
    if not logs:
        return SyncLogsResponse(ok=True, received=0, upserted=0, synced_device_log_ids=[])

    conn = _get_connection()
    synced_ids: list[str] = []

    for log in logs:
        item = _model_dump(log)
        raw_payload = json.dumps(item, ensure_ascii=True, separators=(",", ":"))

        conn.execute(
            """
            INSERT INTO cloud_logs (
                device_log_id,
                device_id,
                local_detection_id,
                timestamp,
                plate_number,
                camera,
                image_path,
                confidence,
                ocr_raw,
                ocr_corrected,
                match_status,
                rfid_status,
                expected_rfid_uid,
                scanned_rfid_uid,
                rfid_verified_at,
                raw_payload,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(device_log_id) DO UPDATE SET
                device_id = excluded.device_id,
                local_detection_id = excluded.local_detection_id,
                timestamp = excluded.timestamp,
                plate_number = excluded.plate_number,
                camera = excluded.camera,
                image_path = excluded.image_path,
                confidence = excluded.confidence,
                ocr_raw = excluded.ocr_raw,
                ocr_corrected = excluded.ocr_corrected,
                match_status = excluded.match_status,
                rfid_status = excluded.rfid_status,
                expected_rfid_uid = excluded.expected_rfid_uid,
                scanned_rfid_uid = excluded.scanned_rfid_uid,
                rfid_verified_at = excluded.rfid_verified_at,
                raw_payload = excluded.raw_payload,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                item.get("device_log_id"),
                item.get("device_id") or payload.device_id,
                item.get("local_detection_id"),
                item.get("timestamp"),
                item.get("plate_number"),
                item.get("camera"),
                item.get("image_path"),
                item.get("confidence"),
                item.get("ocr_raw"),
                item.get("ocr_corrected"),
                item.get("match_status"),
                item.get("rfid_status"),
                item.get("expected_rfid_uid"),
                item.get("scanned_rfid_uid"),
                item.get("rfid_verified_at"),
                raw_payload,
            ),
        )
        synced_ids.append(str(item.get("device_log_id")))

    conn.commit()
    conn.close()

    return SyncLogsResponse(
        ok=True,
        received=len(logs),
        upserted=len(synced_ids),
        synced_device_log_ids=synced_ids,
    )


@app.get("/logs")
def logs(
    limit: int = Query(default=100, ge=1, le=2000),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT
            id,
            device_log_id,
            device_id,
            local_detection_id,
            timestamp,
            plate_number,
            camera,
            confidence,
            match_status,
            rfid_status,
            updated_at
        FROM cloud_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()

    return {
        "ok": True,
        "count": len(rows),
        "logs": [dict(r) for r in rows],
    }


if __name__ == "__main__":
    uvicorn = importlib.import_module("uvicorn")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("cloud_sync_api:app", host="0.0.0.0", port=port)
