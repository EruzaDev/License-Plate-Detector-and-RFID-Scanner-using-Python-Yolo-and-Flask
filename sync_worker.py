"""
sync_worker.py - Offline-first cloud sync loop for local detections.

Phase 4 behavior:
- Keep writing detections locally regardless of network state.
- Periodically check cloud health endpoint.
- Push PENDING/FAILED rows to cloud /sync/logs.
- Mark rows as SYNCED on success, FAILED on submission errors.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from datetime import datetime
from typing import Any

import requests

from database import (
    get_pending_sync_detections,
    mark_detections_sync_attempted,
    mark_detections_synced,
    mark_detections_sync_failed,
    get_sync_status_counts,
)


class CloudSyncWorker:
    """Background worker that syncs local detections to a cloud API."""

    def __init__(self):
        base_url = os.getenv("CLOUD_API_BASE_URL", "").strip().rstrip("/")
        self.cloud_api_base_url = base_url
        self.cloud_api_key = os.getenv("CLOUD_API_KEY", "").strip()
        self.device_id = os.getenv("DEVICE_ID", socket.gethostname()).strip() or socket.gethostname()

        self.sync_interval_seconds = max(
            5,
            int(float(os.getenv("SYNC_INTERVAL_SECONDS", "60"))),
        )
        self.sync_batch_size = max(
            1,
            int(float(os.getenv("SYNC_BATCH_SIZE", "100"))),
        )
        self.http_timeout_seconds = max(
            1,
            int(float(os.getenv("SYNC_HTTP_TIMEOUT_SECONDS", "5"))),
        )

        include_failed = os.getenv("SYNC_INCLUDE_FAILED", "1").strip().lower()
        self.sync_include_failed = include_failed not in {"0", "false", "no", "off"}

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._last_run_at: str | None = None
        self._last_success_at: str | None = None
        self._last_error: str | None = None
        self._last_result: dict[str, Any] = {
            "synced": 0,
            "failed": 0,
            "pending_checked": 0,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.cloud_api_base_url)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the worker thread if cloud sync is enabled."""
        if not self.enabled:
            return False
        if self.running:
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="cloud-sync-worker", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()

    def status(self) -> dict[str, Any]:
        """Return runtime and local queue status."""
        with self._lock:
            result = dict(self._last_result)
            return {
                "enabled": self.enabled,
                "running": self.running,
                "device_id": self.device_id,
                "cloud_api_base_url": self.cloud_api_base_url or None,
                "sync_interval_seconds": self.sync_interval_seconds,
                "sync_batch_size": self.sync_batch_size,
                "sync_include_failed": self.sync_include_failed,
                "last_run_at": self._last_run_at,
                "last_success_at": self._last_success_at,
                "last_error": self._last_error,
                "last_result": result,
                "local_counts": get_sync_status_counts(),
            }

    def sync_once(self) -> dict[str, Any]:
        """Run one sync cycle immediately."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._last_run_at = now

        if not self.enabled:
            result = {
                "ok": False,
                "reason": "disabled",
                "synced": 0,
                "failed": 0,
                "pending_checked": 0,
            }
            with self._lock:
                self._last_error = "Cloud sync is disabled (CLOUD_API_BASE_URL not set)."
                self._last_result = dict(result)
            return result

        health_url = f"{self.cloud_api_base_url}/health"
        try:
            health_res = requests.get(health_url, timeout=self.http_timeout_seconds)
            if not health_res.ok:
                message = f"Cloud health check failed ({health_res.status_code})."
                with self._lock:
                    self._last_error = message
                    self._last_result = {
                        "ok": False,
                        "reason": "health",
                        "synced": 0,
                        "failed": 0,
                        "pending_checked": 0,
                    }
                return dict(self._last_result)
        except requests.RequestException as exc:
            message = f"Cloud offline: {exc}"
            with self._lock:
                self._last_error = message
                self._last_result = {
                    "ok": False,
                    "reason": "offline",
                    "synced": 0,
                    "failed": 0,
                    "pending_checked": 0,
                }
            return dict(self._last_result)

        pending_rows = get_pending_sync_detections(
            limit=self.sync_batch_size,
            include_failed=self.sync_include_failed,
        )
        if not pending_rows:
            result = {
                "ok": True,
                "reason": "empty",
                "synced": 0,
                "failed": 0,
                "pending_checked": 0,
            }
            with self._lock:
                self._last_error = None
                self._last_success_at = now
                self._last_result = dict(result)
            return result

        local_ids = [int(row["id"]) for row in pending_rows]
        mark_detections_sync_attempted(local_ids)

        payload_logs = [self._to_cloud_payload(row) for row in pending_rows]
        headers = {"Content-Type": "application/json"}
        if self.cloud_api_key:
            headers["X-API-Key"] = self.cloud_api_key

        sync_url = f"{self.cloud_api_base_url}/sync/logs"
        try:
            sync_res = requests.post(
                sync_url,
                json={"device_id": self.device_id, "logs": payload_logs},
                headers=headers,
                timeout=self.http_timeout_seconds,
            )
            if not sync_res.ok:
                body = sync_res.text.strip()
                if len(body) > 300:
                    body = body[:300]
                raise RuntimeError(f"Cloud sync failed ({sync_res.status_code}): {body}")

            response_json = sync_res.json() if sync_res.content else {}
            synced_device_log_ids = response_json.get("synced_device_log_ids") or []

            if isinstance(synced_device_log_ids, list) and synced_device_log_ids:
                synced_id_set = {str(value) for value in synced_device_log_ids}
                synced_local_ids = [
                    int(row["id"])
                    for row in pending_rows
                    if self._device_log_id(int(row["id"])) in synced_id_set
                ]
            else:
                synced_local_ids = list(local_ids)

            failed_local_ids = [d_id for d_id in local_ids if d_id not in set(synced_local_ids)]
            if synced_local_ids:
                mark_detections_synced(synced_local_ids)
            if failed_local_ids:
                mark_detections_sync_failed(
                    failed_local_ids,
                    "Cloud response omitted synced marker for these records.",
                )

            result = {
                "ok": True,
                "reason": "synced",
                "synced": len(synced_local_ids),
                "failed": len(failed_local_ids),
                "pending_checked": len(local_ids),
            }
            with self._lock:
                self._last_error = None
                self._last_success_at = now
                self._last_result = dict(result)
            return result

        except Exception as exc:
            mark_detections_sync_failed(local_ids, str(exc))
            result = {
                "ok": False,
                "reason": "sync_error",
                "synced": 0,
                "failed": len(local_ids),
                "pending_checked": len(local_ids),
            }
            with self._lock:
                self._last_error = str(exc)
                self._last_result = dict(result)
            return result

    def _loop(self) -> None:
        """Background loop that syncs on a fixed interval."""
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception as exc:  # safety net to keep thread alive
                with self._lock:
                    self._last_error = f"sync loop error: {exc}"
            self._stop_event.wait(self.sync_interval_seconds)

    def _device_log_id(self, detection_id: int) -> str:
        return f"{self.device_id}:{int(detection_id)}"

    def _to_cloud_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        detection_id = int(row["id"])
        return {
            "device_log_id": self._device_log_id(detection_id),
            "local_detection_id": detection_id,
            "device_id": self.device_id,
            "timestamp": row.get("timestamp"),
            "plate_number": row.get("plate_number"),
            "camera": row.get("camera"),
            "image_path": row.get("image_path"),
            "confidence": row.get("confidence"),
            "ocr_raw": row.get("ocr_raw"),
            "ocr_corrected": row.get("ocr_corrected"),
            "match_status": row.get("match_status"),
            "rfid_status": row.get("rfid_status"),
            "expected_rfid_uid": row.get("expected_rfid_uid"),
            "scanned_rfid_uid": row.get("scanned_rfid_uid"),
            "rfid_verified_at": row.get("rfid_verified_at"),
        }
