# Campus Vehicle & Personnel Entry Logbook System
### Project Planning Document · Raspberry Pi 5 · YOLOv8n + EasyOCR + RFID

---

## Overview

A Raspberry Pi 5–based dual-camera entry/exit logging system for campus vehicle management. The system uses **YOLOv8n** for vehicle detection and license plate localization, **EasyOCR** for OCR, and an **RFID reader** for secondary identity verification. A web interface (Flask backend) provides role-gated access for a Superadmin, Guards, and registered Users. An offline-first sync architecture keeps the system resilient to network outages.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 5                      │
│                                                     │
│  Camera 0 ──► YOLOv8n ──► EasyOCR ──► Flask API    │
│  Camera 1 ──► YOLOv8n ──► EasyOCR ──►  (local)     │
│  RFID Reader ────────────────────────►              │
│                                                     │
│  SQLite (local) ◄──── Sync Agent ────► Cloud DB     │
└─────────────────────────────────────────────────────┘
         │
         ▼
    Web Dashboard (served locally + cloud mirror)
```

---

## Current Code-Verified Completion (2026-04-19, OCR model excluded)

- [x] Flask dashboard with live entrance and exit MJPEG streams
- [x] Camera device scan, assignment, and stop controls (UI + API)
- [x] Motion-triggered YOLO plate detection pipeline with tracking and capture cooldown
- [x] SQLite detection logging with recent/date-filter queries and stats API
- [x] Capture thumbnail serving and dashboard table rendering
- [x] Model bootstrap script for detector weights and EasyOCR assets

---

## Phase 0 — Hardware & Environment Setup

**Goal:** Stable, reproducible dev environment on the RPi5.

- [ ] Flash Raspberry Pi OS (64-bit) and configure SSH, static IP
- [x] Install Python 3.11+, pip, virtualenv
- [x] Install **Ultralytics YOLOv8**, **EasyOCR**, **OpenCV**, **Flask**
- [ ] Wire dual USB cameras; confirm `/dev/video0` and `/dev/video1` are stable
- [ ] Wire RFID reader (determine interface: USB HID, UART, or SPI)
- [x] Verify camera isolation — each camera must be independently addressable
- [x] Set up SQLite database file (`logbook.db`) with initial schema
- [ ] Create a `.env` config file for camera indices, RFID port, cloud API URL, sync interval

**Notes:**
- Keep camera index assignments in config, not hardcoded — this is what the guard's "camera assignment" UI will write to.
- YOLOv8n is chosen for speed on the RPi5's CPU; consider exporting to ONNX or TorchScript if inference is too slow (reference: Polinowski achieves ~6ms inference on GPU; expect ~100–400ms on RPi5 CPU).

---

## Phase 1 — Core Vision Pipeline

**Goal:** Reliable license plate text extraction from video frames.

### 1.1 Vehicle Detection (Stage 1 Model)
Using the pre-trained YOLOv8n COCO model to detect vehicles (class IDs: `2` car, `3` motorcycle, `5` bus, `7` truck).

```python
from ultralytics import YOLO
coco_model = YOLO('yolov8n.pt')
vehicles = [2, 3, 5, 7]
detections = coco_model.track(frame, persist=True)[0]
```

### 1.2 License Plate Detection (Stage 2 Model)
Fine-tuned YOLOv8n model trained on a license plate dataset (e.g., [Roboflow License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)).

- Train for the Philippine LTO license plate format (7-character alphanumeric: `AAA 0000`)
- The model runs only within the vehicle bounding box ROI — reduces false positives significantly
- Export to TorchScript for RPi5: `model.export(format='torchscript')`

### 1.3 Image Preprocessing for OCR
Before feeding the cropped plate to EasyOCR:

```python
plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
_, plate_thresh = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
```

Consider also: resizing to a standard height (e.g., 60px), dilation to thicken characters.

### 1.4 OCR with EasyOCR

```python
import easyocr
reader = easyocr.Reader(['en'], gpu=False)  # RPi5 has no GPU
detections = reader.readtext(plate_thresh)
```

### 1.5 Philippine LTO Plate Format Validation & Cleanup
PH plates follow the format: **3 letters + space + 4 digits** (e.g., `ABC 1234`) or newer **3 letters + 4 digits** (`ABC1234`). Classic plates: `AB 1234`.

Implement a character disambiguation map (common OCR misreads):

```python
char_to_int = {'O': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8', 'Z': '2'}
int_to_char = {v: k for k, v in char_to_int.items()}
```

Then apply position-aware correction: positions 0–2 should be letters, positions 3–6 should be digits.

### 1.6 Fuzzy Plate Matching (Anti-Misread Safeguard)
> **This is critical for production safety.**

When a detected plate doesn't exactly match any registered plate, compute the **Levenshtein distance** between the OCR result and all plates in the database.

```python
from rapidfuzz import fuzz, process

def match_plate(ocr_result, registered_plates, threshold=85):
    best_match, score, _ = process.extractOne(
        ocr_result, registered_plates, scorer=fuzz.ratio
    )
    if score >= threshold:
        return best_match, score, "AUTO_MATCHED"
    elif score >= 60:
        return best_match, score, "NEEDS_REVIEW"   # flag for guard review
    else:
        return None, score, "NO_MATCH"
```

**Behavior by confidence tier:**

| OCR Score | Levenshtein Similarity | Action |
|---|---|---|
| High (>0.85) | ≥85% | Auto-accept, log normally |
| Medium | 60–84% | Flag entry as **"Uncertain — Needs Guard Review"** |
| Low | <60% | Reject auto-match; prompt manual plate entry |

This prevents a plate like `ABC 1234` from being accidentally logged under `ABC 1254`.

---

## Phase 2 — RFID Verification Layer

**Goal:** Use RFID as a 2FA layer to confirm the person driving matches the vehicle.

- Each registered user is issued an RFID card/tag
- On vehicle approach:
  1. Camera detects and reads the license plate
  2. System identifies the vehicle owner from the database
  3. RFID reader prompts the driver to scan their card
  4. System verifies that the scanned RFID UID matches the vehicle owner's registered UID
  5. If both match → **ACCESS GRANTED**, log entry
  6. If plate matches but RFID doesn't → **FLAG as suspicious**, log with warning
  7. If plate unrecognized → **DENY or escalate to guard**

```
Vehicle Detected
     │
     ▼
License Plate OCR ──► Match against DB
     │
     ├── Match found ──► Prompt RFID scan
     │                        │
     │                        ├── RFID matches owner ──► LOG ENTRY ✓
     │                        └── RFID mismatch ──────► FLAG + GUARD ALERT ⚠
     │
     └── No match ──────────────────────────────────────► GUARD REVIEW ⚠
```

**RFID Interface Options (in order of preference for RPi5):**
- USB HID mode (plug-and-play, read as keyboard input via `evdev`)
- UART serial (MFRC522-compatible readers)
- SPI (RC522 module, requires `spidev`)

---

## Phase 3 — Web Application

**Goal:** Full-featured web dashboard served by Flask on the RPi5 LAN, with a cloud mirror.

### 3.1 Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Backend | Flask + SQLAlchemy | Lightweight, fits RPi5 |
| Frontend | Jinja2 + HTMX or plain JS | No build step needed |
| Database | SQLite (local) → PostgreSQL (cloud) | SQLite zero-config on device |
| Auth | Flask-Login + bcrypt | Simple session-based auth |
| Offline sync | Background thread / cron job | Push local → cloud on reconnect |
| Cloud API | FastAPI on Railway/Render | Thin REST layer for sync |

### 3.2 User Roles & Permissions

| Feature | Superadmin | Guard | User |
|---|---|---|---|
| Login / Logout | ✓ | ✓ | ✓ |
| View own logbook entries | ✓ | ✓ | ✓ (own only) |
| View full logbook | ✓ | ✓ | ✗ |
| Create / Edit / Delete users | ✓ | ✗ | ✗ |
| Create / Edit / Delete guards | ✓ | ✗ | ✗ |
| Assign cameras (Entry/Exit) | ✓ | ✓ | ✗ |
| Manual log entry | ✓ | ✓ | ✗ |
| Review flagged entries | ✓ | ✓ | ✗ |
| Export logs (CSV/PDF) | ✓ | ✓ | ✗ |
| System settings | ✓ | ✗ | ✗ |

### 3.3 Authentication Flow

- Login page (`/login`) — email/username + password
- Session-based auth with role stored in session
- Logout clears session (`/logout`)
- All routes protected by a `@login_required` + `@role_required` decorator
- Password hashing with `bcrypt`
- (Optional) Default Superadmin seeded on first run

### 3.4 Superadmin Panel

**Users Management (`/admin/users`)**
- Table view: Name, Email, Role, Associated Plate, RFID UID, Status (Active/Inactive)
- Create user: form with fields for name, email, password, role, plate number (manual or OCR-assisted), RFID UID
- Edit user: all fields editable
- Delete user: soft delete (set `is_active = False`), not hard delete for log integrity
- Search and filter

**Guards Management (`/admin/guards`)**
- Same CRUD interface as users, but role is locked to `guard`

### 3.5 Guard Panel

**Camera Assignment (`/guard/device`)**
- Simple UI: two dropdowns or toggle buttons
  - "Entry Camera" → select `Camera 0` or `Camera 1`
  - "Exit Camera" → the remaining one is auto-assigned
- Writes selection to `.env` or a `device_config` table in SQLite
- Live camera preview (MJPEG stream) to confirm the correct camera is selected

**Flagged Entries Review (`/guard/review`)**
- Table of all entries with status `NEEDS_REVIEW`
- Guard can: Confirm match, Correct plate manually, Reject entry
- After action, entry status changes to `REVIEWED`

### 3.6 Logbook (`/logbook`)

Comprehensive log table with:
- Timestamp (entry/exit)
- Vehicle plate (OCR result + corrected value if reviewed)
- Vehicle owner name
- Direction (ENTRY / EXIT)
- Camera used
- RFID match status
- OCR confidence score
- Entry source (`AUTO` / `MANUAL`)
- Status (`OK` / `FLAGGED` / `REVIEWED`)

**Filters:** Date range, direction, status, user

**Export:**
- CSV export (all visible rows)
- PDF export (formatted report with header, filters applied)

### 3.7 Vehicle Registration — Plate Entry Modes

When creating or editing a user's vehicle record, two modes for entering the plate number:

**Manual Mode:**
- Text input field
- Guard or admin types the plate directly
- Basic format validation (PH LTO format regex)

**OCR-Assisted Mode:**
- Trigger live capture from the assigned Entry or Exit camera
- System runs the full vision pipeline (YOLOv8n → EasyOCR → format correction)
- Displays the captured plate image + OCR result
- Guard/admin confirms or edits the OCR result before saving
- Useful for on-the-spot vehicle enrollment at the gate

---

## Phase 4 — Offline-First Sync Architecture

**Goal:** The system must continue logging without internet. Data syncs to the cloud when connectivity is restored.

### 4.1 Local Storage (Always-On)
- All logs write to SQLite on the RPi5 regardless of connectivity
- SQLite is the single source of truth for the device

### 4.2 Sync Table Schema
Each log entry has a `sync_status` field:

```sql
CREATE TABLE log_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ...
    sync_status TEXT DEFAULT 'PENDING'  -- PENDING | SYNCED | FAILED
);
```

### 4.3 Sync Agent
A background thread (or `cron` job every N minutes) that:

```
1. Check internet connectivity (ping cloud API health endpoint)
2. If online:
   a. Query all rows WHERE sync_status = 'PENDING'
   b. POST batch to Cloud API endpoint
   c. If 200 OK → UPDATE sync_status = 'SYNCED'
   d. If error → UPDATE sync_status = 'FAILED', retry next cycle
3. If offline: skip, continue local operation
```

```python
import threading, time, requests

def sync_worker():
    while True:
        try:
            r = requests.get(CLOUD_API + '/health', timeout=3)
            if r.ok:
                pending = db.session.query(LogEntry).filter_by(sync_status='PENDING').all()
                # batch POST pending records...
        except requests.exceptions.ConnectionError:
            pass  # offline, skip
        time.sleep(SYNC_INTERVAL_SECONDS)

t = threading.Thread(target=sync_worker, daemon=True)
t.start()
```

### 4.4 Cloud API (FastAPI on Railway / Render)
A thin REST API — **not** a full application, just a sync target:

```
POST /sync/logs        — receives batch of log entries, upserts into cloud DB
GET  /health           — used by RPi5 to check connectivity
GET  /logs             — optional: read-only view for remote monitoring
```

- Cloud DB: PostgreSQL (free tier on Railway/Neon)
- Auth: static API key in request header (`X-API-Key`)
- Upsert by `device_log_id` to handle duplicate submissions safely

### 4.5 Conflict Resolution
Since the RPi5 is the only writer (cloud API only receives), conflicts are minimal. The rule is simple:

> **The local device is always the source of truth. Cloud is read-only aggregate.**

---

## Phase 5 — Integration Testing & Hardening

**Goal:** End-to-end validation before deployment.

- [ ] Test camera swap (reassign camera 0↔1) without restarting Flask
- [ ] Test RFID mismatch flow — confirm guard alert triggers
- [ ] Test fuzzy matching: deliberately scan a plate 1–2 chars off, verify it lands in `NEEDS_REVIEW`
- [ ] Test offline mode: unplug network, log 10 entries, reconnect, verify all sync
- [ ] Test simultaneous entry + exit (both cameras active at the same time)
- [ ] Test PDF and CSV export with 500+ rows
- [ ] Stress test OCR pipeline: measure latency at different lighting conditions
- [ ] Security: verify non-admin users cannot access `/admin/*` routes
- [ ] Verify soft-delete: deleted users' historical log entries are preserved

---

## Phase 6 — Deployment

- [ ] Set Flask to run on boot via `systemd` service
- [ ] Configure Nginx as reverse proxy on RPi5 LAN (`http://192.168.x.x`)
- [ ] Deploy cloud sync API to Railway (or Render free tier)
- [ ] Set environment variables on cloud platform (DB URL, API key)
- [ ] Document guard's quickstart: camera assignment, manual log, flagged review
- [ ] Create a one-page operations guide for the security office

---

## Database Schema (Simplified)

```sql
-- Users (includes guards, managed by superadmin)
CREATE TABLE users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,  -- 'superadmin' | 'guard' | 'user'
    rfid_uid        TEXT,
    is_active       INTEGER DEFAULT 1,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Vehicles (one or more per user)
CREATE TABLE vehicles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES users(id),
    plate_number    TEXT NOT NULL,
    plate_image     TEXT,  -- path to stored enrollment image
    registered_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Log Entries
CREATE TABLE log_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id      INTEGER REFERENCES vehicles(id),
    user_id         INTEGER REFERENCES users(id),
    direction       TEXT NOT NULL,      -- 'ENTRY' | 'EXIT'
    camera_used     TEXT NOT NULL,      -- 'camera_0' | 'camera_1'
    ocr_raw         TEXT,               -- raw OCR string before correction
    ocr_corrected   TEXT,               -- post-format-correction string
    ocr_confidence  REAL,
    rfid_match      TEXT,               -- 'MATCH' | 'MISMATCH' | 'NOT_SCANNED'
    entry_source    TEXT DEFAULT 'AUTO',-- 'AUTO' | 'MANUAL'
    status          TEXT DEFAULT 'OK',  -- 'OK' | 'FLAGGED' | 'REVIEWED'
    reviewed_by     INTEGER REFERENCES users(id),
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    sync_status     TEXT DEFAULT 'PENDING'
);

-- Device Config
CREATE TABLE device_config (
    key             TEXT PRIMARY KEY,
    value           TEXT
);
-- e.g., key='entry_camera', value='0'
--       key='exit_camera',  value='1'
```

---

## Key Technical Decisions & Rationale

| Decision | Choice | Why |
|---|---|---|
| OCR engine | EasyOCR | Better accuracy than Tesseract on license plates; GPU optional |
| Plate fuzzy match | RapidFuzz (Levenshtein) | Fast, pure Python, prevents false identity matches |
| YOLO model size | YOLOv8n | Fastest inference on RPi5 CPU |
| Local DB | SQLite | Zero-config, file-based, survives power loss |
| Sync strategy | Device-first, cloud-aggregate | Keeps system functional regardless of connectivity |
| Web framework | Flask | Minimal overhead, easy to run as a service on RPi5 |
| Auth | Session-based (Flask-Login) | No token complexity, appropriate for LAN app |

---

## Open Questions / Decisions to Finalize

1. **RFID hardware**: USB HID (simplest) or SPI RC522 (already wired)?
2. **Plate format**: Support both old (2-letter) and new (3-letter) PH LTO formats in the validator?
3. **Multi-vehicle users**: Can one user register more than one vehicle? (Recommended: yes, 1:many)
4. **Camera stream**: MJPEG over HTTP or direct display? (MJPEG is easier for web dashboard)
5. **Notification**: When a flagged entry occurs, should an alert email/SMS be sent to the guard on duty?
6. **Log retention**: How long are entries kept locally vs. cloud-only archived?

---

*Document version: 0.1 — April 2026*