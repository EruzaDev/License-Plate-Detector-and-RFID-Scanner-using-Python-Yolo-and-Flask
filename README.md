# LPR System — Raspberry Pi 5 License Plate Recognition

Dual-camera license plate recognition system using a dedicated **YOLOv8
license plate detector** and **EasyOCR** for text reading, running on a
Raspberry Pi 5.

## Features

- **Dual USB webcams** — entrance (`/dev/video0`) and exit (`/dev/video1`)
- **Motion detection** (MOG2) → **YOLOv8 plate detection** → **EasyOCR plate reading**
- **RFID verification layer** with pending scan prompts and match/mismatch status
- YOLO bounding boxes drawn on the live MJPEG stream
- 10-second cooldown prevents duplicate captures of the same plate
- Unknown OCR results are queued for required manual plate input (not logged as UNKNOWN)
- Pending manual captures can be explicitly discarded from the dashboard prompt
- Wrong detections can be corrected from the dashboard; corrections are learned and reused by OCR
- Role-based login for `superadmin` and `guard` sessions
- Dedicated guard camera-assignment page with saved device mapping
- Dedicated flagged-entry review workflow (confirm / correct / reject)
- Comprehensive logbook with filters and CSV/PDF export
- SQLite database logs every detection
- Dark-themed Flask dashboard with live feeds, stats, and date filtering

## Hardware

| Component       | Detail                   |
|-----------------|--------------------------|
| Board           | Raspberry Pi 5           |
| Entrance camera | USB webcam → /dev/video0 |
| Exit camera     | USB webcam → /dev/video1 |

## Project Structure

```
lpr_system/
├── camera_system.py     # motion detection + plate detection + capture
├── ocr_processor.py     # EasyOCR plate recognition + image preprocessing
├── database.py          # SQLite setup and queries
├── app.py               # Flask web server + dashboard
├── sync_worker.py       # background local -> cloud sync loop (Phase 4)
├── cloud_sync_api.py    # thin FastAPI cloud mirror service
├── download_models.py   # one-time model download script
├── templates/
│   ├── dashboard.html   # main operations dashboard
│   ├── guard_device.html
│   ├── guard_review.html
│   └── logbook.html
├── models/
│   └── license_plate_detector.pt  # auto-downloaded by download_models.py
├── captures/            # saved plate images
├── requirements.txt
└── README.md
```

## Setup Instructions (Raspberry Pi 5)

### 1. System dependencies

```bash
sudo apt update
sudo apt install -y libopenblas-dev libatlas-base-dev python3-pip
```

### 2. Python packages

```bash
pip install ultralytics lapx easyocr opencv-python flask numpy --break-system-packages
```

Or using the requirements file:

```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Download all models (run once)

```bash
python download_models.py
```

This downloads the `license_plate_detector.pt` model and EasyOCR English
text-detection models.
Subsequent runs will use the cached files with no network needed.

### 4. Start the system

```bash
python app.py
```

The dashboard will be available at **http://<Pi-IP>:5000**.

## Users Module (App Factory)

A separate users/auth module is now included using:

- Flask app-factory pattern
- Flask-SQLAlchemy models (`users`, `vehicles`)
- Flask-Login authentication
- Flask-Bcrypt password hashing
- Flask-WTF CSRF protection
- Admin CRUD for users and associated vehicle plates

Run the users module:

```bash
python users_app.py
```

Open:

- `http://<Pi-IP>:5001/login`

Default seeded superadmin (first run only, when no users exist):

- Email: `admin@campus.local`
- Password: `changeme123`

Environment variables for production use:

- `SECRET_KEY` (required outside development)
- `DATABASE_URL` (optional, defaults to `sqlite:///lpr_system.db`)

Example:

```bash
SECRET_KEY='replace-me' DATABASE_URL='sqlite:///lpr_system.db' python users_app.py
```

## How It Works

```
Frame from webcam
      │
      ▼
 MOG2 motion detection
      │ motion? ──No──▶ skip
      ▼
 YOLOv8 plate detector inference
      │ plate? ──No──▶ skip
      ▼
 Crop plate bounding box
      │
      ▼
 Image preprocessing
 (multiple preprocessing variants)
      │
      ▼
 EasyOCR plate reading
      │
      ▼
 Save image + log to SQLite
```

## Dashboard

- **Live feeds** — side-by-side MJPEG streams with real-time YOLO bounding boxes
- **Stats cards** — total / today / entrance / exit counts
- **Detection log** — plate number, camera, timestamp, confidence, thumbnail
- **Date filter** — view history for any specific date

## Phase 3 Web Routes

- `/login` and `/logout` for session auth
- `/users` for superadmin/guard account creation workflow
- `/guard/device` for entry/exit camera assignment and preview
- `/guard/review` for flagged detection review actions
- `/logbook` for filterable logs with export controls (`?export=csv` or `?export=pdf`)

## Phase 4 Offline-First Sync

Local detections now include a `sync_status` lifecycle (`PENDING`, `FAILED`, `SYNCED`) and are always written to SQLite first.

### Device worker behavior

- `sync_worker.py` runs as a background thread inside `app.py`
- On each cycle:
     - checks cloud health: `GET /health`
     - fetches local `PENDING` (+ optional `FAILED`) detections
     - posts batch to cloud: `POST /sync/logs`
     - marks local rows `SYNCED` on success or `FAILED` on sync submission error

### App sync endpoints

- `GET /api/sync/status` → runtime + queue counters
- `POST /api/sync/run` → trigger one immediate sync cycle

### Environment variables

See `.env.example` for all settings. Important keys:

- `CLOUD_API_BASE_URL`
- `CLOUD_API_KEY`
- `SYNC_INTERVAL_SECONDS`
- `SYNC_BATCH_SIZE`
- `SYNC_HTTP_TIMEOUT_SECONDS`
- `SYNC_INCLUDE_FAILED`
- `DEVICE_ID`

### Cloud mirror API (FastAPI)

Run the thin cloud API service:

```bash
uvicorn cloud_sync_api:app --host 0.0.0.0 --port 8000
```

Cloud API routes:

- `GET /health`
- `POST /sync/logs`
- `GET /logs`

### Railway Deployment (Cloud Sync API)

This repository now includes a `Procfile` for Railway:

```txt
web: uvicorn cloud_sync_api:app --host 0.0.0.0 --port ${PORT:-8000}
```

Deploy steps:

1. Create a new Railway service from this repo.
2. Set environment variables on Railway:
     - `CLOUD_API_KEY` = same shared key used by device `.env`
     - `CLOUD_SYNC_DB_PATH` = `/data/cloud_sync.db` (if using a Railway volume)
3. Confirm health endpoint works at `https://<your-service>.up.railway.app/health`.
4. On the device app, set:
     - `CLOUD_API_BASE_URL=https://<your-service>.up.railway.app`
     - same `CLOUD_API_KEY`

Optional but recommended:

- Mount a Railway volume and use `/data/cloud_sync.db` to persist cloud mirror logs across redeploys.
- For production scale, switch cloud storage from SQLite file to managed PostgreSQL.

## Configuration

Key settings in `camera_system.py`:

| Variable           | Default | Description                               |
|--------------------|---------|-------------------------------------------|
| `MOTION_THRESHOLD` | 8000    | Min contour area (px²) to trigger YOLO    |
| `CAPTURE_COOLDOWN` | 10.0    | Seconds between captures per tracked plate |
| `YOLO_CONF`        | 0.50    | YOLO minimum confidence threshold         |
| `MIN_PLATE_AREA`   | 700     | Minimum plate bounding-box area (px²)     |

Batch-merge tuning (environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_MERGE_IOU` | 0.20 | Overlap threshold for merging near-duplicate detections into one input event |
| `BATCH_CENTER_DISTANCE` | 80.0 | Pixel distance threshold for merging nearby detections into one input event |

Run with overrides:

```bash
BATCH_MERGE_IOU=0.30 BATCH_CENTER_DISTANCE=60 python app.py
```

## License

MIT
