# LPR System — Raspberry Pi 5 License Plate Recognition

Dual-camera license plate recognition system using **YOLOv8s** for vehicle
detection and **EasyOCR** for plate reading, running on a Raspberry Pi 5.

## Features

- **Dual USB webcams** — entrance (`/dev/video0`) and exit (`/dev/video1`)
- **Motion detection** (MOG2) → **YOLOv8s vehicle confirmation** → **EasyOCR plate reading**
- YOLO bounding boxes drawn on the live MJPEG stream
- 5-second cooldown prevents duplicate captures of the same vehicle
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
├── camera_system.py     # motion detection + YOLO vehicle detection + capture
├── ocr_processor.py     # EasyOCR plate recognition + image preprocessing
├── database.py          # SQLite setup and queries
├── app.py               # Flask web server + dashboard
├── download_models.py   # one-time model download script
├── templates/
│   └── dashboard.html   # dark-themed dashboard UI
├── models/
│   └── yolov8s.pt       # auto-downloaded by download_models.py
├── captures/            # saved vehicle images
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

This downloads YOLOv8s (~22 MB) and the EasyOCR English text-detection models.
Subsequent runs will use the cached files with no network needed.

### 4. Start the system

```bash
python app.py
```

The dashboard will be available at **http://<Pi-IP>:5000**.

## How It Works

```
Frame from webcam
      │
      ▼
 MOG2 motion detection
      │ motion? ──No──▶ skip
      ▼
 YOLOv8s inference
      │ vehicle? ──No──▶ skip
      ▼
 Crop vehicle bounding box
      │
      ▼
 Image preprocessing
 (grayscale → CLAHE → denoise)
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

## Configuration

Key settings in `camera_system.py`:

| Variable           | Default | Description                               |
|--------------------|---------|-------------------------------------------|
| `MOTION_THRESHOLD` | 5000    | Min contour area (px²) to trigger YOLO    |
| `CAPTURE_COOLDOWN` | 5.0     | Seconds between captures per camera       |
| `YOLO_CONF`        | 0.35    | YOLO minimum confidence threshold         |
| `VEHICLE_CLASSES`  | 2,3,5,7 | COCO class IDs (car, motorcycle, bus, truck) |

## License

MIT
