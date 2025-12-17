# Avalon Sentinel Dashboard

A production-quality MVP web app for License Plate Recognition (LPR) guard assistant with a cyberpunk SOC-style UI.

## Features

- **Live Camera Feed** with YOLOv8 license plate detection
- **Apple Vision OCR** via `ocrmac` for plate text recognition
- **CSV-based Whitelist** for authorized plates
- **Real-time Decision Engine** with big green/red status display
- **Live Traffic Log** showing all detections and guard actions
- **HTMX-powered** dynamic updates (no React/Vue)
- **Cyberpunk Dark Theme** with neon green/red accents

## Tech Stack

- **Backend**: FastAPI (Python 3)
- **Frontend**: HTML + HTMX + Tailwind CSS
- **Computer Vision**: YOLOv8 (ultralytics) + Apple Vision (ocrmac)
- **Data Storage**: CSV files (whitelist.csv, traffic_log.csv)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The YOLOv8 model will download automatically on first run.

## Configuration

Set environment variables (optional):
```bash
export CAMERA_SOURCE=0  # 0 for webcam, or RTSP URL
```

## Running

```bash
export FASTAPI_APP=app.py
fastapi run
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

Then visit: `http://localhost:5000/`

## CSV Files

### whitelist.csv
Configuration file for authorized plates:
- `plate`: License plate number
- `unit`: Unit/resident number
- `resident_name`: Name of resident
- `destination`: Destination address
- `source`: Source of authorization (e.g., "WhatsApp Pre-Auth")

### traffic_log.csv
Append-only log of all detections and actions:
- `timestamp`: Detection/action timestamp
- `plate`: License plate number
- `decision`: AUTHORIZED, UNKNOWN, ENTERED, or BLOCKED
- `image_path`: Path to saved snapshot
- `visitor_name`: Name from whitelist
- `destination`: Destination from whitelist
- `source`: Source from whitelist

## Project Structure

```
.
├── app.py                 # Main FastAPI application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── whitelist.csv          # Authorized plates (created on first run)
├── traffic_log.csv        # Detection log (created on first run)
├── images/                # Saved snapshots (created automatically)
├── templates/
│   ├── base.html          # Base template with Tailwind/HTMX
│   ├── dashboard.html     # Main 3-zone dashboard
│   ├── admin_login.html   # Admin login page
│   └── partials/
│       ├── decision_panel.html  # Zone B: Decision engine
│       ├── traffic_log.html     # Zone C: Live log
│       └── status_bar.html      # Top status bar
└── static/
    ├── css/
    │   └── custom.css     # Custom animations and styles
    └── js/
        └── app.js         # Minimal client-side JS
```

## Phase 1 MVP Features

✅ Camera feed with YOLOv8 detection
✅ OCR using Apple Vision (ocrmac)
✅ CSV whitelist lookup
✅ Big green/red decision screen
✅ Automatic CSV logging
✅ Guard action buttons (OPEN GATE / DENY ENTRY)
✅ Live traffic log display
✅ Cyberpunk SOC-style UI

## Notes


- Detection runs every 3rd frame by default (configurable in `config.py`)
- Snapshots are saved to `images/` directory with timestamp and plate number
- The system uses in-memory state for current detection, updated via background processing
- HTMX polls endpoints every 1-3 seconds for real-time updates







certifi==2025.11.12
charset-normalizer==3.4.4
contourpy==1.3.2
cycler==0.12.1
filelock==3.20.0
fonttools==4.61.1
fsspec==2025.12.0
idna==3.11
Jinja2==3.1.6
kiwisolver==1.4.9
MarkupSafe==3.0.3
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.6
opencv-python==4.12.0.88
packaging==25.0
pandas==2.3.3
pillow==12.0.0
polars==1.36.1
polars-runtime-32==1.36.1
psutil==7.1.3
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
requests==2.32.5
scipy==1.15.3
six==1.17.0
sympy==1.14.0
torch==2.9.1
torchvision==0.24.1
typing_extensions==4.15.0
tzdata==2025.3
ultralytics==8.3.237
ultralytics-thop==2.0.18
urllib3==2.6.2
