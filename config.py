"""Configuration for Avalon Sentinel Dashboard"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# CSV file paths
WHITELIST_CSV = BASE_DIR / "whitelist.csv"
TRAFFIC_LOG_CSV = BASE_DIR / "traffic_log.csv"

# Images directory for snapshots
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Camera configuration
# For development, can use 0 for webcam or RTSP URL
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")  # 0 for webcam, or RTSP URL

# Detection settings
DETECTION_FRAME_INTERVAL = 3  # Run detection every Nth frame
YOLO_MODEL_PATH = "yolov8n.pt"  # Will download automatically if not present

# Whitelist CSV columns
WHITELIST_COLUMNS = ["plate", "unit", "resident_name", "destination", "source"]

# Traffic log CSV columns
TRAFFIC_LOG_COLUMNS = [
    "timestamp",
    "plate",
    "decision",
    "image_path",
    "visitor_name",
    "destination",
    "source"
]


