"""Avalon Sentinel Dashboard - FastAPI Application"""
import csv
import io
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Fix for PyTorch 2.6+ weights_only issue - patch torch.load before importing YOLO
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
import ocrmac

from config import (
    WHITELIST_CSV,
    TRAFFIC_LOG_CSV,
    IMAGES_DIR,
    CAMERA_SOURCE,
    DETECTION_FRAME_INTERVAL,
    YOLO_MODEL_PATH,
    WHITELIST_COLUMNS,
    TRAFFIC_LOG_COLUMNS,
)

app = FastAPI(title="Avalon Sentinel Dashboard")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Global state for current detection
current_detection: Dict[str, Any] = {
    "plate": None,
    "confidence": 0.0,
    "bbox": None,
    "status": "UNKNOWN",
    "resident_info": None,
    "timestamp": None,
}
detection_lock = Lock()

# Initialize YOLO model (torch.load already patched above)
print("Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLOv8 model loaded.")

# Camera initialization
camera = None
frame_count = 0


def init_camera():
    """Initialize camera source"""
    global camera
    try:
        print(f"[CAMERA] Attempting to initialize camera source: {CAMERA_SOURCE}")
        
        # Release existing camera if any
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
        
        # CAMERA_SOURCE is a string from env, check if it's a digit
        if str(CAMERA_SOURCE).isdigit():
            camera = cv2.VideoCapture(int(CAMERA_SOURCE), cv2.CAP_AVFOUNDATION)  # Use AVFoundation on macOS
            print(f"[CAMERA] Opened camera index: {CAMERA_SOURCE} (using AVFoundation)")
        else:
            camera = cv2.VideoCapture(CAMERA_SOURCE)
            print(f"[CAMERA] Opened camera URL: {CAMERA_SOURCE}")
        
        if not camera.isOpened():
            print(f"[CAMERA] ✗ Warning: Could not open camera {CAMERA_SOURCE}")
            return False
        
        # Wait a bit for camera to be ready (macOS needs this)
        import time
        time.sleep(0.5)
        
        # Try to read a frame to verify it works
        ret, test_frame = camera.read()
        if not ret:
            print(f"[CAMERA] ⚠ Camera opened but cannot read frames yet, will retry...")
            # Try a few more times
            for i in range(5):
                time.sleep(0.2)
                ret, test_frame = camera.read()
                if ret:
                    print(f"[CAMERA] ✓ Camera can read frames after {i+1} attempts")
                    break
            if not ret:
                print(f"[CAMERA] ✗ Camera still cannot read frames after retries")
                # Don't return False yet - let video stream handle it
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = camera.get(cv2.CAP_PROP_FPS)
        print(f"[CAMERA] ✓ Camera {CAMERA_SOURCE} initialized")
        print(f"[CAMERA]   Resolution: {actual_width}x{actual_height}, FPS: {fps}")
        return True
    except Exception as e:
        print(f"[CAMERA] ✗ Error initializing camera: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_csv_files():
    """Initialize CSV files with headers if they don't exist"""
    # Initialize whitelist.csv
    if not WHITELIST_CSV.exists():
        with open(WHITELIST_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=WHITELIST_COLUMNS)
            writer.writeheader()
        print(f"Created {WHITELIST_CSV}")

    # Initialize traffic_log.csv
    if not TRAFFIC_LOG_CSV.exists():
        with open(TRAFFIC_LOG_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAFFIC_LOG_COLUMNS)
            writer.writeheader()
        print(f"Created {TRAFFIC_LOG_CSV}")


def load_whitelist() -> Dict[str, Dict[str, str]]:
    """Load whitelist from CSV into a dictionary"""
    whitelist = {}
    if not WHITELIST_CSV.exists():
        return whitelist

    with open(WHITELIST_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate = row.get("plate", "").strip().upper()
            if plate:
                whitelist[plate] = {
                    "unit": row.get("unit", ""),
                    "resident_name": row.get("resident_name", ""),
                    "destination": row.get("destination", ""),
                    "source": row.get("source", ""),
                }
    return whitelist


def normalize_plate(plate_text: str) -> str:
    """Normalize plate text (uppercase, strip spaces/punctuation)"""
    if not plate_text:
        return ""
    # Remove spaces and common punctuation, convert to uppercase
    normalized = "".join(c.upper() for c in plate_text if c.isalnum())
    return normalized


def detect_plate_yolo(frame: np.ndarray) -> Optional[tuple]:
    """
    Detect license plate using YOLOv8
    Returns: (x, y, w, h, confidence) or None
    
    NOTE: Standard YOLOv8 models don't detect license plates by default.
    For production, you should:
    1. Use a pre-trained license plate detection model, OR
    2. Fine-tune YOLOv8 on license plate data, OR
    3. Use a dedicated LPR model
    
    For MVP, this looks for any high-confidence detection as a potential plate region.
    """
    try:
        results = yolo_model(frame, verbose=False)
        detections_found = 0
        for result in results:
            boxes = result.boxes
            # Get class names
            class_names = result.names if hasattr(result, 'names') else {}
            
            for box in boxes:
                # For MVP: accept any high-confidence detection as potential plate
                # In production, filter by class ID for license plates
                conf = float(box.conf[0])
                class_id = int(box.cls[0]) if hasattr(box, 'cls') else -1
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    # Ensure reasonable size (filter out very small detections)
                    if w > 50 and h > 20:
                        detections_found += 1
                        print(f"[DETECTION] Found: {class_name} ({conf:.2f}) at ({x},{y}) size {w}x{h}")
                        return (x, y, w, h, conf)
                    else:
                        print(f"[DETECTION] Rejected {class_name} ({conf:.2f}): too small ({w}x{h})")
        
        if detections_found == 0:
            print(f"[DETECTION] No valid detections in frame (checked {len(boxes) if 'boxes' in locals() else 0} boxes)")
    except Exception as e:
        print(f"[ERROR] YOLOv8 detection error: {e}")
        import traceback
        traceback.print_exc()
    return None


def ocr_plate_crop(crop: np.ndarray) -> Optional[str]:
    """
    Run OCR on cropped plate region using Apple Vision via ocrmac
    Returns: plate text or None
    """
    try:
        print(f"[OCR] Attempting OCR on crop size: {crop.shape}")
        # Convert numpy array to PIL Image for ocrmac
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        print(f"[OCR] PIL image size: {pil_image.size}")
        
        # Use ocrmac to get text (correct API: ocrmac.ocrmac.OCR)
        from ocrmac.ocrmac import OCR, text_from_image
        
        # Try OCR class first
        try:
            print("[OCR] Trying OCR class...")
            ocr_result = OCR(pil_image)
            if ocr_result and hasattr(ocr_result, 'text') and ocr_result.text:
                result_text = str(ocr_result.text).strip()
                print(f"[OCR] OCR class result: '{result_text}'")
                return result_text
            elif hasattr(ocr_result, 'string') and ocr_result.string:
                result_text = str(ocr_result.string).strip()
                print(f"[OCR] OCR class result (string): '{result_text}'")
                return result_text
            else:
                print("[OCR] OCR class returned no text")
        except Exception as e1:
            print(f"[OCR] OCR class error: {e1}")
        
        # Fallback: try text_from_image function
        try:
            print("[OCR] Trying text_from_image...")
            text = text_from_image(pil_image)
            if text:
                # Handle if text is a list of tuples: [(text, confidence, bbox), ...]
                if isinstance(text, list) and len(text) > 0:
                    # Extract text from first tuple (highest confidence)
                    if isinstance(text[0], tuple) and len(text[0]) > 0:
                        text = text[0][0]  # Get text from first tuple
                        print(f"[OCR] text_from_image result (from tuple): '{text}'")
                    else:
                        text = ' '.join(str(t) for t in text if t)
                        print(f"[OCR] text_from_image result (from list): '{text}'")
                else:
                    print(f"[OCR] text_from_image result: '{text}'")
                return str(text).strip()
            else:
                print("[OCR] text_from_image returned no text")
        except Exception as e2:
            print(f"[OCR] text_from_image error: {e2}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] OCR error: {e}")
        import traceback
        traceback.print_exc()
    print("[OCR] No text extracted")
    return None


def lookup_whitelist(plate: str) -> Optional[Dict[str, str]]:
    """Look up plate in whitelist CSV"""
    whitelist = load_whitelist()
    normalized = normalize_plate(plate)
    return whitelist.get(normalized)


def append_traffic_log(
    plate: str,
    decision: str,
    image_path: str = "",
    visitor_name: str = "",
    destination: str = "",
    source: str = "",
):
    """Append a row to traffic_log.csv"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp,
        "plate": plate,
        "decision": decision,
        "image_path": image_path,
        "visitor_name": visitor_name,
        "destination": destination,
        "source": source,
    }

    with open(TRAFFIC_LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAFFIC_LOG_COLUMNS)
        writer.writerow(row)


def process_frame(frame: np.ndarray) -> tuple:
    """
    Process a single frame: detect plate, OCR, lookup whitelist
    Returns: (frame_with_bbox, detection_info)
    """
    global frame_count
    frame_count += 1

    detection_info = None
    frame_with_bbox = frame.copy()

    # Run detection every Nth frame
    if frame_count % DETECTION_FRAME_INTERVAL == 0:
        print(f"\n[FRAME {frame_count}] Processing frame (every {DETECTION_FRAME_INTERVAL} frames)")
        # YOLOv8 detection
        bbox = detect_plate_yolo(frame)
        
        if bbox:
            x, y, w, h, confidence = bbox
            print(f"[FRAME {frame_count}] Bounding box found: ({x},{y}) size {w}x{h} conf={confidence:.2f}")
            
            # Draw bounding box
            cv2.rectangle(frame_with_bbox, (x, y), (x + w, y + h), (0, 255, 65), 2)
            cv2.putText(
                frame_with_bbox,
                f"PLATE DETECTED {confidence*100:.0f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 65),
                2,
            )

            # Crop plate region
            crop = frame[y : y + h, x : x + w]
            print(f"[FRAME {frame_count}] Cropped region: {crop.shape}")
            
            # OCR using ocrmac
            plate_text = ocr_plate_crop(crop)
            
            if plate_text:
                normalized_plate = normalize_plate(plate_text)
                print(f"[FRAME {frame_count}] OCR extracted: '{plate_text}' -> normalized: '{normalized_plate}'")
                
                # Lookup in whitelist
                resident_info = lookup_whitelist(normalized_plate)
                
                if resident_info:
                    status = "AUTHORIZED"
                    print(f"[FRAME {frame_count}] ✓ AUTHORIZED: {resident_info.get('resident_name')} (Unit {resident_info.get('unit')})")
                else:
                    status = "UNKNOWN"
                    print(f"[FRAME {frame_count}] ✗ UNKNOWN: Plate '{normalized_plate}' not in whitelist")
                
                # Save snapshot
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{timestamp_str}_{normalized_plate}.jpg"
                image_path = IMAGES_DIR / image_filename
                cv2.imwrite(str(image_path), frame)
                print(f"[FRAME {frame_count}] Saved snapshot: {image_path}")
                
                # Update global detection state
                with detection_lock:
                    current_detection.update({
                        "plate": normalized_plate,
                        "confidence": confidence,
                        "bbox": (x, y, w, h),
                        "status": status,
                        "resident_info": resident_info,
                        "timestamp": datetime.now().isoformat(),
                    })
                
                # Log to CSV
                append_traffic_log(
                    plate=normalized_plate,
                    decision=status,
                    image_path=str(image_path),
                    visitor_name=resident_info.get("resident_name", "") if resident_info else "",
                    destination=resident_info.get("destination", "") if resident_info else "",
                    source=resident_info.get("source", "") if resident_info else "",
                )
                print(f"[FRAME {frame_count}] Logged to CSV: {status}")
                
                detection_info = {
                    "plate": normalized_plate,
                    "status": status,
                    "resident_info": resident_info,
                }
            else:
                print(f"[FRAME {frame_count}] No text extracted from OCR")
        else:
            if frame_count % (DETECTION_FRAME_INTERVAL * 10) == 0:  # Log every 10th detection attempt
                print(f"[FRAME {frame_count}] No detections in frame")

    return frame_with_bbox, detection_info


def video_stream():
    """Generator for MJPEG video stream"""
    global camera
    
    print("[VIDEO_STREAM] Video stream generator started")
    
    # Initialize camera if not already done
    if camera is None:
        print("[VIDEO_STREAM] Camera is None, initializing...")
        if not init_camera():
            # Fallback: generate a test pattern with simulated plate
            test_count = 0
            while True:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "Camera not available - Test Mode",
                    (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                # Simulate a plate region for testing
                cv2.rectangle(frame, (200, 300), (450, 350), (0, 255, 65), 2)
                cv2.putText(
                    frame,
                    "ABC123",
                    (220, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 65),
                    2,
                )
                cv2.putText(
                    frame,
                    "PLATE DETECTED",
                    (200, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 65),
                    1,
                )
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(0.1)
            return

    frame_error_count = 0
    frame_read_count = 0
    print("[VIDEO] Starting video stream loop...")
    print(f"[VIDEO] Camera object: {camera}, isOpened: {camera.isOpened() if camera else 'None'}")
    
    # Wait a moment for camera to be ready (macOS)
    time.sleep(0.5)
    
    while True:
        try:
            ret, frame = camera.read()
            frame_read_count += 1
            
            if not ret:
                frame_error_count += 1
                if frame_error_count == 1:
                    print(f"[VIDEO] ⚠ Camera read failed (attempt {frame_error_count}) - waiting...")
                    time.sleep(0.5)  # Wait longer on first failure
                elif frame_error_count <= 5:
                    print(f"[VIDEO] ⚠ Camera read failed (attempt {frame_error_count})")
                    time.sleep(0.2)
                elif frame_error_count > 10:
                    print("[VIDEO] ✗ Camera read failed multiple times, reinitializing...")
                    try:
                        camera.release()
                    except:
                        pass
                    camera = None
                    time.sleep(1)
                    if not init_camera():
                        print("[VIDEO] ✗ Failed to reinitialize camera")
                        break
                    frame_error_count = 0
                    time.sleep(0.5)  # Wait after reinit
                else:
                    time.sleep(0.1)
                continue
            
            if frame_error_count > 0:
                print(f"[VIDEO] ✓ Camera read recovered after {frame_error_count} errors")
            frame_error_count = 0
            
            # Log first frame and then every 30 frames
            if frame_read_count == 1:
                print(f"[VIDEO] ✓ First frame read successfully! Shape: {frame.shape}")
            elif frame_read_count % 30 == 0:  # Log every 30 frames (~1 second at 30fps)
                print(f"[VIDEO] Streaming frame {frame_read_count}, shape: {frame.shape}")

            # Process frame (detection + OCR)
            processed_frame, _ = process_frame(frame)

            # Encode as JPEG
            _, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        except Exception as e:
            print(f"[VIDEO] ✗ Error in video stream loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
            continue


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    init_csv_files()
    init_camera()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera
    if camera:
        camera.release()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream endpoint"""
    print("\n" + "="*50)
    print("[VIDEO_FEED] ⚡ Video feed endpoint accessed - starting stream")
    print("="*50)
    return StreamingResponse(
        video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/partials/decision_panel", response_class=HTMLResponse)
async def decision_panel(request: Request):
    """HTMX endpoint for decision panel"""
    with detection_lock:
        detection = current_detection.copy()
    
    return templates.TemplateResponse(
        "partials/decision_panel.html",
        {"request": request, "detection": detection}
    )


@app.get("/partials/traffic_log", response_class=HTMLResponse)
async def traffic_log(request: Request):
    """HTMX endpoint for traffic log"""
    log_entries = []
    
    if TRAFFIC_LOG_CSV.exists():
        with open(TRAFFIC_LOG_CSV, "r") as f:
            reader = csv.DictReader(f)
            entries = list(reader)
            # Get last 50 entries, newest first
            log_entries = list(reversed(entries[-50:]))
    
    return templates.TemplateResponse(
        "partials/traffic_log.html",
        {"request": request, "log_entries": log_entries}
    )


@app.get("/partials/status_bar", response_class=HTMLResponse)
async def status_bar(request: Request):
    """HTMX endpoint for status bar"""
    return templates.TemplateResponse("partials/status_bar.html", {"request": request})


@app.post("/action/open_gate", response_class=HTMLResponse)
async def open_gate(request: Request):
    """Guard action: Open gate"""
    with detection_lock:
        plate = current_detection.get("plate", "UNKNOWN")
        resident_info = current_detection.get("resident_info")
    
    if plate and plate != "UNKNOWN":
        # Log action
        append_traffic_log(
            plate=plate,
            decision="ENTERED",
            visitor_name=resident_info.get("resident_name", "") if resident_info else "",
            destination=resident_info.get("destination", "") if resident_info else "",
            source=resident_info.get("source", "") if resident_info else "",
        )
        
        # Update status
        with detection_lock:
            current_detection["status"] = "ENTERED"
    
    return templates.TemplateResponse(
        "partials/decision_panel.html",
        {"request": request, "detection": current_detection}
    )


@app.post("/action/deny_entry", response_class=HTMLResponse)
async def deny_entry(request: Request):
    """Guard action: Deny entry"""
    with detection_lock:
        plate = current_detection.get("plate", "UNKNOWN")
        resident_info = current_detection.get("resident_info")
    
    if plate and plate != "UNKNOWN":
        # Log action
        append_traffic_log(
            plate=plate,
            decision="BLOCKED",
            visitor_name=resident_info.get("resident_name", "") if resident_info else "",
            destination=resident_info.get("destination", "") if resident_info else "",
            source=resident_info.get("source", "") if resident_info else "",
        )
        
        # Update status
        with detection_lock:
            current_detection["status"] = "BLOCKED"
    
    return templates.TemplateResponse(
        "partials/decision_panel.html",
        {"request": request, "detection": current_detection}
    )


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login(request: Request):
    """Admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request})


@app.post("/test/simulate_detection", response_class=HTMLResponse)
async def simulate_detection(request: Request):
    """Test endpoint to simulate a plate detection"""
    import json
    try:
        # Try to get JSON data
        try:
            data = await request.json()
            plate = data.get("plate", "ABC123")
        except:
            # Try form data
            form = await request.form()
            plate = form.get("plate", "ABC123")
    except:
        plate = "ABC123"
    
    # Normalize plate
    normalized = normalize_plate(plate)
    
    # Lookup in whitelist
    resident_info = lookup_whitelist(normalized)
    
    if resident_info:
        status = "AUTHORIZED"
    else:
        status = "UNKNOWN"
    
    # Update global detection state
    with detection_lock:
        current_detection.update({
            "plate": normalized,
            "confidence": 0.95,
            "bbox": (200, 300, 250, 50),
            "status": status,
            "resident_info": resident_info,
            "timestamp": datetime.now().isoformat(),
        })
    
    # Log to CSV
    append_traffic_log(
        plate=normalized,
        decision=status,
        image_path="test_simulation.jpg",
        visitor_name=resident_info.get("resident_name", "") if resident_info else "",
        destination=resident_info.get("destination", "") if resident_info else "",
        source=resident_info.get("source", "") if resident_info else "",
    )
    
    # Return updated decision panel HTML
    return templates.TemplateResponse(
        "partials/decision_panel.html",
        {"request": request, "detection": current_detection.copy()}
    )


@app.post("/test/add_sample_logs")
async def add_sample_logs():
    """Add sample log entries for testing"""
    sample_plates = [
        ("ABC123", "AUTHORIZED", "Juan Perez", "Tower 2 Apt 304", "WhatsApp Pre-Auth"),
        ("XYZ789", "AUTHORIZED", "Maria Garcia", "Building A Unit 205", "Live Detection"),
        ("UNK999", "UNKNOWN", "", "", ""),
    ]
    
    for plate, decision, name, dest, src in sample_plates:
        append_traffic_log(
            plate=plate,
            decision=decision,
            image_path=f"sample_{plate}.jpg",
            visitor_name=name,
            destination=dest,
            source=src,
        )
    
    return {"status": "success", "added": len(sample_plates)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

