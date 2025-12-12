import cv2
import threading
import time
import datetime
import re
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# Optional OCR backends: prefer ocrmac, fallback to pytesseract
try:
    from ocrmac import ocrmac as ocrmac_lib
    OCR_BACKEND = "ocrmac"
except Exception:
    OCR_BACKEND = "pytesseract"
    try:
        import pytesseract
    except Exception:
        pytesseract = None

# --- CONFIGURATION ---
WHITELIST_FILE = 'whitelist.csv'
LOG_FILE = 'gate_logs.txt'
RESET_TIME = 5.0           # Time to wait before detecting NEW car
SUCCESS_DISPLAY_TIME = 5.0 # How long to show "ACCESS GRANTED" before resetting

# Camera device indices (change to match your USB cameras)
ROAD_CAMERA_INDEX = 0
ID_CAMERA_INDEX = 1

# --- GLOBAL STATE ---
system_state = {
    "status": "SCANNING",
    "message": "WAITING...",
    "last_plate": None,
    "needs_id_scan": False,
    "frame_road": None,
    "frame_id": None,
    "running": True,
    "success_timer": 0,
    "debug_ocr_text": ""
}

class SmartGatePro:
    def __init__(self, road_cam=ROAD_CAMERA_INDEX, id_cam=ID_CAMERA_INDEX):
        print("Initializing SmartGatePro (model will load lazily)...")
        self.model = None
        self.model_loaded = False
        self.road_cam = road_cam
        self.id_cam = id_cam
        self.whitelist = self.load_whitelist()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def load_whitelist(self):
        try:
            df = pd.read_csv(WHITELIST_FILE)
            df['plate'] = df['plate'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
            return df
        except Exception:
            return pd.DataFrame(columns=['plate', 'owner', 'apartment', 'status'])

    def log(self, plate, action, details):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] PLATE: {plate} | ACTION: {action} | DETAILS: {details}"
        print(entry)
        try:
            with open(LOG_FILE, "a") as f:
                f.write(entry + "\n")
        except Exception:
            pass

    def check_access(self, plate):
        match = self.whitelist[self.whitelist['plate'] == plate]
        if not match.empty:
            return True, f"{match.iloc[0]['owner']} ({match.iloc[0]['apartment']})"
        return False, "Unknown"

    def parse_dimex(self, texts):
        found_number = None
        first_name = None
        last_name = None
        full_text = " ".join(texts).upper()
        numbers = re.findall(r'\d{9,12}', re.sub(r'[^0-9]', '', full_text))
        if numbers:
            found_number = numbers[0]
        try:
            for i, text in enumerate(texts):
                clean = text.upper().strip()
                if "NOMBRE" in clean and i + 1 < len(texts):
                    candidate = texts[i+1].upper().strip()
                    if len(candidate) > 2 and not any(char.isdigit() for char in candidate):
                        first_name = candidate
                if "APELLIDOS" in clean and i + 1 < len(texts):
                    candidate = texts[i+1].upper().strip()
                    if len(candidate) > 2 and not any(char.isdigit() for char in candidate):
                        last_name = candidate
        except Exception:
            pass
        full_name = None
        if first_name and last_name:
            full_name = f"{first_name} {last_name}"
        elif first_name:
            full_name = first_name
        elif last_name:
            full_name = last_name
        return found_number, full_name

    def ocr_scan(self, image_crop):
        try:
            rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            if OCR_BACKEND == "ocrmac":
                annotations = ocrmac_lib.OCR(pil_img).recognize()
                return [ann[0] for ann in annotations]
            else:
                if pytesseract is None:
                    return []
                text = pytesseract.image_to_string(pil_img)
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                return lines
        except Exception:
            return []

    def ensure_model(self):
        if self.model_loaded:
            return True
        try:
            print("Loading YOLO model (this may download weights the first time)...")
            self.model = YOLO('yolov8n.pt')
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
            self.model = None
            self.model_loaded = False
            return False

    # --- THREAD 1: ROAD CAMERA ---
    def road_camera_loop(self):
        print("Road camera thread starting on device", self.road_cam)
        cap = cv2.VideoCapture(self.road_cam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Error: road camera {self.road_cam} not available")
            return

        while system_state["running"] and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Auto reset after success display
            if system_state["status"] == "AUTHORIZED":
                if time.time() - system_state["success_timer"] > SUCCESS_DISPLAY_TIME:
                    with self.lock:
                        system_state["status"] = "SCANNING"
                        system_state["message"] = "READY FOR NEXT CAR"
                        system_state["last_plate"] = None
                        system_state["needs_id_scan"] = False

            # Run detector if model available and not waiting for ID
            if not system_state["needs_id_scan"] and system_state["status"] != "AUTHORIZED":
                if not self.model_loaded:
                    self.ensure_model()
                if self.model is not None:
                    try:
                        results = self.model(frame, verbose=False, classes=[2])  # class 2 = car
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                                plate_crop = frame[y1:y2, x1:x2]
                                if plate_crop.size == 0:
                                    continue
                                texts = self.ocr_scan(plate_crop)
                                for text in texts:
                                    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                                    if 3 <= len(clean) <= 8 and clean != system_state["last_plate"]:
                                        authorized, info = self.check_access(clean)
                                        with self.lock:
                                            system_state["last_plate"] = clean
                                            if authorized:
                                                system_state["status"] = "AUTHORIZED"
                                                system_state["message"] = f"OPEN: {info}"
                                                system_state["success_timer"] = time.time()
                                                self.log(clean, "ENTRY", info)
                                            else:
                                                system_state["status"] = "DENIED"
                                                system_state["message"] = "DENIED - SCAN ID"
                                                system_state["needs_id_scan"] = True
                                                self.log(clean, "DENIED", "Requesting ID")
                    except Exception as e:
                        # don't crash thread on model errors
                        print("Detector error:", e)

            system_state["frame_road"] = frame.copy()
            time.sleep(0.01)

        cap.release()
        print("Road camera thread exiting")

    # --- THREAD 2: ID SCANNER ---
    def id_camera_loop(self):
        print("ID camera thread starting on device", self.id_cam)
        cap = cv2.VideoCapture(self.id_cam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Error: id camera {self.id_cam} not available")
            return

        while system_state["running"] and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if system_state["needs_id_scan"]:
                texts = self.ocr_scan(frame)
                id_num, name = self.parse_dimex(texts)
                if len(texts) > 0:
                    system_state["debug_ocr_text"] = " ".join(texts[:4])
                if id_num:
                    final_name = name if name else "Unknown Visitor"
                    with self.lock:
                        system_state["status"] = "AUTHORIZED"
                        system_state["message"] = f"VERIFIED: {final_name}"
                        system_state["needs_id_scan"] = False
                        system_state["success_timer"] = time.time()
                        self.log(system_state["last_plate"], "ID_VERIFIED", f"{final_name} ({id_num})")

            system_state["frame_id"] = frame.copy()
            time.sleep(0.01)

        cap.release()
        print("ID camera thread exiting")

    def run(self):
        t1 = threading.Thread(target=self.road_camera_loop, daemon=True)
        #t2 = threading.Thread(target=self.id_camera_loop, daemon=True)
        t1.start()
        #t2.start()

        print("UI loop starting")
        try:
            while system_state["running"]:
                if system_state["frame_road"] is not None:
                    disp_road = cv2.flip(system_state["frame_road"], 1)
                    color = (0, 0, 255) if "DENIED" in system_state["status"] else (0, 255, 0)
                    if system_state["status"] == "SCANNING":
                        color = (255, 200, 0)
                    cv2.rectangle(disp_road, (0, 0), (1280, 60), color, -1)
                    cv2.putText(disp_road, system_state["message"], (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.imshow("1. ROAD CAMERA", disp_road)

                if system_state["frame_id"] is not None:
                    disp_id = cv2.flip(system_state["frame_id"], 1)
                    h, w, _ = disp_id.shape
                    if system_state["needs_id_scan"]:
                        cv2.rectangle(disp_id, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (255, 0, 255), 2)
                        cv2.putText(disp_id, "PLACE ID HERE", (int(w*0.3), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                        cv2.putText(disp_id, f"Reading: {system_state['debug_ocr_text'][:40]}...", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if system_state["status"] == "AUTHORIZED" and (time.time() - system_state["success_timer"] < SUCCESS_DISPLAY_TIME):
                        cv2.rectangle(disp_id, (0,0), (w,h), (0,255,0), 20)
                        cv2.putText(disp_id, "ACCESS GRANTED", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

                    cv2.imshow("2. ID SCANNER", disp_id)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    system_state["running"] = False
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            system_state["running"] = False
        finally:
            self._stop_event.set()
            t1.join(timeout=2)
            #t2.join(timeout=2)
            cv2.destroyAllWindows()
            print("Application exiting")

if __name__ == "__main__":
    app = SmartGatePro()
    app.run()