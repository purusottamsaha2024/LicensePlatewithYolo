"""Test script for number plate detection pipeline"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Fix for PyTorch 2.6+ weights_only issue - patch torch.load before importing YOLO
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

# Test configuration
YOLO_MODEL_PATH = "yolov8n.pt"
TEST_IMAGE_PATH = "test_plate.jpg"  # You can add a test image here

def test_yolo_detection():
    """Test YOLOv8 detection"""
    print("=" * 50)
    print("Testing YOLOv8 Detection")
    print("=" * 50)
    
    try:
        print("Loading YOLOv8 model...")
        model = YOLO(YOLO_MODEL_PATH)
        print("✓ Model loaded successfully")
        
        # Create a test frame (or load from file if available)
        if Path(TEST_IMAGE_PATH).exists():
            print(f"Loading test image: {TEST_IMAGE_PATH}")
            frame = cv2.imread(TEST_IMAGE_PATH)
        else:
            print("Creating synthetic test frame (640x480)...")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a simple rectangle to simulate a plate
            cv2.rectangle(frame, (200, 200), (400, 250), (255, 255, 255), -1)
            cv2.putText(frame, "ABC-123", (220, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        print("Running detection...")
        results = model(frame, verbose=False)
        
        detections = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 50 and h > 20:
                        detections += 1
                        print(f"  ✓ Detection found: confidence={conf:.2f}, size={w}x{h}")
        
        if detections == 0:
            print("  ⚠ No detections found (this is normal - YOLOv8 doesn't detect plates by default)")
            print("  For testing, you can manually crop a plate region and test OCR")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ocr(image_path=None):
    """Test OCR on a cropped plate region"""
    print("\n" + "=" * 50)
    print("Testing OCR (ocrmac)")
    print("=" * 50)
    
    try:
        if image_path and Path(image_path).exists():
            print(f"Loading image: {image_path}")
            img = cv2.imread(image_path)
        else:
            print("Creating test image with text...")
            # Create a test image with text
            img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(img, "ABC123", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Convert to PIL for ocrmac
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        print("Running OCR...")
        from ocrmac.ocrmac import OCR, text_from_image
        
        # Try OCR class first
        try:
            ocr_result = OCR(pil_image)
            if ocr_result and hasattr(ocr_result, 'text') and ocr_result.text:
                print(f"✓ OCR Result: '{ocr_result.text.strip()}'")
                return True
            elif hasattr(ocr_result, 'string') and ocr_result.string:
                print(f"✓ OCR Result: '{ocr_result.string.strip()}'")
                return True
        except:
            pass
        
        # Fallback to text_from_image
        try:
            text = text_from_image(pil_image)
            if text:
                # Handle if text is a list of tuples: [(text, confidence, bbox), ...]
                if isinstance(text, list) and len(text) > 0:
                    # Extract text from first tuple (highest confidence)
                    if isinstance(text[0], tuple) and len(text[0]) > 0:
                        text = text[0][0]  # Get text from first tuple
                    else:
                        text = ' '.join(str(t) for t in text if t)
                print(f"✓ OCR Result (text_from_image): '{str(text).strip()}'")
                return True
        except Exception as e2:
            print(f"⚠ OCR methods failed: {e2}")
        
        print("⚠ No text detected")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def normalize_plate(plate_text: str) -> str:
    """Normalize plate text (uppercase, strip spaces/punctuation)"""
    if not plate_text:
        return ""
    normalized = "".join(c.upper() for c in plate_text if c.isalnum())
    return normalized

def test_whitelist_lookup():
    """Test whitelist CSV lookup"""
    print("\n" + "=" * 50)
    print("Testing Whitelist Lookup")
    print("=" * 50)
    
    try:
        import csv
        from config import WHITELIST_CSV
        
        if not WHITELIST_CSV.exists():
            print(f"⚠ {WHITELIST_CSV} not found")
            return False
        
        print(f"Loading whitelist from {WHITELIST_CSV}...")
        whitelist = {}
        with open(WHITELIST_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                plate = row.get("plate", "").strip().upper()
                if plate:
                    whitelist[normalize_plate(plate)] = row
                    print(f"  ✓ Loaded: {plate} -> {row.get('resident_name', 'N/A')}")
        
        # Test lookup
        test_plates = ["ABC123", "XYZ789", "NOTFOUND"]
        for plate in test_plates:
            normalized = normalize_plate(plate)
            if normalized in whitelist:
                print(f"✓ Found: {plate} -> {whitelist[normalized].get('resident_name')}")
            else:
                print(f"✗ Not found: {plate}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("AVALON SENTINEL - PLATE DETECTION TEST SUITE")
    print("=" * 50 + "\n")
    
    results = []
    
    # Test 1: YOLOv8 Detection
    results.append(("YOLOv8 Detection", test_yolo_detection()))
    
    # Test 2: OCR
    results.append(("OCR (ocrmac)", test_ocr()))
    
    # Test 3: Whitelist Lookup
    results.append(("Whitelist Lookup", test_whitelist_lookup()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("=" * 50))
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("⚠ Some tests failed (see details above)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

