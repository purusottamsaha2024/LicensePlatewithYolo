# Testing Guide for Avalon Sentinel Dashboard

## Quick Start

1. **Start the server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

2. **Open in browser:**
   - Navigate to: `http://localhost:5000/`
   - You should see the dark cyberpunk SOC dashboard

## Testing Number Plate Detection

### Method 1: Using Your Webcam (Recommended)

1. **Start the server** (see above)

2. **Point your webcam at a license plate:**
   - Use a physical license plate or a printed image
   - Make sure the plate is clearly visible and well-lit
   - The system will automatically detect and process it

3. **What to expect:**
   - Zone A (center): Live camera feed with green bounding box around detected plate
   - Zone B (right): Decision panel showing AUTHORIZED (green) or UNKNOWN (red)
   - Zone C (left): Traffic log showing all detections

### Method 2: Using Test Images

1. **Prepare a test image:**
   - Take a photo of a license plate
   - Save it as `test_plate.jpg` in the project directory
   - Or use any image with visible text

2. **Run the test script:**
```bash
python test_plate_detection.py
```

This will test:
- ✓ YOLOv8 model loading
- ✓ OCR functionality
- ✓ Whitelist lookup

### Method 3: Manual Testing with Sample Plates

1. **Add test plates to whitelist.csv:**
   - Edit `whitelist.csv` and add your test plate:
   ```csv
   plate,unit,resident_name,destination,source
   YOUR123,101,Test User,Building A,Live Detection
   ```

2. **Test the workflow:**
   - Start the server
   - Point camera at the plate (or use test image)
   - System should detect plate → OCR → lookup → show AUTHORIZED (green)

### Method 4: Simulate Detection (For Development)

If you don't have a camera or test images, you can manually trigger detection by:

1. **Modify the code temporarily** to use a test image:
   - In `app.py`, modify `video_stream()` to load a test image
   - Or create a test endpoint that processes a specific image

## Testing Different Scenarios

### Scenario 1: Authorized Plate (Green Screen)
- Plate in whitelist.csv → Should show **AUTHORIZED** in green
- Shows resident name and unit number
- [OPEN GATE] button is enabled

### Scenario 2: Unknown Plate (Red Screen)
- Plate NOT in whitelist.csv → Should show **UNKNOWN** in red
- [DENY ENTRY] button is enabled

### Scenario 3: Guard Actions
- Click [OPEN GATE] → Logs "ENTERED" to traffic_log.csv
- Click [DENY ENTRY] → Logs "BLOCKED" to traffic_log.csv
- Check Zone C (left) to see the log entries

## Troubleshooting

### Camera Not Working
- Check if camera is available: `python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL'); cap.release()"`
- Try different camera index: Set `CAMERA_SOURCE=1` or `2` in environment
- For RTSP camera: `export CAMERA_SOURCE="rtsp://your-camera-url"`

### No Detections
- **Normal**: YOLOv8 doesn't detect license plates by default (it detects general objects)
- For production, you need a license plate-specific model
- For testing, you can manually crop a plate region and test OCR

### OCR Not Working
- Make sure the plate text is clear and readable
- Try with a high-contrast image (dark text on light background)
- Check that `ocrmac` is installed: `pip list | grep ocrmac`

### Whitelist Not Found
- Check that `whitelist.csv` exists in the project directory
- Verify the plate format matches (normalized: uppercase, alphanumeric only)
- Example: "ABC-123" becomes "ABC123" in lookup

## Expected Behavior

✅ **Working correctly when:**
- Server starts without errors
- Dashboard loads in browser
- Camera feed shows in Zone A (or "Camera not available" message)
- Status bar shows "SYSTEM: ONLINE" with green dot
- Traffic log updates every 3 seconds
- Decision panel updates every 1 second

## Next Steps for Production

1. **License Plate Detection Model:**
   - Train YOLOv8 on license plate dataset, OR
   - Use a pre-trained license plate detection model
   - Current setup uses general object detection (works for testing)

2. **Improve OCR Accuracy:**
   - Preprocess images (contrast, sharpening)
   - Use multiple OCR attempts with different settings
   - Add confidence thresholds

3. **Add More Features:**
   - Admin panel for managing whitelist
   - Export logs to PDF/Excel
   - Email/SMS notifications
   - Multiple camera support


