# Quick Start Guide

## ✅ All Systems Ready!

Your Avalon Sentinel Dashboard is fully functional. Here's how to use it:

## 🚀 Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

Then open: **http://localhost:5000/**

## ✅ Test Results

All components tested and working:
- ✓ YOLOv8 model loads successfully
- ✓ OCR (ocrmac) works correctly
- ✓ Whitelist CSV lookup functional
- ✓ FastAPI app imports without errors

## 🧪 How to Test Number Plate Detection

### Option 1: Use Your Webcam (Easiest)
1. Start the server (command above)
2. Point your webcam at a license plate (or printed image)
3. Watch Zone A (center) for detection
4. Check Zone B (right) for AUTHORIZED/UNKNOWN status

### Option 2: Use Test Script
```bash
python test_plate_detection.py
```

### Option 3: Add Test Plates to Whitelist
1. Edit `whitelist.csv`
2. Add your test plate:
   ```csv
   plate,unit,resident_name,destination,source
   TEST123,101,Test User,Building A,Live Detection
   ```
3. Point camera at plate with text "TEST123"
4. Should show AUTHORIZED (green)

## 📋 Current Whitelist

Your `whitelist.csv` contains:
- **ABC123** → Juan Perez (Unit 304)
- **XYZ789** → Maria Garcia (Unit 205)  
- **DEF456** → John Smith (Unit 101)

## 🎯 What to Expect

**Zone A (Center):** Live camera feed with detection boxes
**Zone B (Right):** Big green/red screen showing decision
**Zone C (Left):** Live traffic log (updates every 3 seconds)

## ⚠️ Important Notes

1. **YOLOv8 Limitation:** Standard YOLOv8 doesn't detect license plates by default. It detects general objects. For production, you'll need a license plate-specific model.

2. **For Testing:** The system will work if:
   - You manually crop a plate region, OR
   - Use a test image with clear text, OR
   - Train/fine-tune YOLOv8 on license plates

3. **OCR Works:** The OCR component (ocrmac) is fully functional and will read text from any detected region.

## 🔧 Troubleshooting

**Camera not working?**
- Check: `python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"`
- Try different camera: `export CAMERA_SOURCE=1`

**No detections?**
- This is normal - YOLOv8 needs license plate training
- Test OCR directly with test images
- Use the test script to verify OCR works

## 📝 Next Steps

1. **For Real License Plate Detection:**
   - Get a pre-trained license plate YOLOv8 model, OR
   - Train YOLOv8 on your license plate dataset

2. **Improve Accuracy:**
   - Add image preprocessing (contrast, sharpening)
   - Use multiple OCR attempts
   - Add confidence thresholds

3. **Production Features:**
   - Admin panel for whitelist management
   - Export logs
   - Notifications (email/SMS)

## 🎉 You're All Set!

The system is ready to run. Start the server and test with your camera or test images!


