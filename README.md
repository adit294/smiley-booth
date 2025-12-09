# Smiley Booth - Smart Photobooth 📸

**CS445 Computational Photography - Final Project**

**Team:** Shobhit Sinha (ss194), Jay Goenka (jgoenka2), Adit Agarwal (adit3)

---

## Overview

Smiley Booth is an intelligent photobooth application that uses computer vision to automatically capture photos when the user is:
1. **Centered** in the frame
2. **Smiling**

The application features real-time face detection, smile recognition, and 15 creative artistic filters.

## Features

### 🎯 Smart Detection
- **Face Detection:** Uses Haar Cascades for robust face detection
- **Smile Detection:** Combines Haar Cascades and MediaPipe for accurate smile recognition
- **Centering Feedback:** Visual guides help users position themselves correctly
- **Temporal Smoothing:** Reduces false positives for stable detection

### 🎨 Creative Filters (15 Total)
| Filter | Description |
|--------|-------------|
| Normal | No effect - original image |
| Pencil Sketch | Black and white pencil drawing |
| Color Sketch | Colored pencil effect |
| Glitch | Digital glitch with RGB shifting |
| Thermal | Infrared/heat vision effect |
| Pinhole | Vignette with radial blur |
| Vintage | Retro sepia with film grain |
| Pop Art | Bold posterized colors |
| Neon | Glowing edge highlights |
| Cartoon | Cel-shading effect |
| Emboss | 3D relief texture |
| Watercolor | Soft painting effect |
| Noir | High contrast black & white |
| Cyberpunk | Neon cyan/magenta aesthetic |
| Vaporwave | Pink/purple retro gradient |

### 📷 Capture Modes
- **Auto Mode:** Automatically captures when centered and smiling
- **Manual Mode:** 3-second countdown triggered by spacebar

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam

### Setup

1. **Clone/Download the project:**
```bash
cd /Users/adit/Downloads/cs445_project
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Photobooth
```bash
python smiley_booth.py
```

### Command Line Options
```bash
python smiley_booth.py --help

# Options:
#   --camera, -c    Camera device ID (default: 0)
#   --output, -o    Output directory (default: captured_photos)
#   --demo, -d      Run filter demo mode
#   --image, -i     Image file for demo mode
```

### Examples
```bash
# Use default camera
python smiley_booth.py

# Use external camera (ID 1)
python smiley_booth.py --camera 1

# Save photos to custom folder
python smiley_booth.py --output my_photos

# Test filters without camera
python smiley_booth.py --demo

# Test filters on an image
python smiley_booth.py --demo --image sample.jpg
```

---

## Controls

| Key | Action |
|-----|--------|
| `H` | Toggle help overlay |
| `SPACE` | Take photo / Start countdown |
| `M` | Toggle Auto/Manual mode |
| `←` / `,` | Previous filter |
| `→` / `.` | Next filter |
| `F` | Toggle filter preview strip |
| `S` | Save current frame |
| `R` | Reset detection |
| `1-9` | Quick filter selection |
| `Q` / `ESC` | Quit |

---

## How It Works

### Face Detection Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  Webcam     │───▶│ Face         │───▶│ Centering     │
│  Feed       │    │ Detection    │    │ Check         │
└─────────────┘    └──────────────┘    └───────────────┘
                          │                    │
                          ▼                    ▼
                   ┌──────────────┐    ┌───────────────┐
                   │ Smile        │───▶│ Auto-Capture  │
                   │ Detection    │    │ Controller    │
                   └──────────────┘    └───────────────┘
                                              │
                                              ▼
                                       ┌───────────────┐
                                       │ Apply Filter  │
                                       │ & Save Photo  │
                                       └───────────────┘
```

### Detection Methods

1. **Haar Cascades:** Fast, reliable face and smile detection
2. **MediaPipe Face Mesh:** 468 facial landmarks for precise smile analysis
3. **Hybrid Approach:** Combines both for robust detection

### Smile Detection Logic
- Analyzes mouth aspect ratio (wider = smiling)
- Measures lip corner elevation
- Uses temporal smoothing (5-frame history)
- Requires consistent smile for capture trigger

---

## Project Structure

```
cs445_project/
├── smiley_booth.py    # Main application
├── detection.py       # Face & smile detection module
├── filters.py         # Creative image filters
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── captured_photos/   # Output directory (created automatically)
```

---

## Technical Details

### Dependencies
- **OpenCV (cv2):** Image processing, face detection, webcam capture
- **MediaPipe:** Advanced facial landmark detection
- **NumPy:** Numerical operations for filters
- **Pillow:** Additional image processing support

### Performance
- Target framerate: 30 FPS
- Face detection: ~10-15ms per frame
- Filter application: 5-50ms depending on filter complexity
- Overall latency: <100ms for responsive experience

---

## Evaluation Criteria

1. **Detection Accuracy:** Face and smile detection under various lighting
2. **Filter Quality:** Visual appeal of artistic effects
3. **User Experience:** Responsive interface and helpful feedback
4. **Robustness:** Stable operation across different users and environments

---

## Troubleshooting

### Camera not detected
```bash
# List available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# Try different camera ID
python smiley_booth.py --camera 1
```

### Slow performance
- Close other applications using the camera
- Reduce resolution in the code
- Disable filter preview strip with `F` key

### False smile detection
- Ensure good lighting on face
- Face the camera directly
- Smile naturally (exaggerated smiles may not be detected)

### MediaPipe installation issues (Apple Silicon)
```bash
pip install mediapipe-silicon  # For M1/M2 Macs
```

---

## Future Improvements

- [ ] Multi-face support
- [ ] Photo collage mode
- [ ] Social media sharing
- [ ] Custom filter creation
- [ ] Video recording mode
- [ ] Touch screen support

---

## License

This project is created for educational purposes as part of CS445 Computational Photography course.

---

## Acknowledgments

- OpenCV team for computer vision tools
- MediaPipe team for facial landmark detection
- CS445 course staff for guidance and support

