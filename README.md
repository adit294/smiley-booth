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

| Filter | Technical Implementation |
|--------|-------------------------|
| **Normal** | Pass-through, no transformation applied |
| **Pencil Sketch** | BGR→Grayscale conversion, bitwise inversion, Gaussian blur (21×21 kernel), color dodge blending via `cv2.divide(gray, 255-blurred, scale=256)` |
| **Color Sketch** | Grayscale sketch + BGR original blended with `cv2.addWeighted(0.4, 0.6)`, HSV saturation channel multiplied by 1.3× |
| **Glitch** | RGB channel separation via `cv2.split()`, per-channel affine warp displacement (±10px), random horizontal slice shifts, scan line overlay (every 4th row at 70% brightness), random noise block injection |
| **Thermal** | BGR→Grayscale, `cv2.applyColorMap(COLORMAP_JET)`, BGR→LAB conversion, CLAHE on L-channel (clipLimit=3.0, 8×8 tiles) |
| **Pinhole** | Euclidean distance mask from center, radial vignette `1-(dist/max)^1.5`, Gaussian blur (15×15) blended at edges, sepia matrix transform `[[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]]` |
| **Vintage** | Sepia color matrix transformation, R-channel ×1.1, B-channel ×0.9, HSV saturation ×0.7, quadratic vignette falloff, Gaussian noise (σ=15) |
| **Pop Art** | Color quantization to 6 levels via integer division `(px//42)*42`, HSV saturation ×2.0, value ×1.2, Canny edge detection (100,200 thresholds), dilated black edge overlay |
| **Neon** | BGR→Grayscale, Canny edges (50,150), morphological dilation (3×3 kernel, 2 iterations), BGR channel assignment from edges, Gaussian blur glow (15×15), dark background blend (original ×0.2) |
| **Cartoon** | Bilateral filter (d=9, σ_color=300, σ_space=300), median blur (7×7) on grayscale, adaptive threshold (block=9, C=9), color posterization `(px//32)*32`, bitwise AND with edge mask |
| **Emboss** | 3×3 convolution kernel `[[-2,-1,0],[-1,1,1],[0,1,2]]` via `cv2.filter2D()`, +128 offset for visibility |
| **Watercolor** | Triple bilateral filter pass (d=9, σ=75), HSV saturation ×0.8, Gaussian noise texture (σ=10) with 5×5 blur |
| **Noir** | BGR→Grayscale, CLAHE (clipLimit=4.0, 8×8 tiles), contrast curve `gray×1.3-30`, B-channel ×1.1 for cold tint, power vignette `1-(dist/max)^1.5 × 0.6` |
| **Cyberpunk** | BGR→LAB, CLAHE on L-channel, HSV saturation ×1.5, B+30/G+15 global shift, conditional R+40/B+20 on bright pixels (mean>128), scan lines every 3rd row at 80% |
| **Vaporwave** | HSV hue rotation +150° (mod 180), saturation ×1.4, vertical BGR gradient overlay (pink→cyan), `cv2.addWeighted(0.7, 0.3)` blend, horizontal scan lines every 4th row at 85% |

### 📷 Capture Modes
- **Auto Capture:** Automatically captures when centered and smiling
- **Manual Capture:** Press SPACE to take a photo instantly

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
| `SPACE` | Take photo manually |
| `←` / `,` | Previous filter |
| `→` / `.` | Next filter |
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

