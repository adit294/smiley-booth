# 📸 Smiley Booth - Smart Photobooth

**CS445 Computational Photography - Final Project**

**Team:** Shobhit Sinha (ss194), Jay Goenka (jgoenka2), Adit Agarwal (adit3)

---

## 🎯 What is Smiley Booth?

Smiley Booth is a **smart photobooth** that automatically takes your photo when you:
1. **Stand in the center** of the camera frame
2. **Smile** for about 3 seconds

It also has **15 fun filters** to make your photos look cool!

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the photobooth
python smiley_booth.py
```

That's it! A window will open with your camera. Center yourself and smile!

---

## 🎮 Controls

| Key | What it does |
|-----|--------------|
| `SPACE` | Take a photo right now |
| `←` or `,` | Previous filter |
| `→` or `.` | Next filter |
| `1-9` | Jump to filter 1-9 |
| `Q` | Quit |

---

## 📁 Project Files Explained

Our project has **4 main files**. Here's what each one does:

### 1️⃣ `smiley_booth.py` - The Main App

**What it does:** This is the "brain" of the application. It connects everything together.

**How it works (step by step):**

```
1. Opens your webcam
2. Reads each video frame (30 times per second)
3. Sends frame to detection.py to find your face and smile
4. Sends frame to filters.py to apply cool effects
5. Shows everything on screen
6. When you smile long enough → saves the photo!
```

**Key parts:**
- `SmileyBooth` class - the main application
- `init_camera()` - turns on your webcam
- `trigger_capture()` - takes and saves the photo
- `draw_ui()` - draws the buttons, boxes, and info on screen
- `run()` - the main loop that keeps everything running

---

### 2️⃣ `detection.py` - Face & Smile Detection

**What it does:** Finds your face and figures out if you're smiling.

**The technology:** We use **MediaPipe Face Mesh** from Google. It finds **468 points** on your face!

```
         👁️          👁️        ← Eye landmarks
            
             👃               ← Nose landmark
    
    Point 61 → 👄 ← Point 291  ← Mouth corner landmarks
```

**How smile detection works:**

We measure 4 things to detect a smile:

| Feature | What we check | Why |
|---------|---------------|-----|
| **Mouth Width** | Distance between mouth corners | Smiles are WIDER |
| **Corner Lift** | Are corners above the center? | Smiles lift UP ↑ |
| **Mouth Opening** | Is mouth slightly open? | Smiles often show teeth |
| **Symmetry** | Are both sides equal? | Frowns are often uneven |

**The math (simplified):**
```
smile_score = (mouth_width × 0.35) + (corner_lift × 0.40) + (opening × 0.15) + (angle × 0.10)

If smile_score > 55% → You're smiling! ✓
```

**Centering check:**
- We find the center of your face
- We find the center of the camera frame
- If they're close (within 12%) → You're centered! ✓

---

### 3️⃣ `filters.py` - Creative Photo Effects

**What it does:** Makes your photos look artistic with 15 different filters.

**The filters and how they work:**

| Filter | How it's made |
|--------|---------------|
| **Pencil Sketch** | Convert to gray → Invert → Blur → Blend (looks like pencil drawing) |
| **Color Sketch** | Same as pencil but keep some original colors |
| **Glitch** | Split RGB colors → Shift them apart → Add noise blocks |
| **Thermal** | Convert to gray → Apply heat-map colors (red=hot, blue=cold) |
| **Pinhole** | Darken the edges → Blur the corners (old camera look) |
| **Vintage** | Add brown/yellow tint → Add film grain noise |
| **Pop Art** | Reduce colors to 6 → Make them super bright → Add black edges |
| **Neon** | Find edges → Color them bright → Add glow effect |
| **Cartoon** | Smooth the colors → Find edges → Combine them |
| **Emboss** | Apply a 3x3 pattern that makes things look 3D |
| **Watercolor** | Smooth colors multiple times → Add paper texture |
| **Noir** | Black & white → High contrast → Dark edges |
| **Cyberpunk** | Boost contrast → Add cyan/magenta colors → Add scan lines |
| **Vaporwave** | Shift colors to pink/purple → Add gradient → Add scan lines |

**Color spaces we use:**
- **BGR** - Normal color (Blue, Green, Red)
- **Grayscale** - Black and white
- **HSV** - Hue (color), Saturation (intensity), Value (brightness)
- **LAB** - Lightness and color channels (good for contrast)

---

### 4️⃣ `requirements.txt` - What You Need to Install

```
opencv-python        → For camera and image processing
opencv-contrib-python → Extra OpenCV features
numpy                → For math operations on images
mediapipe            → For face detection (Google's AI)
Pillow               → Extra image support
```

---

## 🔄 How Everything Works Together

```
┌─────────────────────────────────────────────────────────────┐
│                    smiley_booth.py                          │
│                    (Main Controller)                        │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   WEBCAM    │ │ detection.py│ │  filters.py │
    │   (Input)   │ │ (Find Face) │ │  (Effects)  │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           │               │               │
           ▼               ▼               ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    YOUR SCREEN                          │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │                                                 │   │
    │  │    [Centering Guide]     [Filter Name]         │   │
    │  │                                                 │   │
    │  │              ┌─────────┐                       │   │
    │  │              │  YOUR   │                       │   │
    │  │              │  FACE   │                       │   │
    │  │              │  HERE   │                       │   │
    │  │              └─────────┘                       │   │
    │  │                                                 │   │
    │  │    [Smile: YES/NO]    [Confidence Bar]         │   │
    │  │                                                 │   │
    │  │  [Filter 1][Filter 2][Filter 3]...[Filter 15]  │   │
    │  └─────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘
```

---

## 📷 The Auto-Capture Process

```
Step 1: Camera reads your face
            ↓
Step 2: Are you CENTERED?
        ├── NO → Show arrow (← LEFT, RIGHT →, etc.)
        └── YES → Continue to Step 3
            ↓
Step 3: Are you SMILING?
        ├── NO → Show "Smile: No (need 55%)"
        └── YES → Start counting!
            ↓
Step 4: Keep smiling for 80 frames (~3 seconds)
        ├── Stopped smiling? → Reset counter to 0
        └── Still smiling? → Counter goes up
            ↓
Step 5: Counter reaches 80?
        └── YES → 📸 FLASH! Photo saved!
            ↓
Step 6: Wait 45 frames (~1.5 sec) before next photo
```

---

## 🎨 Understanding the Filters (Technical)

### Color Spaces

**BGR (Blue-Green-Red):**
- How computers store color images
- Each pixel has 3 values: B, G, R (0-255 each)
- Example: Pure red = (0, 0, 255)

**Grayscale:**
- Just brightness, no color
- Each pixel is one value (0=black, 255=white)

**HSV (Hue-Saturation-Value):**
- H = What color (0-180: red→yellow→green→cyan→blue→magenta)
- S = How vivid (0=gray, 255=pure color)
- V = How bright (0=dark, 255=bright)

### Common Operations

**Gaussian Blur:** Smooths the image by averaging nearby pixels
```python
blurred = cv2.GaussianBlur(image, (21, 21), 0)
#                          size of blur area ↑
```

**Edge Detection (Canny):** Finds outlines in images
```python
edges = cv2.Canny(gray_image, 50, 150)
#                 low threshold ↑   ↑ high threshold
```

**Color Conversion:**
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Color → Gray
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # BGR → HSV
```

**Blending Two Images:**
```python
result = cv2.addWeighted(image1, 0.7, image2, 0.3, 0)
#                        weight ↑         ↑ weight (must add to 1.0)
```

---

## 📂 Where Photos Are Saved

All captured photos go to:
```
captured_photos/
├── smiley_booth_20241208_143052_original.jpg   ← Original photo
├── smiley_booth_20241208_143052_vintage.jpg    ← With filter applied
├── smiley_booth_20241208_143055_original.jpg
├── smiley_booth_20241208_143055_neon.jpg
└── ...
```

The filename format: `smiley_booth_DATE_TIME_FILTERNAME.jpg`

---

## ❓ Troubleshooting

**Camera not working?**
```bash
# Try a different camera
python smiley_booth.py --camera 1
```

**Smile not detected?**
- Make sure your face is well-lit
- Look directly at the camera
- Try a natural smile (not forced!)

**Too slow?**
- Close other apps using the camera
- The filters work in real-time, some are slower than others

---

## 🎓 What We Learned

1. **Computer Vision:** How to use OpenCV and MediaPipe
2. **Face Detection:** Using 468 landmark points to find facial features
3. **Image Processing:** Converting between color spaces, applying filters
4. **Real-time Processing:** Making everything work at 30 FPS
5. **Software Design:** Organizing code into modules (detection, filters, main app)

---

## 📚 Libraries Used

| Library | What it does |
|---------|--------------|
| **OpenCV** | Camera capture, image processing, drawing on images |
| **MediaPipe** | AI-powered face detection with 468 landmarks |
| **NumPy** | Fast math operations on image arrays |

---

## 🏆 Credits

- **OpenCV** - opencv.org
- **MediaPipe** - Google's face detection AI
- **CS445 Course Staff** - For guidance and support

---

Made with ❤️ for CS445 Computational Photography
