# Smiley Booth: A Smart Photobooth with Real-Time Smile Detection and Creative Filters

**CS 445 Computational Photography - Final Project Report**

**Authors:** Shobhit Sinha (ss194), Jay Goenka (jgoenka2), Adit Agarwal (adit3)

**Date:** December 10, 2024

---

## 1. Motivation and Impact

### 1.1 Motivation

We chose to develop Smiley Booth because it represents a compelling intersection of computer vision and computational photography in a practical, interactive application. Traditional photobooths require manual operation or timed captures, often resulting in poorly framed shots or unflattering expressions. Our system addresses this by automating the capture process using facial landmark detection and expression recognition.

The project provided an opportunity to explore real-time image processing pipelines, specifically integrating facial landmark tracking, geometric smile detection, and dynamic image filtering. We aimed to deepen our understanding of how these techniques work together in a cohesive system while creating something visually engaging and functional.

### 1.2 Impact

Smart photobooths have applications beyond entertainment. Expression-triggered capture systems could be used in accessibility applications for users with motor impairments, automated portrait photography, video conferencing tools that capture screenshots at optimal moments, and research applications studying human expressions. The techniques we implemented—real-time face mesh analysis and image filtering—are foundational to many modern computer vision applications including augmented reality, facial analysis systems, and computational photography tools.

---

## 2. Approach

Our implementation consists of three primary modules: face and smile detection, creative image filters, and the main application controller.

### 2.1 Face Detection and Localization

We use MediaPipe Face Mesh, which provides 468 facial landmarks in real-time. This approach was chosen over traditional Haar cascades because it offers:
- Precise localization of facial features (eyes, nose, mouth corners, lips)
- Robust tracking across different lighting conditions
- Sub-pixel accuracy for geometric calculations

The face bounding box is computed from the face oval landmarks (36 points defining the face contour). We extract the minimum and maximum x,y coordinates with a 20-pixel padding to create the bounding box.

**Centering Detection:** We calculate the center of the detected face bounding box and compare it to the frame center. A 12% tolerance threshold determines whether the face is sufficiently centered:

```
tolerance_x = frame_width × 0.12
tolerance_y = frame_height × 0.12
is_centered = |face_center_x - frame_center_x| < tolerance_x AND
              |face_center_y - frame_center_y| < tolerance_y
```

### 2.2 Smile Detection Algorithm

Rather than using a binary classifier, we implemented a geometric approach that analyzes mouth shape using specific facial landmarks. Our algorithm computes four features:

**Feature 1: Mouth Aspect Ratio (MAR)**
The ratio of mouth width to face width. When smiling, the mouth stretches horizontally.
```
mouth_width = distance(landmark_61, landmark_291)
face_width = distance(landmark_33, landmark_263)
MAR = mouth_width / face_width
```

**Feature 2: Lip Corner Elevation**
We measure whether the mouth corners are above the vertical center of the mouth, which is characteristic of smiling.
```
mouth_center_y = (upper_lip_y + lower_lip_y) / 2
corner_lift = mouth_center_y - average(left_corner_y, right_corner_y)
```

**Feature 3: Mouth Opening**
Smiles often involve slight mouth opening showing teeth.
```
opening = distance(upper_inner_lip, lower_inner_lip)
normalized_opening = opening / face_width
```

**Feature 4: Corner Angle**
The angle of mouth corners relative to horizontal, where positive angles indicate upturned corners.

**Score Combination:**
```
smile_score = (MAR_score × 0.35) + (lift_score × 0.40) + 
              (opening_score × 0.15) + (angle_score × 0.10) - 
              asymmetry_penalty - frown_penalty
```

The asymmetry penalty reduces false positives from grimaces, while the frown penalty addresses cases where corners are below the mouth center. A threshold of 55% determines smile detection.

**Temporal Smoothing:** To reduce noise and false positives, we maintain an 8-frame history and require 70% of recent frames to indicate smiling before triggering capture.

### 2.3 Auto-Capture Controller

The capture controller requires both centering and smiling conditions to be met for 80 consecutive frames (approximately 2.7 seconds at 30 FPS) before triggering capture. This prevents accidental captures from brief expressions. After capture, a 45-frame cooldown prevents rapid successive captures.

### 2.4 Creative Filters

We implemented 15 image filters using various computational photography techniques:

| Filter | Technique |
|--------|-----------|
| Pencil Sketch | Grayscale conversion, inversion, Gaussian blur (21×21), color dodge blending |
| Color Sketch | Pencil sketch blended with original using addWeighted(0.4, 0.6), HSV saturation boost (×1.3) |
| Glitch | RGB channel separation, per-channel affine warping, horizontal slice displacement, scan line overlay, random noise block injection |
| Thermal | Grayscale conversion, COLORMAP_JET application, LAB color space CLAHE contrast enhancement |
| Pinhole | Euclidean distance-based vignette mask, radial Gaussian blur blending, sepia color matrix transformation |
| Vintage | Sepia matrix transform, RGB channel adjustment (R×1.1, B×0.9), HSV saturation reduction (×0.7), Gaussian noise (σ=15), vignette |
| Pop Art | Color quantization to 6 levels, HSV saturation boost (×2.0), Canny edge detection, black edge overlay |
| Neon | Canny edge detection, morphological dilation, multi-channel edge assignment, Gaussian blur glow, dark background blend |
| Cartoon | Bilateral filtering, median blur, adaptive thresholding, color posterization, edge mask combination |
| Emboss | 3×3 convolution kernel [[-2,-1,0],[-1,1,1],[0,1,2]], +128 offset |
| Watercolor | Triple bilateral filter pass, HSV saturation reduction, Gaussian noise texture |
| Noir | Grayscale, CLAHE contrast enhancement, contrast curve adjustment, blue tint, power vignette |
| Cyberpunk | LAB CLAHE, HSV saturation boost, conditional RGB shifts based on brightness, scan lines |
| Vaporwave | HSV hue rotation (+150°), saturation boost, vertical gradient overlay, scan lines |

---

## 3. Results

### 3.1 Detection Performance

We tested the system with multiple users under varying lighting conditions (natural daylight, indoor fluorescent, low-light). The face detection using MediaPipe proved robust across all conditions, successfully tracking faces even with partial occlusion or head rotation up to approximately 30 degrees.

The smile detection algorithm successfully distinguished between:
- Genuine smiles (detected correctly)
- Neutral expressions (correctly rejected)
- Frowns (correctly rejected due to negative corner lift)
- Asymmetric expressions (correctly rejected due to asymmetry penalty)

The centering feedback system provided intuitive visual guidance, allowing users to position themselves correctly within 2-3 seconds on average.

### 3.2 Filter Quality

All 15 filters produced visually distinct and aesthetically pleasing results. The real-time performance remained above 25 FPS for most filters, with computationally intensive filters (watercolor, cartoon) dropping to approximately 20 FPS due to multiple bilateral filter passes.

### 3.3 System Usability

The complete system runs in real-time on standard hardware (tested on MacBook with M1 chip). The interface provides clear feedback through:
- Color-coded bounding box (orange=not centered, yellow=centered but not smiling, green=ready to capture)
- Directional guidance text for centering
- Confidence bar showing smile detection progress
- Visual countdown during capture sequence
- Flash animation and preview thumbnail on successful capture

---

## 4. Implementation Details

### 4.1 Technical Stack

- **Programming Language:** Python 3.12
- **Computer Vision:** OpenCV 4.12.0, OpenCV-contrib 4.12.0
- **Face Detection:** MediaPipe 0.10.14 (Face Mesh with 468 landmarks)
- **Numerical Computing:** NumPy 2.2.6
- **Image Support:** Pillow 10.0+

### 4.2 Code Organization

```
smiley_booth/
├── smiley_booth.py    # Main application (404 lines)
│                      # - Camera initialization and capture loop
│                      # - UI rendering and keyboard handling
│                      # - Photo saving and flash effects
│
├── detection.py       # Face and smile detection (481 lines)
│                      # - MediaPipe Face Mesh integration
│                      # - Geometric smile analysis
│                      # - Centering calculations
│                      # - Temporal smoothing
│
├── filters.py         # Image filters (551 lines)
│                      # - 15 filter implementations
│                      # - Filter preview strip generation
│
├── requirements.txt   # Dependencies
└── captured_photos/   # Output directory
```

### 4.3 External Resources

- **MediaPipe Face Mesh:** Google's pre-trained face landmark model (used as-is, not retrained)
- **OpenCV Functions:** Standard library functions for image processing
- **No external datasets:** All testing performed with live webcam input

Our original contributions include:
- The complete smile detection algorithm using geometric analysis of mouth landmarks
- All 15 filter implementations
- The auto-capture logic with centering requirements and temporal smoothing
- The user interface and visual feedback system
- System integration and optimization

---

## 5. Challenge and Innovation

### 5.1 Technical Challenges Overcome

**Smile Detection Without Machine Learning Classification:**
Rather than training a classifier or using pre-built smile detection (which often produces false positives), we developed a geometric approach analyzing mouth shape. This required understanding the relationship between facial landmark positions and expressions, then translating that into quantifiable metrics. The challenge was balancing sensitivity (detecting genuine smiles) with specificity (rejecting non-smiles).

**Real-Time Performance:**
Processing 468 facial landmarks, computing geometric features, applying filters, and rendering UI elements while maintaining 25+ FPS required careful optimization. We implemented lazy evaluation for smile detection (only computed when centered) and efficient NumPy operations for filter computations.

**Temporal Stability:**
Raw frame-by-frame detection produced flickering results. We implemented a smoothing system requiring 70% agreement across 8 frames, which eliminated false triggers while remaining responsive to genuine smiles.

**Filter Implementation:**
Each filter required understanding specific image processing techniques and their mathematical foundations. The glitch filter involved channel separation and affine transforms; the thermal filter required color mapping and contrast enhancement; the cartoon filter combined edge detection with bilateral filtering.

### 5.2 Innovation

Our project demonstrates a novel combination of existing techniques into a cohesive application:
1. **Geometric smile detection** using mouth aspect ratio, corner elevation, and symmetry analysis rather than binary classification
2. **Conditional capture system** requiring both spatial (centering) and temporal (sustained smile) conditions
3. **15 distinct artistic filters** implemented from fundamental image processing operations
4. **Real-time visual feedback** guiding users through the capture process

### 5.3 Expected Points Justification

We believe this project merits **15-20 points** for the Innovation/Challenge component based on the following:

- The project implements multiple moderately complex techniques (face mesh processing, geometric expression analysis, 15 image filters) that together required significantly more than 15 hours per person
- We developed an original smile detection algorithm rather than using pre-built classifiers
- The system successfully integrates multiple components into a functional real-time application
- All filters were implemented from scratch using fundamental OpenCV operations
- The project addresses a practical application with clear utility

The project goes beyond implementing a single paper or basic technique, instead combining facial landmark analysis, geometric expression recognition, and computational photography filters into an integrated system. While the individual techniques are not novel, their combination and the specific implementation of geometric smile detection represent meaningful technical work.

---

## 6. Contributions

- **Shobhit Sinha:** Face detection implementation, smile detection algorithm design and optimization, MediaPipe integration, temporal smoothing logic
- **Adit Agarwal:** All 15 filter implementations, color space transformations, filter preview system, visual effects
- **Jay Goenka:** Main application development, UI/UX design, system integration, testing across lighting conditions, documentation

---

## 7. References

1. Lugaresi, C., et al. "MediaPipe: A Framework for Building Perception Pipelines." arXiv:1906.08172, 2019.
2. OpenCV Documentation. https://docs.opencv.org/
3. Bradski, G. "The OpenCV Library." Dr. Dobb's Journal of Software Tools, 2000.

---

## Code Repository

GitHub: https://github.com/adit294/smiley-booth

The repository includes complete source code, requirements.txt, and a README with usage instructions.

