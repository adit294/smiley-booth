"""
Detection Module for Smiley Booth
Handles face detection, centering feedback, and smile detection
Using MediaPipe Face Mesh for all detection
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math


@dataclass
class FaceData:
    """Data class to hold face detection results"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    is_centered: bool
    is_smiling: bool
    smile_confidence: float
    landmarks: Optional[List] = None


class FaceDetector:
    """
    Face and Smile Detector using MediaPipe Face Mesh
    Uses 468 facial landmarks for accurate detection
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Smile detection parameters
        self.smile_threshold = 0.55  # Threshold for smile detection
        self.min_confidence_for_capture = 0.5  # Minimum confidence needed
        self.center_tolerance = 0.12  # 12% tolerance from center
        
        # Smoothing for stability
        self.smile_history = []
        self.history_size = 8  # Frames for stability
        
    def _get_landmark_point(self, landmarks, idx, w, h) -> Tuple[float, float]:
        """Get landmark coordinates in pixels"""
        lm = landmarks.landmark[idx]
        return (lm.x * w, lm.y * h)
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_face_bbox_from_landmarks(self, landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Calculate face bounding box from MediaPipe landmarks
        Uses face oval landmarks (indices for face contour)
        """
        # Face oval landmark indices
        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        x_coords = []
        y_coords = []
        
        for idx in FACE_OVAL:
            pt = self._get_landmark_point(landmarks, idx, w, h)
            x_coords.append(pt[0])
            y_coords.append(pt[1])
        
        # Calculate bounding box with padding
        padding = 20
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def detect_smile_mediapipe(self, frame: np.ndarray, landmarks) -> Tuple[bool, float]:
        """
        Detect smile using MediaPipe landmarks
        Uses multiple geometric features for accurate smile detection
        """
        if landmarks is None:
            return False, 0.0
        
        h, w = frame.shape[:2]
        
        try:
            # ============ KEY LANDMARK INDICES ============
            # Mouth corners
            LEFT_CORNER = 61
            RIGHT_CORNER = 291
            
            # Upper lip
            UPPER_LIP_TOP = 13
            
            # Lower lip  
            LOWER_LIP_BOTTOM = 17
            
            # Mouth opening (inner lips)
            UPPER_INNER_LIP = 13
            LOWER_INNER_LIP = 14
            
            # Nose tip for reference
            NOSE_TIP = 4
            
            # Eye corners (for face width reference)
            LEFT_EYE_OUTER = 33
            RIGHT_EYE_OUTER = 263
            
            # Get key points
            left_corner = self._get_landmark_point(landmarks, LEFT_CORNER, w, h)
            right_corner = self._get_landmark_point(landmarks, RIGHT_CORNER, w, h)
            upper_lip = self._get_landmark_point(landmarks, UPPER_LIP_TOP, w, h)
            lower_lip = self._get_landmark_point(landmarks, LOWER_LIP_BOTTOM, w, h)
            upper_inner = self._get_landmark_point(landmarks, UPPER_INNER_LIP, w, h)
            lower_inner = self._get_landmark_point(landmarks, LOWER_INNER_LIP, w, h)
            nose_tip = self._get_landmark_point(landmarks, NOSE_TIP, w, h)
            left_eye = self._get_landmark_point(landmarks, LEFT_EYE_OUTER, w, h)
            right_eye = self._get_landmark_point(landmarks, RIGHT_EYE_OUTER, w, h)
            
            # ============ FEATURE 1: MOUTH ASPECT RATIO (MAR) ============
            # Wider mouth relative to height = smile
            mouth_width = self._distance(left_corner, right_corner)
            mouth_height = self._distance(upper_lip, lower_lip)
            
            # Normalize by face width
            face_width = self._distance(left_eye, right_eye)
            if face_width == 0:
                return False, 0.0
                
            normalized_mouth_width = mouth_width / face_width
            
            # MAR: higher when smiling (mouth gets wider)
            mar = normalized_mouth_width
            
            # ============ FEATURE 2: LIP CORNER ELEVATION ============
            # When smiling, corners lift up relative to center of mouth
            mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
            
            # Corners should be ABOVE (lower y value) the mouth center when smiling
            left_corner_lift = mouth_center_y - left_corner[1]
            right_corner_lift = mouth_center_y - right_corner[1]
            avg_corner_lift = (left_corner_lift + right_corner_lift) / 2
            
            # Normalize by face height
            face_height = self._distance(nose_tip, (nose_tip[0], 0))
            if face_height > 0:
                normalized_lift = avg_corner_lift / (face_height * 0.1)
            else:
                normalized_lift = 0
            
            # ============ FEATURE 3: MOUTH OPENING ============
            # Smiles often show teeth (mouth slightly open)
            mouth_opening = self._distance(upper_inner, lower_inner)
            normalized_opening = mouth_opening / face_width if face_width > 0 else 0
            
            # ============ FEATURE 4: CORNER ANGLE ============
            # Angle of mouth corners relative to horizontal
            # Positive angle = corners lifted (smile)
            
            # Left corner relative to mouth center
            left_angle = math.atan2(mouth_center_y - left_corner[1], 
                                    left_corner[0] - (left_corner[0] + right_corner[0])/2)
            right_angle = math.atan2(mouth_center_y - right_corner[1],
                                     (left_corner[0] + right_corner[0])/2 - right_corner[0])
            
            # Both angles should be positive for a smile
            angle_score = (left_angle + right_angle) / 2
            
            # ============ COMBINE FEATURES INTO SMILE SCORE ============
            
            # Score components (all should be positive for smile)
            mar_score = max(0, (mar - 0.35) * 3)  # MAR > 0.35 indicates smile
            lift_score = max(0, normalized_lift * 2)  # Corner lift
            opening_score = max(0, min(normalized_opening * 2, 0.3))  # Slight opening bonus
            angle_bonus = max(0, angle_score * 0.5) if angle_score > 0 else 0
            
            # Penalty for asymmetric mouth (frown or grimace)
            asymmetry_penalty = abs(left_corner_lift - right_corner_lift) / (face_width * 0.1 + 0.001)
            asymmetry_penalty = min(asymmetry_penalty * 0.3, 0.3)
            
            # Penalty if corners are below mouth center (frown)
            frown_penalty = 0
            if avg_corner_lift < -2:  # Corners below center
                frown_penalty = 0.5
            
            # Final smile score
            smile_score = (
                mar_score * 0.35 +      # Mouth width is important
                lift_score * 0.40 +      # Corner lift is most important
                opening_score * 0.15 +   # Small bonus for open mouth
                angle_bonus * 0.10       # Angle bonus
            ) - asymmetry_penalty - frown_penalty
            
            # Clamp to [0, 1]
            smile_confidence = max(0, min(smile_score, 1.0))
            
            # Apply strict threshold
            is_smiling = smile_confidence > self.smile_threshold
            
            return is_smiling, smile_confidence
            
        except (IndexError, AttributeError, ZeroDivisionError):
            return False, 0.0
    
    def check_centering(self, face_center: Tuple[int, int], frame_shape: Tuple[int, int]) -> bool:
        """Check if face is centered in frame"""
        frame_h, frame_w = frame_shape[:2]
        frame_center = (frame_w // 2, frame_h // 2)
        
        # Calculate tolerance in pixels
        tolerance_x = frame_w * self.center_tolerance
        tolerance_y = frame_h * self.center_tolerance
        
        dx = abs(face_center[0] - frame_center[0])
        dy = abs(face_center[1] - frame_center[1])
        
        return dx < tolerance_x and dy < tolerance_y
    
    def get_centering_direction(self, face_center: Tuple[int, int], frame_shape: Tuple[int, int]) -> str:
        """Get direction to move for centering"""
        frame_h, frame_w = frame_shape[:2]
        frame_center = (frame_w // 2, frame_h // 2)
        
        dx = face_center[0] - frame_center[0]
        dy = face_center[1] - frame_center[1]
        
        directions = []
        
        tolerance_x = frame_w * self.center_tolerance
        tolerance_y = frame_h * self.center_tolerance
        
        if abs(dx) > tolerance_x:
            if dx > 0:
                directions.append("← LEFT")
            else:
                directions.append("RIGHT →")
        
        if abs(dy) > tolerance_y:
            if dy > 0:
                directions.append("↑ UP")
            else:
                directions.append("DOWN ↓")
        
        return " & ".join(directions) if directions else "CENTERED ✓"
    
    def smooth_smile_detection(self, is_smiling: bool, confidence: float) -> Tuple[bool, float]:
        """Apply temporal smoothing to smile detection"""
        self.smile_history.append((is_smiling, confidence))
        
        if len(self.smile_history) > self.history_size:
            self.smile_history.pop(0)
        
        # Need sufficient history for reliable detection
        if len(self.smile_history) < self.history_size // 2:
            return False, confidence
        
        # Calculate smoothed values
        avg_confidence = sum(c for _, c in self.smile_history) / len(self.smile_history)
        smile_count = sum(1 for s, _ in self.smile_history if s)
        
        # Require MAJORITY of recent frames to show smile (70%)
        min_smile_frames = int(len(self.smile_history) * 0.7)
        smoothed_smiling = smile_count >= min_smile_frames
        
        # Also require minimum average confidence
        if avg_confidence < self.min_confidence_for_capture:
            smoothed_smiling = False
        
        return smoothed_smiling, avg_confidence
    
    def detect(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Main detection function - detects face and smile using MediaPipe
        Returns FaceData with all detection results
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        mp_results = self.face_mesh.process(rgb)
        
        # Check if face detected
        if not mp_results.multi_face_landmarks:
            self.smile_history.clear()
            return None
        
        landmarks = mp_results.multi_face_landmarks[0]
        
        # Get face bounding box from landmarks
        bbox = self.get_face_bbox_from_landmarks(landmarks, w, h)
        x, y, bw, bh = bbox
        face_center = (x + bw // 2, y + bh // 2)
        
        # Check centering
        is_centered = self.check_centering(face_center, frame.shape)
        
        # Only process smile if centered
        if not is_centered:
            self.smile_history.clear()
            return FaceData(
                bbox=bbox,
                center=face_center,
                is_centered=False,
                is_smiling=False,
                smile_confidence=0.0,
                landmarks=landmarks
            )
        
        # Detect smile
        is_smiling, confidence = self.detect_smile_mediapipe(frame, landmarks)
        
        # Apply smoothing
        smoothed_smile, smoothed_conf = self.smooth_smile_detection(is_smiling, confidence)
        
        return FaceData(
            bbox=bbox,
            center=face_center,
            is_centered=is_centered,
            is_smiling=smoothed_smile,
            smile_confidence=smoothed_conf,
            landmarks=landmarks
        )
    
    def draw_detection_overlay(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        """Draw detection visualization on frame"""
        overlay = frame.copy()
        
        if face_data is None:
            cv2.putText(
                overlay, "No face detected - Please position yourself",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            return overlay
        
        x, y, w, h = face_data.bbox
        
        # Draw face bounding box
        if face_data.is_centered and face_data.is_smiling:
            color = (0, 255, 0)  # Green when ready to capture
        elif face_data.is_centered:
            color = (0, 255, 255)  # Yellow when centered but not smiling
        else:
            color = (0, 165, 255)  # Orange when not centered
        
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
        
        # Draw center crosshair
        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = frame_w // 2, frame_h // 2
        crosshair_size = 30
        
        cv2.line(overlay, (center_x - crosshair_size, center_y), 
                 (center_x + crosshair_size, center_y), (255, 255, 255), 2)
        cv2.line(overlay, (center_x, center_y - crosshair_size), 
                 (center_x, center_y + crosshair_size), (255, 255, 255), 2)
        
        # Draw centering zone rectangle
        tol_x = int(frame_w * self.center_tolerance)
        tol_y = int(frame_h * self.center_tolerance)
        cv2.rectangle(overlay, 
                     (center_x - tol_x, center_y - tol_y),
                     (center_x + tol_x, center_y + tol_y),
                     (100, 100, 100), 1)
        
        # Draw face center point
        cv2.circle(overlay, face_data.center, 8, color, -1)
        
        # Draw centering guide
        direction = self.get_centering_direction(face_data.center, frame.shape)
        cv2.putText(
            overlay, direction,
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )
        
        # Draw smile indicator
        if face_data.is_centered:
            if face_data.is_smiling:
                smile_text = "Smile: YES!"
                smile_color = (0, 255, 0)
            else:
                smile_text = f"Smile: No (need {self.smile_threshold:.0%})"
                smile_color = (0, 0, 255)
        else:
            smile_text = "Smile: Center first!"
            smile_color = (0, 165, 255)
            
        cv2.putText(
            overlay, smile_text,
            (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, smile_color, 2
        )
        
        # Draw confidence bar
        bar_x, bar_y = 50, 115
        bar_width, bar_height = 200, 20
        
        # Background
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Threshold marker
        threshold_x = bar_x + int(bar_width * self.smile_threshold)
        cv2.line(overlay, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), (255, 255, 255), 2)
        
        # Confidence fill
        filled_width = int(bar_width * face_data.smile_confidence)
        bar_color = (0, 255, 0) if face_data.smile_confidence >= self.smile_threshold else (0, 100, 255)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
        
        # Border
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Confidence percentage
        conf_text = f"{face_data.smile_confidence:.0%}"
        cv2.putText(overlay, conf_text, (bar_x + bar_width + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay


class CaptureController:
    """
    Controls automatic photo capture based on smile and centering
    """
    
    def __init__(self, required_smile_frames: int = 150, cooldown_frames: int = 60):
        self.required_smile_frames = required_smile_frames
        self.cooldown_frames = cooldown_frames
        self.consecutive_smile_frames = 0
        self.cooldown_counter = 0
        self.capture_triggered = False
        
    def update(self, face_data: Optional[FaceData]) -> bool:
        """
        Update capture controller state
        Returns True if a capture should be triggered
        """
        self.capture_triggered = False
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        if face_data is None:
            self.consecutive_smile_frames = 0
            return False
        
        if face_data.is_centered and face_data.is_smiling:
            self.consecutive_smile_frames += 1
            
            if self.consecutive_smile_frames >= self.required_smile_frames:
                self.capture_triggered = True
                self.consecutive_smile_frames = 0
                self.cooldown_counter = self.cooldown_frames
                return True
        else:
            self.consecutive_smile_frames = 0
        
        return False
    
    def get_countdown(self) -> int:
        """Get countdown to capture (frames remaining)"""
        if self.consecutive_smile_frames > 0:
            return self.required_smile_frames - self.consecutive_smile_frames
        return -1
    
    def reset(self):
        """Reset the capture controller"""
        self.consecutive_smile_frames = 0
        self.cooldown_counter = 0
        self.capture_triggered = False
