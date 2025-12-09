"""
Filters Module for Smiley Booth
Creative image effects and artistic filters
"""

import cv2
import numpy as np
from typing import Tuple, Callable, Dict
import random
import time


class FilterManager:
    """
    Manages all creative filters for the photobooth
    """
    
    def __init__(self):
        self.filters: Dict[str, Callable] = {
            'normal': self.filter_normal,
            'pencil_sketch': self.filter_pencil_sketch,
            'color_sketch': self.filter_color_sketch,
            'glitch': self.filter_glitch,
            'thermal': self.filter_thermal,
            'pinhole': self.filter_pinhole,
            'vintage': self.filter_vintage,
            'pop_art': self.filter_pop_art,
            'neon': self.filter_neon,
            'cartoon': self.filter_cartoon,
            'emboss': self.filter_emboss,
            'watercolor': self.filter_watercolor,
            'noir': self.filter_noir,
            'cyberpunk': self.filter_cyberpunk,
            'vaporwave': self.filter_vaporwave,
        }
        
        self.filter_names = list(self.filters.keys())
        self.current_filter_index = 0
        
        # Glitch animation state
        self.glitch_offset = 0
        self.glitch_intensity = 0.5
        
    def get_current_filter_name(self) -> str:
        """Get current filter name"""
        return self.filter_names[self.current_filter_index]
    
    def next_filter(self):
        """Switch to next filter"""
        self.current_filter_index = (self.current_filter_index + 1) % len(self.filter_names)
    
    def prev_filter(self):
        """Switch to previous filter"""
        self.current_filter_index = (self.current_filter_index - 1) % len(self.filter_names)
    
    def set_filter(self, name: str):
        """Set filter by name"""
        if name in self.filters:
            self.current_filter_index = self.filter_names.index(name)
    
    def apply_current_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply the currently selected filter"""
        filter_func = self.filters[self.filter_names[self.current_filter_index]]
        return filter_func(frame)
    
    def apply_filter(self, frame: np.ndarray, filter_name: str) -> np.ndarray:
        """Apply a specific filter by name"""
        if filter_name in self.filters:
            return self.filters[filter_name](frame)
        return frame
    
    # ==================== FILTER IMPLEMENTATIONS ====================
    
    def filter_normal(self, frame: np.ndarray) -> np.ndarray:
        """No filter - return original"""
        return frame.copy()
    
    def filter_pencil_sketch(self, frame: np.ndarray) -> np.ndarray:
        """
        Pencil sketch effect using edge detection and blending
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Invert the grayscale image
        inverted = cv2.bitwise_not(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
        
        # Blend using color dodge
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        # Convert back to BGR
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        
        return sketch_bgr
    
    def filter_color_sketch(self, frame: np.ndarray) -> np.ndarray:
        """
        Color pencil sketch effect
        """
        # Get grayscale sketch
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        # Create colored version by blending with original
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        
        # Blend sketch with original colors
        color_sketch = cv2.addWeighted(frame, 0.4, sketch_bgr, 0.6, 0)
        
        # Enhance saturation
        hsv = cv2.cvtColor(color_sketch, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
        color_sketch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return color_sketch
    
    def filter_glitch(self, frame: np.ndarray) -> np.ndarray:
        """
        Digital glitch effect with RGB channel shifting and noise
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Update glitch animation
        self.glitch_offset = (self.glitch_offset + 1) % 30
        
        # Random horizontal slice shifts
        num_slices = random.randint(3, 8)
        for _ in range(num_slices):
            y_start = random.randint(0, h - 20)
            slice_height = random.randint(5, 30)
            shift = random.randint(-30, 30)
            
            y_end = min(y_start + slice_height, h)
            
            # Shift the slice
            if shift > 0:
                result[y_start:y_end, shift:w] = frame[y_start:y_end, :w-shift]
                result[y_start:y_end, :shift] = frame[y_start:y_end, w-shift:w]
            elif shift < 0:
                result[y_start:y_end, :w+shift] = frame[y_start:y_end, -shift:w]
                result[y_start:y_end, w+shift:w] = frame[y_start:y_end, :-shift]
        
        # RGB channel separation
        b, g, r = cv2.split(result)
        
        shift_r = random.randint(-10, 10)
        shift_b = random.randint(-10, 10)
        
        # Shift red channel
        M_r = np.float32([[1, 0, shift_r], [0, 1, 0]])
        r = cv2.warpAffine(r, M_r, (w, h))
        
        # Shift blue channel
        M_b = np.float32([[1, 0, shift_b], [0, 1, 0]])
        b = cv2.warpAffine(b, M_b, (w, h))
        
        result = cv2.merge([b, g, r])
        
        # Add scan lines
        for y in range(0, h, 4):
            result[y:y+1, :] = result[y:y+1, :] * 0.7
        
        # Add random noise blocks
        if random.random() > 0.7:
            block_x = random.randint(0, w - 50)
            block_y = random.randint(0, h - 30)
            block_w = random.randint(30, 100)
            block_h = random.randint(10, 40)
            
            noise = np.random.randint(0, 255, (block_h, block_w, 3), dtype=np.uint8)
            y_end = min(block_y + block_h, h)
            x_end = min(block_x + block_w, w)
            result[block_y:y_end, block_x:x_end] = noise[:y_end-block_y, :x_end-block_x]
        
        return result
    
    def filter_thermal(self, frame: np.ndarray) -> np.ndarray:
        """
        Thermal/infrared vision effect
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thermal colormap
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Enhance contrast
        lab = cv2.cvtColor(thermal, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        thermal = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return thermal
    
    def filter_pinhole(self, frame: np.ndarray) -> np.ndarray:
        """
        Pinhole camera/vignette zoom effect
        """
        h, w = frame.shape[:2]
        
        # Create vignette mask
        X = np.arange(0, w)
        Y = np.arange(0, h)
        X, Y = np.meshgrid(X, Y)
        
        center_x, center_y = w // 2, h // 2
        
        # Calculate distance from center
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        
        # Normalize and create vignette
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - (dist / max_dist) ** 1.5
        vignette = np.clip(vignette, 0, 1)
        
        # Apply subtle radial blur to edges
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        
        # Blend based on distance from center
        blend_mask = (dist / max_dist).clip(0, 1)
        blend_mask = np.stack([blend_mask] * 3, axis=-1)
        
        result = (frame * (1 - blend_mask * 0.5) + blurred * blend_mask * 0.5).astype(np.uint8)
        
        # Apply vignette
        vignette_3ch = np.stack([vignette] * 3, axis=-1)
        result = (result * vignette_3ch).astype(np.uint8)
        
        # Add slight sepia tone
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
        sepia = cv2.transform(result, sepia_filter)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        # Blend with original
        result = cv2.addWeighted(result, 0.7, sepia, 0.3, 0)
        
        return result
    
    def filter_vintage(self, frame: np.ndarray) -> np.ndarray:
        """
        Vintage/retro photo effect
        """
        # Reduce contrast and shift colors
        result = frame.copy().astype(np.float32)
        
        # Apply sepia tone
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
        result = cv2.transform(result, sepia_filter)
        
        # Add warm color cast
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)  # Red
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)  # Blue
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Reduce saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * 0.7).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add vignette
        h, w = result.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        vignette = 1 - (dist / max_dist) ** 2 * 0.5
        vignette = np.stack([vignette] * 3, axis=-1)
        result = (result * vignette).astype(np.uint8)
        
        # Add film grain
        noise = np.random.normal(0, 15, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def filter_pop_art(self, frame: np.ndarray) -> np.ndarray:
        """
        Pop art style with posterization and bold colors
        """
        # Posterize the image
        n_colors = 6
        
        # Reduce colors using k-means style quantization
        result = frame.copy()
        
        # Simple posterization
        result = (result // (256 // n_colors)) * (256 // n_colors)
        
        # Enhance saturation dramatically
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add edge outlines
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None)
        
        # Overlay black edges
        result[edges > 0] = [0, 0, 0]
        
        return result
    
    def filter_neon(self, frame: np.ndarray) -> np.ndarray:
        """
        Neon glow effect with edge highlighting
        """
        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges for glow effect
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Create colored glow
        glow = np.zeros_like(frame)
        
        # Multi-colored neon effect
        glow[:, :, 0] = edges_dilated  # Blue
        glow[:, :, 1] = cv2.dilate(edges, kernel, iterations=1)  # Green  
        glow[:, :, 2] = edges  # Red (pink/magenta)
        
        # Apply Gaussian blur for glow
        glow = cv2.GaussianBlur(glow, (15, 15), 0)
        
        # Dark background
        dark_bg = (frame * 0.2).astype(np.uint8)
        
        # Combine
        result = cv2.addWeighted(dark_bg, 1, glow, 2, 0)
        
        return result
    
    def filter_cartoon(self, frame: np.ndarray) -> np.ndarray:
        """
        Cartoon/cel-shading effect
        """
        # Reduce noise while keeping edges sharp
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 9)
        
        # Posterize colors
        color = (color // 32) * 32
        
        # Combine color with edges
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_colored)
        
        return cartoon
    
    def filter_emboss(self, frame: np.ndarray) -> np.ndarray:
        """
        Emboss/relief effect
        """
        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])
        
        # Apply emboss
        embossed = cv2.filter2D(frame, -1, kernel)
        
        # Add offset to bring values into visible range
        embossed = embossed + 128
        
        return np.clip(embossed, 0, 255).astype(np.uint8)
    
    def filter_watercolor(self, frame: np.ndarray) -> np.ndarray:
        """
        Watercolor painting effect
        """
        # Apply bilateral filter multiple times for smooth color regions
        result = frame.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        
        # Reduce saturation slightly
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * 0.8).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add paper texture effect
        h, w = result.shape[:2]
        noise = np.random.normal(0, 10, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (5, 5), 0)
        noise = np.stack([noise] * 3, axis=-1)
        
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def filter_noir(self, frame: np.ndarray) -> np.ndarray:
        """
        Film noir black and white with high contrast
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Add dramatic shadows
        gray = np.clip(gray * 1.3 - 30, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Add slight blue tint for cold feeling
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.1, 0, 255).astype(np.uint8)
        
        # Add vignette
        h, w = result.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        vignette = 1 - (dist / max_dist) ** 1.5 * 0.6
        vignette = np.stack([vignette] * 3, axis=-1)
        result = (result * vignette).astype(np.uint8)
        
        return result
    
    def filter_cyberpunk(self, frame: np.ndarray) -> np.ndarray:
        """
        Cyberpunk aesthetic with neon colors and high contrast
        """
        # Increase contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Shift colors toward cyan/magenta
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        
        # Boost saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add color tint (cyan shadows, magenta highlights)
        b, g, r = cv2.split(result)
        
        # Cyan in shadows
        b = np.clip(b.astype(np.float32) + 30, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) + 15, 0, 255).astype(np.uint8)
        
        # Magenta in highlights
        mask = result.mean(axis=2) > 128
        r[mask] = np.clip(r[mask].astype(np.float32) + 40, 0, 255).astype(np.uint8)
        b[mask] = np.clip(b[mask].astype(np.float32) + 20, 0, 255).astype(np.uint8)
        
        result = cv2.merge([b, g, r])
        
        # Add scan lines
        h, w = result.shape[:2]
        for y in range(0, h, 3):
            result[y:y+1, :] = (result[y:y+1, :] * 0.8).astype(np.uint8)
        
        return result
    
    def filter_vaporwave(self, frame: np.ndarray) -> np.ndarray:
        """
        Vaporwave aesthetic with pink/purple/cyan palette
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Shift hue toward pink/purple
        hsv[:, :, 0] = (hsv[:, :, 0] + 150) % 180
        
        # Boost saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add gradient overlay
        h, w = result.shape[:2]
        gradient = np.zeros((h, w, 3), dtype=np.float32)
        
        for y in range(h):
            ratio = y / h
            # Pink to cyan gradient
            gradient[y, :, 0] = 255 * ratio  # Blue increases
            gradient[y, :, 1] = 100 * (1 - ratio)  # Green decreases
            gradient[y, :, 2] = 255 * (1 - ratio)  # Red decreases
        
        # Blend gradient
        result = cv2.addWeighted(result, 0.7, gradient.astype(np.uint8), 0.3, 0)
        
        # Add horizontal scan lines for retro feel
        for y in range(0, h, 4):
            result[y:y+2, :] = (result[y:y+2, :] * 0.85).astype(np.uint8)
        
        return result


def create_filter_preview_strip(frame: np.ndarray, filter_manager: FilterManager, 
                                 preview_height: int = 80) -> np.ndarray:
    """
    Create a strip showing previews of all available filters
    """
    h, w = frame.shape[:2]
    preview_width = w // len(filter_manager.filter_names)
    
    # Resize frame for preview
    small_frame = cv2.resize(frame, (preview_width, preview_height))
    
    strip = np.zeros((preview_height + 30, w, 3), dtype=np.uint8)
    
    for i, filter_name in enumerate(filter_manager.filter_names):
        x_start = i * preview_width
        x_end = x_start + preview_width
        
        # Apply filter to small frame
        filtered = filter_manager.apply_filter(small_frame, filter_name)
        strip[:preview_height, x_start:x_end] = filtered
        
        # Highlight current filter
        if i == filter_manager.current_filter_index:
            cv2.rectangle(strip, (x_start, 0), (x_end, preview_height), (0, 255, 0), 3)
        
        # Add filter name
        label = filter_name[:8]  # Truncate long names
        cv2.putText(strip, label, (x_start + 5, preview_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return strip

