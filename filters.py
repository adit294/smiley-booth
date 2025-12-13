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
    Photobooth Filters
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
        
        self.glitch_offset = 0
        self.glitch_intensity = 0.5
        
    def get_current_filter_name(self) -> str:
        return self.filter_names[self.current_filter_index]
    
    def next_filter(self):
        self.current_filter_index = (self.current_filter_index + 1) % len(self.filter_names)
    
    def prev_filter(self):
        self.current_filter_index = (self.current_filter_index - 1) % len(self.filter_names)
    
    def set_filter(self, name: str):
        if name in self.filters:
            self.current_filter_index = self.filter_names.index(name)
    
    def apply_current_filter(self, frame: np.ndarray) -> np.ndarray:
        filter_func = self.filters[self.filter_names[self.current_filter_index]]
        return filter_func(frame)
    
    def apply_filter(self, frame: np.ndarray, filter_name: str) -> np.ndarray:
        if filter_name in self.filters:
            return self.filters[filter_name](frame)
        return frame
    
    # helpers for the filters as a lot of them use vignette and sepia
    def _apply_sepia(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        # sepia tone. strength 0 to 1 blends original with sepia
        sepia_mat = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ], dtype=np.float32)

        base = img.astype(np.float32)
        sep = cv2.transform(base, sepia_mat)
        sep = np.clip(sep, 0, 255)

        if strength < 1.0:
            out = base * (1.0 - strength) + sep * strength
        else:
            out = sep

        return out.astype(np.uint8)


    def _apply_vignette(self, img: np.ndarray, power: float = 1.5, strength: float = 0.6) -> np.ndarray:
        # darken edges. power controls falloff curve, strength controls amount
        h, w = img.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        cx, cy = w / 2.0, h / 2.0

        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)

        mask = 1.0 - (dist / max_dist) ** power * strength
        mask = np.clip(mask, 0.0, 1.0)
        mask_3ch = np.dstack([mask, mask, mask])

        out = img.astype(np.float32) * mask_3ch
        return np.clip(out, 0, 255).astype(np.uint8)

    # all the filters
    
    def filter_normal(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()
    
    def filter_pencil_sketch(self, frame: np.ndarray) -> np.ndarray:
        # turn the frame into a pencil sketch by using an inverted blurred grayscale image to “dodge blend” edges into bright strokes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        return sketch_bgr
    
    def filter_color_sketch(self, frame: np.ndarray) -> np.ndarray:
        # make a pencil sketch, blend it back with the original colors, then boost saturation for a colored sketch look
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        color_sketch = cv2.addWeighted(frame, 0.4, sketch_bgr, 0.6, 0)
        
        hsv = cv2.cvtColor(color_sketch, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
        color_sketch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return color_sketch
    
    def filter_glitch(self, frame: np.ndarray) -> np.ndarray:
        # create a “digital glitch” by shifting random horizontal slices, offsetting RGB channels, adding scanlines, and with random noise
        result = frame.copy()
        h, w = frame.shape[:2]
        
        self.glitch_offset = (self.glitch_offset + 1) % 30
        
        num_slices = random.randint(3, 8)
        for _ in range(num_slices):
            y_start = random.randint(0, h - 20)
            slice_height = random.randint(5, 30)
            shift = random.randint(-30, 30)
            
            y_end = min(y_start + slice_height, h)
            
            if shift > 0:
                result[y_start:y_end, shift:w] = frame[y_start:y_end, :w-shift]
                result[y_start:y_end, :shift] = frame[y_start:y_end, w-shift:w]
            elif shift < 0:
                result[y_start:y_end, :w+shift] = frame[y_start:y_end, -shift:w]
                result[y_start:y_end, w+shift:w] = frame[y_start:y_end, :-shift]
        
        b, g, r = cv2.split(result)
        
        shift_r = random.randint(-10, 10)
        shift_b = random.randint(-10, 10)
        
        M_r = np.float32([[1, 0, shift_r], [0, 1, 0]])
        r = cv2.warpAffine(r, M_r, (w, h))
        
        M_b = np.float32([[1, 0, shift_b], [0, 1, 0]])
        b = cv2.warpAffine(b, M_b, (w, h))
        
        result = cv2.merge([b, g, r])
        
        for y in range(0, h, 4):
            result[y:y+1, :] = result[y:y+1, :] * 0.7
        
        # random noise 
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
        # fake a thermal camera by mapping grayscale intensity to a heat colormap, then boosting local contrast using CLAHE on the L channel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        lab = cv2.cvtColor(thermal, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        thermal = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return thermal
    
    def filter_pinhole(self, frame: np.ndarray) -> np.ndarray:
        # add a pinhole look by blurring the edges with a radial mask, then finishing with a vignette darkening and a light sepia tint
        h, w = frame.shape[:2]

        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        cx, cy = w // 2, h // 2
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)

        blurred = cv2.GaussianBlur(frame, (15, 15), 0)

        blend_mask = (dist / max_dist).clip(0, 1)
        blend_mask = np.stack([blend_mask] * 3, axis=-1)

        result = (frame * (1 - blend_mask * 0.5) + blurred * blend_mask * 0.5).astype(np.uint8)

        result = self._apply_vignette(result, power=1.5, strength=0.9)
        result = self._apply_sepia(result, strength=0.3)

        return result

    
    def filter_vintage(self, frame: np.ndarray) -> np.ndarray:
        # create a vintage photo look by applying sepia and warm color bias, lowering saturation, adding a vignette, and sprinkling in film grain noise
        result = self._apply_sepia(frame, strength=1.0).astype(np.float32)

        result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)
        result = result.astype(np.uint8)

        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * 0.7).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        result = self._apply_vignette(result, power=2.0, strength=0.5)

        noise = np.random.normal(0, 15, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return result

    
    def filter_pop_art(self, frame: np.ndarray) -> np.ndarray:
        # posterize the image, crank up saturation, and draw bold black edges for a pop art comic look
        n_colors = 6
        
        result = frame.copy()
        
        result = (result // (256 // n_colors)) * (256 // n_colors)
        
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None)
        
        result[edges > 0] = [0, 0, 0]
        
        return result
    
    def filter_neon(self, frame: np.ndarray) -> np.ndarray:
        # detect edges, turn them into glowing colored outlines, and blend them over a darkened background for a neon effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        glow = np.zeros_like(frame)
        
        glow[:, :, 0] = edges_dilated  #b
        glow[:, :, 1] = cv2.dilate(edges, kernel, iterations=1)  # g
        glow[:, :, 2] = edges  # r
        
        glow = cv2.GaussianBlur(glow, (15, 15), 0)
        
        dark_bg = (frame * 0.2).astype(np.uint8)
        result = cv2.addWeighted(dark_bg, 1, glow, 2, 0)
        
        return result
    
    def filter_cartoon(self, frame: np.ndarray) -> np.ndarray:
        # smooth colors with a bilateral filter, reduce color depth, and overlay sharp adaptive edges to get a cartoon style
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 9)
        
        color = (color // 32) * 32
        
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_colored)
        
        return cartoon
    
    def filter_emboss(self, frame: np.ndarray) -> np.ndarray:
        # apply an emboss kernel to highlight edges and shift intensities so the relief effect is visible
        kernel = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])

        embossed = cv2.filter2D(frame, -1, kernel)
        
        embossed = embossed + 128
        
        return np.clip(embossed, 0, 255).astype(np.uint8)
    
    def filter_watercolor(self, frame: np.ndarray) -> np.ndarray:
        # smooth colors with repeated bilateral filtering, lower saturation, and add soft noise to mimic watercolor paper texture
        result = frame.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * 0.8).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        h, w = result.shape[:2]
        noise = np.random.normal(0, 10, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (5, 5), 0)
        noise = np.stack([noise] * 3, axis=-1)
        
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def filter_noir(self, frame: np.ndarray) -> np.ndarray:
        # create a high-contrast black and white look with deep shadows, a cold tint, and a strong vignette

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = np.clip(gray * 1.3 - 30, 0, 255).astype(np.uint8)

        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.1, 0, 255).astype(np.uint8)
        result = self._apply_vignette(result, power=1.5, strength=0.6)

        return result

    
    def filter_cyberpunk(self, frame: np.ndarray) -> np.ndarray:
        # boost contrast and saturation, push colors toward cyan and magenta, and add scan lines for a cyberpunk neon look

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        b, g, r = cv2.split(result)
        
        b = np.clip(b.astype(np.float32) + 30, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) + 15, 0, 255).astype(np.uint8)
        
        mask = result.mean(axis=2) > 128
        r[mask] = np.clip(r[mask].astype(np.float32) + 40, 0, 255).astype(np.uint8)
        b[mask] = np.clip(b[mask].astype(np.float32) + 20, 0, 255).astype(np.uint8)
        
        result = cv2.merge([b, g, r])
        
        h, w = result.shape[:2]
        for y in range(0, h, 3):
            result[y:y+1, :] = (result[y:y+1, :] * 0.8).astype(np.uint8)
        
        return result
    
    def filter_vaporwave(self, frame: np.ndarray) -> np.ndarray:
        # shift hues toward pink and purple, boost saturation, blend in a vertical neon gradient, and add scan lines for a vaporwave vibe
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 0] = (hsv[:, :, 0] + 150) % 180
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        h, w = result.shape[:2]
        gradient = np.zeros((h, w, 3), dtype=np.float32)
        
        for y in range(h):
            ratio = y / h
            gradient[y, :, 0] = 255 * ratio 
            gradient[y, :, 1] = 100 * (1 - ratio)
            gradient[y, :, 2] = 255 * (1 - ratio) 
        
        result = cv2.addWeighted(result, 0.7, gradient.astype(np.uint8), 0.3, 0)
        
        for y in range(0, h, 4):
            result[y:y+2, :] = (result[y:y+2, :] * 0.85).astype(np.uint8)
        
        return result


def create_filter_preview_strip(frame: np.ndarray, filter_manager: FilterManager, 
                                 preview_height: int = 80) -> np.ndarray:
    # strip showing the filters
    h, w = frame.shape[:2]
    preview_width = w // len(filter_manager.filter_names)
    
    small_frame = cv2.resize(frame, (preview_width, preview_height))
    
    strip = np.zeros((preview_height + 30, w, 3), dtype=np.uint8)
    
    for i, filter_name in enumerate(filter_manager.filter_names):
        x_start = i * preview_width
        x_end = x_start + preview_width
        
        filtered = filter_manager.apply_filter(small_frame, filter_name)
        strip[:preview_height, x_start:x_end] = filtered
        
        if i == filter_manager.current_filter_index:
            cv2.rectangle(strip, (x_start, 0), (x_end, preview_height), (0, 255, 0), 3)
        
        label = filter_name[:8] 
        cv2.putText(strip, label, (x_start + 5, preview_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return strip

