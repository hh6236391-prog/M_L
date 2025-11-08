import cv2
import numpy as np
from skimage import exposure, filters, measure
from scipy import ndimage
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityMetrics:
    def __init__(self, config: Dict):
        self.config = config
        self.thresholds = config.get('thresholds', {})
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Multiple sharpness measures
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Additional sharpness metric using Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            sobel_var = np.var(sobel_magnitude)
            
            # Combine metrics
            sharpness = (laplacian_var + sobel_var) / 2
            return max(sharpness, 0.0)
            
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.0
    
    def calculate_brightness(self, image: np.ndarray) -> Tuple[float, str]:
        """Calculate brightness and exposure assessment"""
        try:
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                brightness = np.mean(hsv[:,:,2])
            else:
                brightness = np.mean(image)
            
            # Exposure assessment
            if brightness < self.thresholds.get('min_brightness', 30):
                exposure_status = "underexposed"
            elif brightness > self.thresholds.get('max_brightness', 220):
                exposure_status = "overexposed"
            else:
                exposure_status = "well_exposed"
                
            return brightness, exposure_status
            
        except Exception as e:
            logger.warning(f"Brightness calculation failed: {e}")
            return 0.0, "error"
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast using standard deviation"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            contrast = gray.std()
            return max(contrast, 0.0)
            
        except Exception as e:
            logger.warning(f"Contrast calculation failed: {e}")
            return 0.0
    
    def assess_white_balance(self, image: np.ndarray) -> float:
        """Assess white balance quality"""
        try:
            if len(image.shape) != 3:
                return 0.5  # Neutral score for grayscale
            
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Calculate mean values for A and B channels
            a_mean = np.mean(a)
            b_mean = np.mean(b)
            
            # Ideal values should be around 128 for neutral balance
            a_deviation = abs(a_mean - 128)
            b_deviation = abs(b_mean - 128)
            
            # Normalize to 0-1 score (lower deviation = better)
            max_deviation = 128
            a_score = 1.0 - (a_deviation / max_deviation)
            b_score = 1.0 - (b_deviation / max_deviation)
            
            white_balance_score = (a_score + b_score) / 2
            return max(white_balance_score, 0.0)
            
        except Exception as e:
            logger.warning(f"White balance assessment failed: {e}")
            return 0.5
    
    def detect_noise(self, image: np.ndarray) -> float:
        """Detect noise level in image"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Use wavelet-based noise estimation
            noise_level = np.mean(cv2.blur(gray, (3, 3)) - gray)
            noise_score = 1.0 - min(abs(noise_level) / 50.0, 1.0)
            
            return max(noise_score, 0.0)
            
        except Exception as e:
            logger.warning(f"Noise detection failed: {e}")
            return 0.5
    
    def detect_artifacts(self, image: np.ndarray) -> float:
        """Detect compression artifacts and other issues"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect blocking artifacts (common in JPEG compression)
            horizontal_diff = np.diff(gray, axis=1)
            vertical_diff = np.diff(gray, axis=0)
            
            # High differences at block boundaries indicate artifacts
            block_size = 8
            artifact_score = 0.0
            
            # Check for periodic patterns in differences
            for i in range(0, horizontal_diff.shape[0], block_size):
                for j in range(0, horizontal_diff.shape[1], block_size):
                    if i + block_size < horizontal_diff.shape[0] and j + block_size < horizontal_diff.shape[1]:
                        block_var = np.var(horizontal_diff[i:i+block_size, j:j+block_size])
                        artifact_score += block_var
            
            artifact_score /= (horizontal_diff.shape[0] * horizontal_diff.shape[1] / (block_size * block_size))
            
            # Normalize to 0-1 scale (lower artifacts = higher score)
            artifact_quality = 1.0 - min(artifact_score / 1000.0, 1.0)
            return max(artifact_quality, 0.0)
            
        except Exception as e:
            logger.warning(f"Artifact detection failed: {e}")
            return 0.5
    
    def assess_composition(self, image: np.ndarray, face_bbox: Optional[Tuple] = None) -> float:
        """Assess image composition quality"""
        try:
            height, width = image.shape[:2]
            composition_score = 0.5  # Base score
            
            if face_bbox:
                x, y, w, h = face_bbox
                face_center_x = x + w / 2
                face_center_y = y + h / 2
                
                # Rule of thirds assessment
                third_x = width / 3
                third_y = height / 3
                
                # Check if face is near rule-of-thirds points
                vertical_thirds = abs(face_center_x - third_x) < third_x / 2 or abs(face_center_x - 2 * third_x) < third_x / 2
                horizontal_thirds = abs(face_center_y - third_y) < third_y / 2 or abs(face_center_y - 2 * third_y) < third_y / 2
                
                if vertical_thirds and horizontal_thirds:
                    composition_score += 0.3
                
                # Check if face is too close to edges
                margin = 0.1 * min(width, height)
                if (face_center_x > margin and face_center_x < width - margin and
                    face_center_y > margin and face_center_y < height - margin):
                    composition_score += 0.2
            
            return min(composition_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Composition assessment failed: {e}")
            return 0.5