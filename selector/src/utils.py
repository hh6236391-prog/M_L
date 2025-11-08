import cv2
import numpy as np
from pathlib import Path
from typing import List
import yaml
import logging
import os

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file if available"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return get_default_config()


def get_default_config() -> dict:
    """Default configuration tuned for best image selection"""
    return {
        'quality_weights': {
            # Weighted priorities — tuned for realistic photo capture
            'face_detection': 0.15,
            'sharpness': 0.25,        # Strongest factor (crisp detail)
            'focus_quality': 0.15,    # Ensures subject is sharp
            'pose_quality': 0.10,     # Prefer frontal head pose
            'composition': 0.08,      # Balanced frame placement
            'brightness': 0.05,       # Prevent overexposure bias
            'contrast': 0.07,         # Ensure depth and clarity
            'white_balance': 0.07,    # Maintain color accuracy
            'noise_level': 0.05,      # Penalize noisy low-light shots
            'artifacts': 0.03         # Reduce compressed/blurry artifacts
        },

        'thresholds': {
            'min_face_size': 0.08,     # Minimum 8% of frame
            'max_face_size': 0.75,     # Prevent cropped faces
            'min_sharpness': 40.0,     # Below this = blurry
            'min_brightness': 30.0,
            'max_brightness': 220.0,
            'min_contrast': 20.0,
            'min_pose_quality': 0.3
        },

        'computation': {
            'max_image_size': 1024     # Resize large photos before scoring
        }
    }

# -------------------------------------------------------------------
# IMAGE PROCESSING HELPERS
# -------------------------------------------------------------------
def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    if max_size is None:
        return image
    
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def validate_image(image: np.ndarray) -> np.ndarray:
    """Ensure image has correct format and data type"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# -------------------------------------------------------------------
# IMAGE COLLECTION HELPERS
# -------------------------------------------------------------------
def get_image_files(folder_path: str, extensions: List[str] = None, recursive: bool = True) -> List[Path]:
    """Fetch all valid image files from a folder (recursively if needed)"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.jfif']

    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    image_files = []
    patterns = []
    for ext in extensions:
        patterns.append(f'*{ext}')
        patterns.append(f'*{ext.upper()}')

    if recursive:
        for pattern in patterns:
            image_files.extend(folder.rglob(pattern))
    else:
        for pattern in patterns:
            image_files.extend(folder.glob(pattern))

    # Filter duplicates and validate
    image_files = list(set(image_files))
    valid_image_files = [img for img in image_files if _is_valid_image_file(img)]
    valid_image_files.sort()
    return valid_image_files


def _is_valid_image_file(file_path: Path) -> bool:
    """Verify if a file is a valid readable image"""
    try:
        if file_path.stat().st_size < 1024:
            return False
        from PIL import Image
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            pass

        img = cv2.imread(str(file_path))
        if img is not None and img.size > 0:
            return True
        return False
    except Exception as e:
        logger.debug(f"File validation failed for {file_path}: {e}")
        return False


def count_images_in_folder(folder_path: str, recursive: bool = True) -> int:
    """Count total readable image files in a folder"""
    try:
        return len(get_image_files(folder_path, recursive=recursive))
    except Exception as e:
        logger.error(f"Error counting images: {e}")
        return 0

# -------------------------------------------------------------------
# BEST PHOTO SAVING
# -------------------------------------------------------------------
def save_best_photos(best_photos: List[dict], output_dir: str):
    """Save selected best photo(s) and generate reports"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, photo in enumerate(best_photos):
        src_path = Path(photo['file_path'])
        dst_path = output_path / f"best_{i+1:02d}_{src_path.name}"

        # Copy the image
        import shutil
        shutil.copy2(src_path, dst_path)

        # Save analysis report for transparency
        report_path = output_path / f"best_{i+1:02d}_{src_path.stem}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Analysis Report for {src_path.name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Final Score: {photo['final_score']:.3f}\n")
            f.write(f"Face Detected: {photo['face_detected']}\n")

            if photo['face_detected']:
                f.write(f"Face Confidence: {photo.get('primary_face_confidence', 0):.3f}\n")
                f.write(f"Face Area Ratio: {photo.get('primary_face_area_ratio', 0):.3f}\n")

            f.write("\nQuality Metrics:\n")
            for key in [
                'sharpness', 'focus_quality', 'brightness', 'contrast',
                'white_balance', 'pose_quality', 'composition',
                'noise_level', 'artifacts'
            ]:
                if key in photo:
                    f.write(f" - {key.replace('_', ' ').title()}: {photo[key]:.3f}\n")

            if photo['rejection_reasons']:
                f.write(f"\nRejection Reasons: {', '.join(photo['rejection_reasons'])}\n")

            minor_issues = photo.get('minor_issues', [])
            if minor_issues:
                f.write(f"Minor Issues: {', '.join(minor_issues)}\n")

        logger.info(f"✅ Saved best photo: {dst_path.name}")
        logger.info(f"🧾 Report generated: {report_path.name}")
