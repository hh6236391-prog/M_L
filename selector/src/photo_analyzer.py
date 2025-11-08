import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from .quality_metrics import QualityMetrics
from .face_analyzer import FaceAnalyzer
from .utils import resize_image, validate_image

logger = logging.getLogger(__name__)

class PhotoAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.quality_metrics = QualityMetrics(config)
        self.face_analyzer = FaceAnalyzer(config)
        self.thresholds = config.get('thresholds', {})
        self.weights = config.get('quality_weights', {})

    # ------------------------------------------------------------------
    def analyze_image(self, image_path: Path) -> Dict:
        """Comprehensive analysis of a single image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image = validate_image(image)
            image = resize_image(image, self.config.get('computation', {}).get('max_image_size', 1024))

            analysis = {
                'file_path': str(image_path),
                'file_name': image_path.name,
                'image_size': image.shape[:2],
                'face_detected': False,
                'rejection_reasons': []
            }

            # Face detection
            faces = self.face_analyzer.detect_faces(image)
            face_analysis = self._analyze_faces(faces, image.shape)
            analysis.update(face_analysis)

            # Image quality metrics
            quality_scores = self._calculate_quality_metrics(image, face_analysis.get('primary_face_bbox'))
            analysis.update(quality_scores)

            # Apply lenient rejection checks
            self._apply_rejection_criteria(analysis)

            # Calculate final score
            analysis['final_score'] = self._calculate_final_score(analysis)

            # Add debug info
            analysis['debug_info'] = self._get_debug_info(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return {
                'file_path': str(image_path),
                'file_name': image_path.name,
                'error': str(e),
                'final_score': 0.0,
                'rejection_reasons': [f'Analysis error: {e}'],
                'face_detected': False
            }

    # ------------------------------------------------------------------
    def _analyze_faces(self, faces: List[Dict], image_shape: Tuple) -> Dict:
        """Analyze detected faces and select the primary one"""
        result = {
            'face_detected': False,
            'num_faces': len(faces),
            'primary_face_bbox': None,
            'face_analysis': {}
        }

        if not faces:
            return result

        valid_faces = []
        image_area = image_shape[0] * image_shape[1]

        for face in faces:
            face_area = face['area']
            area_ratio = face_area / image_area
            min_size = self.thresholds.get('min_face_size', 0.05)
            max_size = self.thresholds.get('max_face_size', 0.9)
            conf_thresh = self.config.get('face_detection', {}).get('confidence_threshold', 0.5)

            if min_size <= area_ratio <= max_size and face['confidence'] > conf_thresh:
                valid_faces.append(face)

        if not valid_faces:
            largest_face = max(faces, key=lambda x: x['area'])
            area_ratio = largest_face['area'] / image_area
            if area_ratio >= self.thresholds.get('min_face_size', 0.05):
                valid_faces.append(largest_face)

        if not valid_faces:
            return result

        primary_face = max(valid_faces, key=lambda x: x['area'])
        result.update({
            'face_detected': True,
            'primary_face_bbox': primary_face['bbox'],
            'primary_face_confidence': primary_face['confidence'],
            'primary_face_area_ratio': primary_face['area'] / image_area
        })
        return result

    # ------------------------------------------------------------------
    def _calculate_quality_metrics(self, image: np.ndarray, face_bbox: Optional[Tuple]) -> Dict:
        """Compute all quality metrics"""
        metrics = {
            'sharpness': self.quality_metrics.calculate_sharpness(image),
            'brightness': None,
            'contrast': self.quality_metrics.calculate_contrast(image),
            'white_balance': self.quality_metrics.assess_white_balance(image),
            'noise_level': self.quality_metrics.detect_noise(image),
            'artifacts': self.quality_metrics.detect_artifacts(image),
        }

        metrics['brightness'], metrics['exposure_status'] = self.quality_metrics.calculate_brightness(image)

        if face_bbox:
            pose = self.face_analyzer.analyze_pose(image, face_bbox)
            metrics.update(pose)
            metrics['composition'] = self.quality_metrics.assess_composition(image, face_bbox)
            metrics['focus_quality'] = self._assess_focus_quality(image, face_bbox)
        else:
            metrics.update({
                'pose_quality': 0.4,
                'frontal_quality': 0.4,
                'composition': 0.5,
                'focus_quality': 0.5,
                'eyes_visible': False
            })

        return metrics

    # ------------------------------------------------------------------
    def _assess_focus_quality(self, image: np.ndarray, face_bbox: Tuple) -> float:
        """Assess subject focus"""
        try:
            x, y, w, h = face_bbox
            face_roi = image[y:y+h, x:x+w]
            if face_roi.size == 0:
                return 0.5

            face_sharpness = self.quality_metrics.calculate_sharpness(face_roi)
            max_expected_sharpness = 1000
            focus_quality = min(face_sharpness / max_expected_sharpness, 1.0)
            return max(focus_quality, 0.1)
        except Exception as e:
            logger.warning(f"Focus quality assessment failed: {e}")
            return 0.5

    # ------------------------------------------------------------------
    def _apply_rejection_criteria(self, analysis: Dict):
        """Reject only critical issues"""
        reasons = analysis['rejection_reasons']
        critical = []

        if analysis['sharpness'] < self.thresholds.get('min_sharpness', 20.0):
            critical.append('Image too blurry')

        brightness = analysis['brightness']
        if brightness < 10 or brightness > 250:
            critical.append(f'Extreme exposure: {analysis["exposure_status"]}')

        if analysis['face_detected'] and analysis.get('pose_quality', 0) < 0.2:
            critical.append('Poor face pose')

        analysis['rejection_reasons'].extend(critical)

        minor = []
        if analysis['contrast'] < self.thresholds.get('min_contrast', 15.0):
            minor.append('Low contrast')
        if analysis['noise_level'] < 0.4:
            minor.append('Some noise')

        analysis['minor_issues'] = minor

    # ------------------------------------------------------------------
    def _calculate_final_score(self, analysis: Dict) -> float:
        """Compute weighted final score — optimized for best photo selection"""
        total_score = 0.0
        total_weight = 0.0

        base_score = 0.2
        total_score += base_score
        total_weight += 0.1

        # Face weighting
        if analysis['face_detected']:
            conf = analysis.get('primary_face_confidence', 0.0)
            area_ratio = analysis.get('primary_face_area_ratio', 0.0)
            if area_ratio < 0.15:
                area_score = area_ratio / 0.15
            elif area_ratio <= 0.5:
                area_score = 1.0
            else:
                area_score = max(0.3, 1.0 - (area_ratio - 0.5))

            face_score = (conf * 0.7 + area_score * 0.3)
            face_weight = self.weights.get('face_detection', 0.15)
            total_score += face_score * face_weight
            total_weight += face_weight
        else:
            total_score += 0.3 * self.weights.get('face_detection', 0.15)
            total_weight += self.weights.get('face_detection', 0.15)

        # Normalize major metrics
        norm = {
            'sharpness': self._normalize_sharpness_balanced(analysis.get('sharpness', 0)),
            'brightness': self._normalize_brightness_balanced(analysis.get('brightness', 0)),
            'contrast': self._normalize_contrast_balanced(analysis.get('contrast', 0)),
            'white_balance': analysis.get('white_balance', 0.5),
            'pose_quality': analysis.get('pose_quality', 0.5),
            'composition': analysis.get('composition', 0.5),
            'focus_quality': analysis.get('focus_quality', 0.5),
            'noise_level': analysis.get('noise_level', 0.5),
            'artifacts': analysis.get('artifacts', 0.5)
        }

        # Weighted total
        for m, s in norm.items():
            w = self.weights.get(m, 0.1)
            total_score += s * w
            total_weight += w

        # Compute final score
        final_score = total_score / total_weight if total_weight > 0 else 0.0

        # Penalty adjustment (lighter penalties)
        final_score -= len(analysis['rejection_reasons']) * 0.04
        final_score -= len(analysis.get('minor_issues', [])) * 0.01

        # Frontal face bonus
        if analysis.get('face_detected', False):
            if analysis.get('pose_quality', 0) > 0.8 and analysis.get('frontal_quality', 0) > 0.8:
                final_score += 0.05

        final_score = np.clip(final_score, 0.0, 1.0)

        # Debug print
        print(f"[DEBUG] {analysis['file_name']}: Sharp={analysis['sharpness']:.1f}, "
              f"Bright={analysis['brightness']:.1f}, Face={analysis.get('primary_face_confidence', 0):.2f}, "
              f"Final={final_score:.3f}")
        return final_score

    # ------------------------------------------------------------------
    def _normalize_sharpness_balanced(self, sharpness: float) -> float:
        """Improved normalization — gives more reward for sharpness"""
        if sharpness < 30:
            return 0.1
        elif sharpness < 100:
            return 0.4 + (sharpness - 30) / 70.0 * 0.3
        elif sharpness < 500:
            return 0.7 + (sharpness - 100) / 400.0 * 0.25
        else:
            return 1.0

    def _normalize_brightness_balanced(self, brightness: float) -> float:
        """Brightness normalization (ideal ~127)"""
        if 50 <= brightness <= 200:
            deviation = abs(brightness - 127) / 73.0
            return 1.0 - (deviation * 0.6)
        elif 25 <= brightness < 50 or 200 < brightness <= 225:
            return 0.4
        else:
            return 0.2

    def _normalize_contrast_balanced(self, contrast: float) -> float:
        """Contrast normalization"""
        if contrast < 15:
            return 0.2
        elif contrast < 30:
            return 0.2 + (contrast - 15) / 15.0 * 0.4
        elif contrast <= 70:
            return 0.6 + (contrast - 30) / 40.0 * 0.4
        else:
            return 1.0

    # ------------------------------------------------------------------
    def _get_debug_info(self, analysis: Dict) -> Dict:
        return {
            'face_detected': analysis.get('face_detected', False),
            'sharpness_raw': analysis.get('sharpness', 0),
            'brightness_raw': analysis.get('brightness', 0),
            'contrast_raw': analysis.get('contrast', 0),
            'rejection_count': len(analysis.get('rejection_reasons', [])),
            'minor_issues_count': len(analysis.get('minor_issues', [])),
            'face_confidence': analysis.get('primary_face_confidence', 0),
            'face_area_ratio': analysis.get('primary_face_area_ratio', 0)
        }

    # ------------------------------------------------------------------
    def _apply_score_differentiation(self, analyses: List[Dict]) -> List[Dict]:
        """Add slight variation when scores are close"""
        if len(analyses) <= 1:
            return analyses

        scores = [a.get('final_score', 0) for a in analyses]
        if max(scores) - min(scores) < 0.1:
            for a in analyses:
                bonus = 0.0
                if a.get('sharpness', 0) > 300:
                    bonus += 0.05
                if a.get('face_detected', False) and a.get('primary_face_confidence', 0) > 0.8:
                    bonus += 0.05
                if 100 <= a.get('brightness', 0) <= 150:
                    bonus += 0.03
                a['final_score'] = np.clip(a['final_score'] + bonus, 0, 1)

        return analyses

    # ------------------------------------------------------------------
    def select_single_best_photo(self, analyses: List[Dict]) -> Dict:
        """Select the single best photo"""
        if not analyses:
            return None
        candidates = [a for a in analyses if a.get('final_score', 0) > 0.5]
        if not candidates:
            return max(analyses, key=lambda x: x.get('final_score', 0))

        best = max(candidates, key=lambda x: (
            x.get('final_score', 0) +
            x.get('sharpness', 0) / 500.0 * 0.3 +
            x.get('pose_quality', 0) * 0.2 +
            x.get('composition', 0) * 0.1
        ))
        return best
