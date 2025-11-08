import cv2  # OpenCV library for computer vision
import numpy as np  # For numerical operations and array handling
from typing import List, Dict, Tuple, Optional  # Type hints
import logging  # For tracking what the code is doing

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        # Load OpenCV face detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

    # -------------------------------------------------------------------
    # FACE DETECTION
    # -------------------------------------------------------------------
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar cascades."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect frontal and profile faces
            frontal_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            profile_faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            faces = []
            # Frontal faces
            for (x, y, w, h) in frontal_faces:
                confidence = self._calculate_face_confidence(image, (x, y, w, h))
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'area': w * h,
                    'type': 'frontal'
                })
            # Profile faces
            for (x, y, w, h) in profile_faces:
                confidence = self._calculate_face_confidence(image, (x, y, w, h)) * 0.7
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'area': w * h,
                    'type': 'profile'
                })
            return faces

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []

    # -------------------------------------------------------------------
    # FACE CONFIDENCE
    # -------------------------------------------------------------------
    def _calculate_face_confidence(self, image: np.ndarray, bbox: Tuple) -> float:
        """Calculate confidence score for detected face."""
        try:
            x, y, w, h = bbox
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]

            if face_roi.size == 0:
                return 0.0

            confidence = 0.5  # base
            aspect_ratio = w / h
            if 0.6 <= aspect_ratio <= 1.4:
                confidence += 0.2

            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            if len(eyes) >= 2:
                confidence += 0.3

            return min(confidence, 1.0)

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    # -------------------------------------------------------------------
    # POSE + EYE OPENNESS ANALYSIS
    # -------------------------------------------------------------------
    def analyze_pose(self, image: np.ndarray, face_bbox: Tuple) -> Dict:
        """Analyze face pose, frontal quality, and eye openness."""
        try:
            x, y, w, h = face_bbox
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]

            if face_roi.size == 0:
                return {
                    'pose_quality': 0.0,
                    'eyes_visible': False,
                    'frontal_quality': 0.0,
                    'eyes_open_score': 0.0
                }

            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            eyes_visible = len(eyes) >= 2
            pose_quality = 0.5
            frontal_quality = 0.5
            eyes_open_score = 0.5  # neutral default

            if eyes_visible:
                pose_quality = 0.8
                frontal_quality = 0.8

                # Sort two largest eyes
                eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                eye_scores = []

                for (ex, ey, ew, eh) in eyes:
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    if eye_roi.size == 0:
                        continue
                    blur = cv2.Laplacian(eye_roi, cv2.CV_64F).var()
                    # Closed eyes have lower Laplacian variance
                    eye_scores.append(1.0 if blur > 25 else 0.3)

                if eye_scores:
                    eyes_open_score = np.mean(eye_scores)

                # Penalize closed eyes
                if eyes_open_score < 0.5:
                    pose_quality *= 0.6
                    frontal_quality *= 0.7

                # Horizontal alignment (frontal face)
                if len(eyes) == 2:
                    eye_y_diff = abs(eyes[0][1] - eyes[1][1])
                    eye_x_diff = abs((eyes[0][0] + eyes[1][0]) - w / 2)
                    if eye_y_diff < h * 0.1 and eye_x_diff < w * 0.4:
                        frontal_quality += 0.1

            return {
                'pose_quality': min(pose_quality, 1.0),
                'eyes_visible': eyes_visible,
                'frontal_quality': min(frontal_quality, 1.0),
                'eyes_open_score': eyes_open_score
            }

        except Exception as e:
            logger.warning(f"Pose analysis failed: {e}")
            return {
                'pose_quality': 0.0,
                'eyes_visible': False,
                'frontal_quality': 0.0,
                'eyes_open_score': 0.0
            }
