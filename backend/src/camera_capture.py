
import cv2
import os
import json
import time
import logging
import numpy as np
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from insightface.app import FaceAnalysis


class OrganizedFaceCapture:
    def __init__(
        self,
        base_dir: str = "images",
        max_images_per_person: int = 10,
        face_detection_threshold: float = 0.45,
        stability_frames: int = 3,
        label_persistence_time: float = 3.0
    ):
        """Initialize directories, configurations, and model."""
        self.base_dir = Path(base_dir)
        self.people_dir = self.base_dir / "people"
        self.people_dir.mkdir(parents=True, exist_ok=True)

        # Settings
        self.max_images = max_images_per_person
        self.face_threshold = face_detection_threshold
        self.stability_frames = stability_frames
        self.label_persistence_time = label_persistence_time

        # Tracking
        self.known_faces: Dict[str, np.ndarray] = {}
        self.face_folders: Dict[str, Path] = {}
        self.image_counts: Dict[str, int] = {}
        self.person_counter = 0
        self._recent_unmatched: List[np.ndarray] = []
        self._last_seen: Dict[str, float] = {}
        self._last_new_person_time: float = 0.0

        # Setup
        self._setup_logging()
        self._setup_face_model()

        logging.info("✅ System ready — full-frame image capture mode")

    # -------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------
    def _setup_logging(self):
        """Configure logging output."""
        log_file = self.base_dir / "face_capture.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='a')
            ]
        )

    def _setup_face_model(self):
        """Load the InsightFace model."""
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logging.info("✅ InsightFace model loaded successfully")

    # -------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------
    def _next_person_id(self) -> str:
        """Generate next unique person ID."""
        self.person_counter += 1
        return f"person_{self.person_counter:03d}"

    def _create_folder(self, person_id: str) -> Path:
        """Create folder for each person."""
        folder = self.people_dir / person_id
        folder.mkdir(parents=True, exist_ok=True)
        self.face_folders[person_id] = folder
        self.image_counts[person_id] = 0
        return folder

    def _find_person(self, embedding: np.ndarray) -> Optional[str]:
        """Find or register person with stability and cooldown logic."""
        best_match = None
        best_similarity = 0.0

        for pid, emb in self.known_faces.items():
            similarity = np.dot(embedding, emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(emb)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pid

        # ✅ Match existing person
        if best_similarity > self.face_threshold:
            # smoother averaging for stability
            self.known_faces[best_match] = (
                0.85 * self.known_faces[best_match] + 0.15 * embedding
            )
            # normalize
            self.known_faces[best_match] /= np.linalg.norm(self.known_faces[best_match])
            self._last_seen[best_match] = time.time()
            return best_match

        # ✅ Cooldown – avoid new ID spam
        if time.time() - self._last_new_person_time < 5:
            if self.known_faces:
                recent_pid = list(self.known_faces.keys())[-1]
                self._last_seen[recent_pid] = time.time()
                return recent_pid

        # ✅ New face stability
        self._recent_unmatched.append(embedding)
        if len(self._recent_unmatched) > self.stability_frames:
            self._recent_unmatched.pop(0)

        if len(self._recent_unmatched) == self.stability_frames:
            avg_new = np.mean(self._recent_unmatched, axis=0)
            self._recent_unmatched.clear()
            new_pid = self._next_person_id()
            self.known_faces[new_pid] = avg_new / np.linalg.norm(avg_new)
            self._create_folder(new_pid)
            self._last_seen[new_pid] = time.time()
            self._last_new_person_time = time.time()
            logging.info(f"🆕 New person confirmed → {new_pid}")
            return new_pid

        return None

    # -------------------------------------------------------------------
    # CAPTURE PHASE
    # -------------------------------------------------------------------
    def capture_faces(self, camera_index: int = 0, duration: int = 300):
        """Capture and store full-frame images."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logging.error("❌ Cannot access camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        logging.info("🎥 Starting capture... Press 'q' to quit")

        start_time = time.time()
        last_capture_time = 0
        capture_interval = 2.0

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.app.get(frame)
            current_time = time.time()

            for face in faces:
                if face.embedding is None:
                    continue

                pid = self._find_person(face.embedding)
                if not pid:
                    continue

                if (self.image_counts[pid] < self.max_images and
                        current_time - last_capture_time >= capture_interval):
                    self._save_full_frame(pid, frame, face.bbox)
                    last_capture_time = current_time

            self._display_frame(frame, faces)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary()

    # -------------------------------------------------------------------
    # SAVE IMAGE + METADATA
    # -------------------------------------------------------------------
    def _save_full_frame(self, pid: str, frame: np.ndarray, bbox: Tuple):
        """Save image and metadata."""
        folder = self.face_folders[pid]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = folder / f"{pid}_{timestamp}.jpg"
        cv2.imwrite(str(filepath), frame)

        self.image_counts[pid] += 1
        self._save_metadata(pid, filepath, bbox)
        logging.info(f"📸 Saved image for {pid} ({self.image_counts[pid]}/{self.max_images})")

    def _save_metadata(self, pid: str, image_path: Path, bbox: Tuple):
        """Save bounding box data."""
        meta_file = self.face_folders[pid] / "face_locations.json"
        data = {}
        if meta_file.exists():
            with open(meta_file, "r") as f:
                data = json.load(f)
        data[image_path.name] = {"bbox": list(map(int, bbox)), "time": datetime.now().isoformat()}
        with open(meta_file, "w") as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------------
    # DISPLAY
    # -------------------------------------------------------------------
    def _display_frame(self, frame: np.ndarray, faces: List):
        """Show live annotated frame."""
        current_time = time.time()
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            pid = self._find_person(face.embedding)
            if pid:
                self._last_seen[pid] = current_time
            else:
                pid = self._get_recent_person(current_time)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, pid or "Unknown", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        stats = f"Persons: {len(self.known_faces)} | Images: {sum(self.image_counts.values())}"
        cv2.putText(frame, stats, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Full Frame Capture", frame)

    def _get_recent_person(self, current_time: float) -> Optional[str]:
        """Return recently seen person."""
        for pid, last_seen_time in self._last_seen.items():
            if current_time - last_seen_time <= self.label_persistence_time:
                return pid
        return None

    # -------------------------------------------------------------------
    # SUMMARY + SELECTOR
    # -------------------------------------------------------------------
    def _print_summary(self):
        """Print summary and run selector."""
        logging.info("📊 Capture Summary:")
        for pid, count in self.image_counts.items():
            logging.info(f"   {pid}: {count} images")

        self._run_best_image_selector()
        logging.info("✅ Capture complete.")

    def _run_best_image_selector(self):
        """Run best image selector after capture.

        NOTE: selector/main.py expects the INPUT folder as a positional argument
              (not --input_folder). We pass input_dir positionally to avoid
              the 'unrecognized arguments' error.
        """
        selector_script = Path(__file__).resolve().parents[2] / "selector" / "main.py"
        input_dir = str(self.people_dir.resolve())
        output_dir = str((self.base_dir / "best_images").resolve())
        os.makedirs(output_dir, exist_ok=True)

        logging.info("🔍 Running Best Image Selector...")
        logging.info(f"➡️ Selector path: {selector_script}")
        logging.info(f"➡️ Input dir: {input_dir}")
        logging.info(f"➡️ Output dir: {output_dir}")

        if not selector_script.exists():
            logging.error(f"❌ Selector script not found: {selector_script}")
            return

        try:
            # Pass input_dir as positional (first) argument; selector expects this
            subprocess.run(
                [
                    sys.executable,
                    str(selector_script),
                    input_dir,
                    "--output", output_dir,
                    "--num-best", "1"
                ],
                check=True
            )
            logging.info(f"🎯 Best image selection complete → {output_dir}")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Selector failed: {e}")
        except Exception as e:
            logging.error(f"⚠️ Error while running selector: {e}")


# -------------------------------------------------------------------
def main():
    """Main entry point."""
    config = {
        "base_dir": "images",
        "max_images_per_person": 10,
        "face_detection_threshold": 0.45
    }

    try:
        system = OrganizedFaceCapture(**config)
        system.capture_faces(duration=300)
    except Exception as e:
        logging.error(f"❌ Capture failed: {e}")
        logging.info("💡 Check your camera or model setup.")


if __name__ == "__main__":
    main()
