"""
OrganizedFaceCapture (Integrated Multi-Person System)
----------------------------------------------------
Captures full-frame images for multiple persons using InsightFace,
organizes them into folders, and automatically triggers:
  1. Best image selector for each person
  2. Global best selector for all persons
"""

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
        face_detection_threshold: float = 0.6
    ):
        """Initialize directories, configurations, and model."""
        self.base_dir = Path(base_dir)
        self.people_dir = self.base_dir / "people"
        self.people_dir.mkdir(parents=True, exist_ok=True)

        self.max_images = max_images_per_person
        self.face_threshold = face_detection_threshold

        self.known_faces: Dict[str, np.ndarray] = {}
        self.face_folders: Dict[str, Path] = {}
        self.image_counts: Dict[str, int] = {}
        self.person_counter = 0
        self.last_embedding: Optional[np.ndarray] = None

        self._setup_logging()
        self._setup_face_model()

        logging.info("✅ System ready — Multi-Person Capture Mode Activated")

    # -------------------------------------------------------------------
    # Setup
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
    # Helper Functions
    # -------------------------------------------------------------------
    def _next_person_id(self) -> str:
        self.person_counter += 1
        return f"person_{self.person_counter:03d}"

    def _create_folder(self, person_id: str) -> Path:
        """Create folder for a person."""
        folder = self.people_dir / person_id
        folder.mkdir(parents=True, exist_ok=True)
        self.face_folders[person_id] = folder
        self.image_counts[person_id] = 0
        return folder

    def _find_person(self, embedding: np.ndarray) -> Optional[str]:
        """Compare face embedding with known faces."""
        for pid, emb in self.known_faces.items():
            similarity = np.dot(embedding, emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(emb)
            )
            if similarity > self.face_threshold:
                return pid
        return None

    # -------------------------------------------------------------------
    # Capture
    # -------------------------------------------------------------------
    def capture_faces(self, camera_index: int = 0, duration: int = 300):
        """Capture full-frame images for each person."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logging.error("❌ Cannot access camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        logging.info("🎥 Starting capture... Press 'q' to stop.")
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
                    pid = self._next_person_id()
                    self.known_faces[pid] = face.embedding
                    self._create_folder(pid)
                    logging.info(f"🆕 New person detected: {pid}")

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
    # Save & Metadata
    # -------------------------------------------------------------------
    def _save_full_frame(self, pid: str, frame: np.ndarray, bbox: Tuple):
        folder = self.face_folders[pid]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = folder / f"{pid}_{timestamp}.jpg"
        cv2.imwrite(str(filepath), frame)

        self.image_counts[pid] += 1
        self._save_metadata(pid, filepath, bbox)

        logging.info(f"📸 {pid} → saved {self.image_counts[pid]}/{self.max_images}")

    def _save_metadata(self, pid: str, image_path: Path, bbox: Tuple):
        meta_file = self.face_folders[pid] / "face_locations.json"
        data = {}
        if meta_file.exists():
            with open(meta_file, "r") as f:
                data = json.load(f)
        data[image_path.name] = {"bbox": list(map(int, bbox)), "time": datetime.now().isoformat()}
        with open(meta_file, "w") as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------
    def _display_frame(self, frame: np.ndarray, faces: List):
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            pid = self._find_person(face.embedding) or "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, pid, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        stats = f"Persons: {len(self.known_faces)} | Images: {sum(self.image_counts.values())}"
        cv2.putText(frame, stats, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Full Frame Capture", frame)

    # -------------------------------------------------------------------
    # Summary + Automatic Best Selector
    # -------------------------------------------------------------------
    def _print_summary(self):
        logging.info("📊 Capture Summary:")
        for pid, count in self.image_counts.items():
            logging.info(f"   {pid}: {count} images")

        # Trigger automatic selectors
        self._run_best_image_selector()
        self._run_global_best_selector()

        logging.info("✅ Capture and Selection Complete.")

    # -------------------------------------------------------------------
    # Run Selectors Automatically
    # -------------------------------------------------------------------
    def _run_best_image_selector(self):
        """Run per-person best selector."""
        selector_script = Path(__file__).resolve().parent.parent.parent / "selector" / "main.py"
        input_dir = str(self.people_dir.resolve())
        output_dir = str((self.base_dir / "best_images").resolve())

        os.makedirs(output_dir, exist_ok=True)
        logging.info("🔍 Running Per-Person Best Image Selector...")

        try:
            subprocess.run(
                [
                    sys.executable, str(selector_script),
                    "--input_dir", input_dir,
                    "--output_dir", output_dir,
                    "--num-best", "3"
                ],
                check=True
            )
            logging.info(f"✅ Per-person best selection complete → {output_dir}")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Selector failed: {e}")

    def _run_global_best_selector(self):
        """Run global best selector automatically."""
        global_script = Path(__file__).resolve().parent.parent.parent / "selector" / "src" / "global_best_selector.py"
        best_images_root = str((self.base_dir / "best_images").resolve())
        output_dir = str((self.base_dir / "top_results").resolve())

        os.makedirs(output_dir, exist_ok=True)
        logging.info("🌍 Running Global Best Selector...")

        try:
            subprocess.run(
                [
                    sys.executable, str(global_script),
                    "--best_images_root", best_images_root,
                    "--output_dir", output_dir,
                    "--top_k", "5"
                ],
                check=True
            )
            logging.info(f"🏁 Global ranking complete → {output_dir}")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Global selector failed: {e}")


# -------------------------------------------------------------------
def main():
    config = {
        "base_dir": "images",
        "max_images_per_person": 10,
        "face_detection_threshold": 0.6
    }

    try:
        system = OrganizedFaceCapture(**config)
        system.capture_faces(duration=60)  # shorter duration for testing
    except Exception as e:
        logging.error(f"❌ Capture failed: {e}")


if __name__ == "__main__":
    main()
