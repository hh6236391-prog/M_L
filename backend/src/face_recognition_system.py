import cv2, os, json, time, logging, subprocess
import numpy as np
import face_recognition
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class FutureFaceCapture:
    def __init__(self,
                 images_dir="/Users/hariharans/Desktop/dataset",
                 max_images=10, tolerance=0.6,
                 camera_source="0", capture_interval=2.0,
                 blur_threshold=100.0, resolution=(1280, 720)):
        
        # Paths
        self.images_dir = Path(images_dir); self.trash_dir = self.images_dir / "trash"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir.mkdir(parents=True, exist_ok=True)

        # Camera setup
        src = 0 if camera_source.isdigit() else camera_source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened(): raise RuntimeError(f"❌ Cannot access camera: {camera_source}")
        w, h = resolution; self.cap.set(3, w); self.cap.set(4, h)

        # Config
        self.max_images, self.tolerance = max_images, tolerance
        self.capture_interval, self.blur_threshold = capture_interval, blur_threshold

        # Memory
        self.known_encodings, self.known_names = [], []
        self.image_counts, self.last_capture = {}, {}

        # Stats
        self.frame_count, self.fps, self.last_fps = 0, 0, time.time()
        print(f"💾 Saving full frames to: {self.images_dir}")

    # ---------------------- Helper Functions ----------------------
    def _is_good(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return 30 < np.mean(gray) < 220 and cv2.Laplacian(gray, cv2.CV_64F).var() >= self.blur_threshold

    def _should_save(self, name):
        now = time.time()
        return (self.image_counts.get(name, 0) < self.max_images and 
                now - self.last_capture.get(name, 0) >= self.capture_interval)

    def _find_match(self, enc):
        if not self.known_encodings: return None
        dists = face_recognition.face_distance(self.known_encodings, enc)
        return self.known_names[np.argmin(dists)] if np.min(dists) < self.tolerance else None

    def _save_frame(self, name, frame):
        folder = self.images_dir / name; folder.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = folder / f"{name}_{ts}.jpg"
        if self._is_good(frame):
            cv2.imwrite(str(path), frame)
            self.image_counts[name] = self.image_counts.get(name, 0) + 1
            self.last_capture[name] = time.time()
            logging.info(f"📸 Saved {path.name}")
        else:
            cv2.imwrite(str(self.trash_dir / f"BLUR_{ts}.jpg"), frame)

    def _draw(self, f, loc, name):
        t, r, b, l = loc
        color = (0, 255, 0) if self.image_counts.get(name, 0) < self.max_images else (0, 0, 255)
        label = f"{name} ({self.image_counts.get(name,0)}/{self.max_images})"
        cv2.rectangle(f, (l, t), (r, b), color, 2)
        cv2.putText(f, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _update_fps(self):
        self.frame_count += 1; now = time.time()
        if now - self.last_fps >= 1:
            self.fps = self.frame_count / (now - self.last_fps)
            self.frame_count, self.last_fps = 0, now

    # ---------------------- Main Loop ----------------------
    def capture_faces(self):
        print("🎥 FULL FRAME CAPTURE MODE — Press 'q' to quit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb)
                encs = face_recognition.face_encodings(rgb, faces)

                for enc, loc in zip(encs, faces):
                    name = self._find_match(enc) or f"person_{len(self.known_names)+1:03d}"
                    if name not in self.known_names:
                        self.known_names.append(name); self.known_encodings.append(enc)
                        self.image_counts[name] = 0; logging.info(f"🆕 New person: {name}")
                    if self._should_save(name): self._save_frame(name, frame)
                    self._draw(frame, loc, name)

                self._update_fps()
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 0, 0.7, (0, 255, 255), 2)
                cv2.imshow("FutureFaceCapture - Full Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally: self._cleanup()

    # ---------------------- Cleanup ----------------------
    def _cleanup(self):
        if self.cap.isOpened(): self.cap.release()
        cv2.destroyAllWindows()
        summary = {
            "persons": len(self.known_names),
            "total_images": sum(self.image_counts.values()),
            "details": self.image_counts,
            "timestamp": datetime.now().isoformat()
        }
        summary_path = self.images_dir / "capture_summary.json"
        with open(summary_path, "w") as f: json.dump(summary, f, indent=4)
        print(f"\n📊 Capture Complete → {summary_path}")
        subprocess.run(["open", str(self.images_dir)], check=False)


# ---------------------- Main ----------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = dict(images_dir="/Users/hariharans/Desktop/dataset", max_images=10,
               tolerance=0.6, camera_source="0", capture_interval=2.0,
               blur_threshold=100.0, resolution=(1280, 720))
    try:
        FutureFaceCapture(**cfg).capture_faces()
    except Exception as e:
        logging.error(f"❌ {e}")
        print("💡 Check camera connection or permissions.")


if __name__ == "__main__":
    main()
