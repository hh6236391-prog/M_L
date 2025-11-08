"""
global_best_selector.py
-----------------------
Selects the best image for each person and then ranks them globally.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from shutil import copy2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def calculate_image_score(image_path: Path) -> float:
    """Compute image quality score based on sharpness + brightness."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sharpness: variance of Laplacian
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness: average pixel intensity
        brightness = np.mean(gray)

        # Normalize brightness (ideal range around 128)
        normalized_brightness = 1 - abs(brightness - 128) / 128

        # Weighted overall score
        score = (sharpness / 5000) * 0.7 + normalized_brightness * 0.3
        return round(max(0.0, min(score, 1.0)), 3)
    except Exception as e:
        logging.warning(f"⚠️ Failed to score {image_path}: {e}")
        return 0.0


def select_best_per_person(best_images_root: Path, best_per_person_dir: Path) -> list:
    """Select best image for each person folder."""
    best_per_person_dir.mkdir(parents=True, exist_ok=True)
    person_best_list = []

    for person_dir in best_images_root.glob("*"):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        best_score = 0.0
        best_image = None

        for img_file in person_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                continue

            score = calculate_image_score(img_file)
            if score > best_score:
                best_score = score
                best_image = img_file

        if best_image:
            dst_folder = best_per_person_dir / person_name
            dst_folder.mkdir(parents=True, exist_ok=True)
            dst_path = dst_folder / best_image.name
            copy2(best_image, dst_path)
            logging.info(f"📸 {person_name}: best = {best_image.name} (score={best_score:.3f})")
            person_best_list.append({
                "person": person_name,
                "file_path": str(best_image),
                "score": best_score
            })

    return person_best_list


def rank_and_save_global(best_images_root: str, output_dir: str, top_k: int = 5):
    """
    Selects best per person first, then ranks them globally.
    """
    best_images_root = Path(best_images_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: best per person
    logging.info("🧠 Selecting best image per person...")
    best_per_person_dir = best_images_root.parent / "best_images_final"
    best_per_person = select_best_per_person(best_images_root, best_per_person_dir)

    if not best_per_person:
        logging.error("❌ No valid images found for any person.")
        return

    # Step 2: global ranking
    logging.info("🌍 Ranking best images globally...")
    best_per_person.sort(key=lambda x: x["score"], reverse=True)

    # Save top K globally
    top_k = min(top_k, len(best_per_person))
    global_top = best_per_person[:top_k]

    for i, entry in enumerate(global_top, 1):
        src = Path(entry["file_path"])
        dst = output_dir / f"global_top_{i}_{entry['person']}_{src.name}"
        copy2(src, dst)
        logging.info(f"💾 Saved {dst.name} (score={entry['score']:.3f})")

    # Save reports
    with open(output_dir / "global_ranking.json", "w") as f:
        json.dump(best_per_person, f, indent=2)

    with open(output_dir / "global_ranking.txt", "w") as f:
        for i, entry in enumerate(global_top, 1):
            f.write(f"{i}. {entry['person']} — {Path(entry['file_path']).name} | Score={entry['score']:.3f}\n")

    # Step 3: save global top 1 to final_best
    final_best_dir = best_images_root.parent / "final_best"
    final_best_dir.mkdir(parents=True, exist_ok=True)
    best_global = global_top[0]
    top1_path = final_best_dir / "global_best_overall.jpg"
    copy2(best_global["file_path"], top1_path)
    logging.info(f"🏆 Saved global_best_overall.jpg (person={best_global['person']}, score={best_global['score']:.3f})")

    logging.info(f"✅ Completed global ranking. Top {top_k} results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rank global best images (multi-person)")
    parser.add_argument("--best_images_root", required=True, help="Path to 'best_images' folder")
    parser.add_argument("--output_dir", required=True, help="Path to save top results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top images to select globally")
    args = parser.parse_args()

    rank_and_save_global(args.best_images_root, args.output_dir, args.top_k)
