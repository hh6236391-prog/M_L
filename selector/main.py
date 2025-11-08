import argparse
import logging
from pathlib import Path
import sys
import os

# -------------------------------------------------------------------
# Add src folder to Python path (so imports work both in backend and CLI)
# -------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.photo_analyzer import PhotoAnalyzer
    from src.utils import load_config, get_image_files, save_best_photos, get_default_config
    # ✅ Import global selector
    from src.global_best_selector import rank_and_save_global
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Make sure all required dependencies are installed.")
    sys.exit(1)

# -------------------------------------------------------------------
# Setup logging configuration
# -------------------------------------------------------------------
def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("photo_analysis.log", mode="a"),
            logging.StreamHandler()
        ]
    )

# -------------------------------------------------------------------
# Check dependencies
# -------------------------------------------------------------------
def check_dependencies():
    """Verify required dependencies."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import skimage
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install all dependencies using: pip install -r requirements.txt")
        return False

# -------------------------------------------------------------------
# Main image analysis and best-photo selection
# -------------------------------------------------------------------
def main():
    if not check_dependencies():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Select best photos from captured images")

    # ✅ Added backend integration args
    parser.add_argument("--input_dir", help="Directory containing captured images (for backend integration)", default=None)
    parser.add_argument("--output_dir", help="Directory to save best images (for backend integration)", default=None)

    # ✅ Standard CLI arguments for standalone use
    parser.add_argument("input_folder", nargs="?", help="Folder containing photos to analyze")
    parser.add_argument("--output", "-o", default="results/best_photos", help="Output folder for best photos")
    parser.add_argument("--config", "-c", default="config/default_config.yaml", help="Path to configuration file")
    parser.add_argument("--num-best", "-n", type=int, default=1, help="Number of best photos to select")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum score to consider a photo valid")
    parser.add_argument("--no-face-detection", action="store_true", help="Disable face detection")
    args = parser.parse_args()

    # ✅ Compatibility layer for backend subprocess call
    input_folder = args.input_dir or args.input_folder
    output_folder = args.output_dir or args.output

    if not input_folder:
        print("❌ Error: No input folder specified.")
        sys.exit(1)

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Best Image Selection Process...")

    # -------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"✅ Loaded config from {args.config}")
    else:
        config = get_default_config()
        logger.warning(f"⚠️ Config file {args.config} not found — using default settings")

    logger.info(f"📁 Input folder: {input_folder}")
    logger.info(f"💾 Output folder: {output_folder}")

    # -------------------------------------------------------------------
    # Initialize analyzer and gather image files
    # -------------------------------------------------------------------
    analyzer = PhotoAnalyzer(config)
    image_files = get_image_files(input_folder)

    if not image_files:
        print("❌ No images found! Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP")
        sys.exit(1)

    logger.info(f"🔍 Found {len(image_files)} images for analysis")

    # -------------------------------------------------------------------
    # Analyze images and calculate scores
    # -------------------------------------------------------------------
    results = []
    for i, image_path in enumerate(image_files):
        logger.info(f"🧠 Analyzing {i+1}/{len(image_files)}: {image_path.name}")
        try:
            analysis = analyzer.analyze_image(image_path)
            results.append(analysis)
        except Exception as e:
            logger.error(f"❌ Failed to analyze {image_path}: {e}")
            continue

    # -------------------------------------------------------------------
    # Post-process scores and filter best photos
    # -------------------------------------------------------------------
    if hasattr(analyzer, "_apply_score_differentiation"):
        results = analyzer._apply_score_differentiation(results)

    valid_results = [r for r in results if r.get("final_score", 0) >= args.min_score]
    valid_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    best_photos = valid_results[:args.num_best]
    os.makedirs(output_folder, exist_ok=True)
    save_best_photos(best_photos, output_folder)

    # -------------------------------------------------------------------
    # Generate summary
    # -------------------------------------------------------------------
    print("\n" + "="*60)
    print("🏆 BEST PHOTOS SUMMARY")
    print("="*60)
    for i, photo in enumerate(best_photos):
        score = photo.get("final_score", 0)
        faces = photo.get("num_faces", 0)
        print(f"{i+1:2d}. {photo['file_name']:30} | Score: {score:.3f} | Faces: {faces}")

    print(f"\n🎯 Best photo(s) saved to: {output_folder}")
    logger.info(f"✅ Best photo(s) saved to {output_folder}")
    logger.info("🏁 Best Image Selection Completed Successfully\n")

    # -------------------------------------------------------------------
    # 🌍 NEW: Run Global Best Selector (for all persons)
    # -------------------------------------------------------------------
    try:
        logger.info("🌍 Running Global Best Image Selector...")

        # These paths are relative to backend structure
        project_root = Path(__file__).resolve().parent.parent
        best_images_root = project_root / "backend" / "images" / "best_images"
        global_results = project_root / "backend" / "images" / "top_results"

        global_results.mkdir(parents=True, exist_ok=True)
        rank_and_save_global(str(best_images_root), str(global_results), top_k=5)

        logger.info("🏁 Global best image ranking complete.")
    except Exception as e:
        logger.error(f"❌ Global best selector failed: {e}")


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
