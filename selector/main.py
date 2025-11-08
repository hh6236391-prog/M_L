import argparse
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.photo_analyzer import PhotoAnalyzer
    from src.utils import load_config, get_image_files, save_best_photos, get_default_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('photo_analysis.log'),
            logging.StreamHandler()
        ]
    )


def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import skimage
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        return False


def main():
    if not check_dependencies():
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Select best photos from captured images')

    # ✅ Added backend integration arguments
    parser.add_argument('--input_dir', help='Directory containing captured images', default=None)
    parser.add_argument('--output_dir', help='Directory to save best images', default=None)

    # Keep original CLI arguments for standalone use
    parser.add_argument('input_folder', nargs='?', help='Folder containing photos')
    parser.add_argument('--output', '-o', default='results/best_photos',
                       help='Output folder for best photos')
    parser.add_argument('--config', '-c', default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num-best', '-n', type=int, default=1,
                       help='Number of best photos to select')
    parser.add_argument('--min-score', type=float, default=0.3,
                       help='Minimum score to consider photo valid')
    parser.add_argument('--no-face-detection', action='store_true',
                       help='Disable face detection (use general image quality only)')
    args = parser.parse_args()

    # ✅ Compatibility handling
    input_folder = args.input_dir or args.input_folder
    output_folder = args.output_dir or args.output

    if not input_folder:
        print("❌ Error: No input folder provided.")
        sys.exit(1)

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = get_default_config()
        logger.warning(f"Config file {args.config} not found — using defaults")

    logger.info(f"📁 Input directory: {input_folder}")
    logger.info(f"💾 Output directory: {output_folder}")

    # Initialize analyzer
    analyzer = PhotoAnalyzer(config)
    logger.info("Photo analyzer initialized")

    # Get all image files
    image_files = get_image_files(input_folder)
    if not image_files:
        print("❌ No images found! Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP")
        sys.exit(1)

    logger.info(f"Found {len(image_files)} images for analysis.")

    # Analyze images
    results = []
    for i, image_path in enumerate(image_files):
        logger.info(f"Analyzing {i+1}/{len(image_files)}: {image_path.name}")
        try:
            analysis = analyzer.analyze_image(image_path)
            results.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze {image_path}: {e}")
            continue

    # Apply differentiation and sort
    results = analyzer._apply_score_differentiation(results)
    valid_results = [r for r in results if r.get('final_score', 0) >= args.min_score]
    valid_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)

    # Select best photos
    best_photos = valid_results[:args.num_best]
    os.makedirs(output_folder, exist_ok=True)
    save_best_photos(best_photos, output_folder)

    # Summary
    print("\n" + "="*60)
    print("BEST PHOTOS SUMMARY")
    print("="*60)
    for i, photo in enumerate(best_photos):
        score = photo.get('final_score', 0)
        faces = photo.get('num_faces', 0)
        print(f"{i+1:2d}. {photo['file_name']:30} Score: {score:.3f} | Faces: {faces}")

    print(f"\n🎯 Best photo(s) saved to: {output_folder}")
    logger.info(f"✅ Best photo(s) saved to {output_folder}")


if __name__ == "__main__":
    main()
