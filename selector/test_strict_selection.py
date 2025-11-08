import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.photo_analyzer import PhotoAnalyzer
from src.utils import load_config, get_image_files

def test_strict_selection(folder_path):
    config = load_config("config/default_config.yaml")
    analyzer = PhotoAnalyzer(config)
    
    image_files = get_image_files(folder_path)
    
    print(f"Analyzing {len(image_files)} images with STRICT selection")
    print("=" * 80)
    
    results = []
    for image_path in image_files:
        analysis = analyzer.analyze_image(image_path)
        results.append(analysis)
    
    # Apply differentiation
    results = analyzer._apply_score_differentiation(results)
    
    # Sort by score
    results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    print("\nTOP 3 PHOTOS (with scores):")
    print("-" * 50)
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result['file_name']}")
        print(f"   Score: {result['final_score']:.3f}")
        print(f"   Face: {result['face_detected']} (Confidence: {result.get('primary_face_confidence', 0):.3f})")
        print(f"   Sharpness: {result.get('sharpness', 0):.1f}")
        print(f"   Brightness: {result.get('brightness', 0):.1f}")
        print(f"   Pose Quality: {result.get('pose_quality', 0):.3f}")
        print()
    
    # Select single best
    best_photo = analyzer.select_single_best_photo(results)
    if best_photo:
        print("🎯 SINGLE BEST PHOTO SELECTED:")
        print("=" * 50)
        print(f"📸 {best_photo['file_name']}")
        print(f"⭐ Final Score: {best_photo['final_score']:.3f}")
        print(f"👤 Face Confidence: {best_photo.get('primary_face_confidence', 0):.3f}")
        print(f"🔍 Sharpness: {best_photo.get('sharpness', 0):.1f}")
        print(f"💡 Brightness: {best_photo.get('brightness', 0):.1f}")
        print(f"🎭 Pose Quality: {best_photo.get('pose_quality', 0):.3f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_strict_selection(sys.argv[1])
    else:
        print("Usage: python test_strict_selection.py /path/to/your/photos")