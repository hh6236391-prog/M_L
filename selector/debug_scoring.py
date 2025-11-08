import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.photo_analyzer import PhotoAnalyzer
from src.utils import load_config, get_image_files

def debug_scoring(folder_path):
    config = load_config("config/default_config.yaml")
    analyzer = PhotoAnalyzer(config)
    
    image_files = get_image_files(folder_path)
    
    print(f"🔍 DEBUGGING SCORING FOR {len(image_files)} IMAGES")
    print("=" * 80)
    
    results = []
    for image_path in image_files:
        print(f"\n📸 ANALYZING: {image_path.name}")
        print("-" * 40)
        
        analysis = analyzer.analyze_image(image_path)
        results.append(analysis)
        
        # Print key metrics
        print(f"✅ Final Score: {analysis['final_score']:.3f}")
        print(f"👤 Face Detected: {analysis['face_detected']}")
        if analysis['face_detected']:
            print(f"   Confidence: {analysis.get('primary_face_confidence', 0):.3f}")
            print(f"   Area Ratio: {analysis.get('primary_face_area_ratio', 0):.3f}")
        
        print(f"📊 Quality Metrics:")
        print(f"   Sharpness: {analysis.get('sharpness', 0):.1f}")
        print(f"   Brightness: {analysis.get('brightness', 0):.1f} ({analysis.get('exposure_status', 'N/A')})")
        print(f"   Contrast: {analysis.get('contrast', 0):.1f}")
        print(f"   White Balance: {analysis.get('white_balance', 0):.3f}")
        print(f"   Pose Quality: {analysis.get('pose_quality', 0):.3f}")
        print(f"   Composition: {analysis.get('composition', 0):.3f}")
        
        if analysis['rejection_reasons']:
            print(f"❌ Rejection Reasons: {', '.join(analysis['rejection_reasons'])}")
        else:
            print(f"✅ No critical issues")
        
        if analysis.get('minor_issues'):
            print(f"⚠️  Minor Issues: {', '.join(analysis['minor_issues'])}")
    
    # Show ranking
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    print("\n" + "=" * 80)
    print("🏆 FINAL RANKING")
    print("=" * 80)
    
    for i, result in enumerate(results):
        status = "✅ SELECTED" if i == 0 else "📝"
        print(f"{i+1:2d}. {status} {result['file_name']:30} Score: {result['final_score']:.3f}")
        
        if i == 0 and result['final_score'] < 0.2:
            print("   💡 SUGGESTION: Lower minimum score threshold to 0.1")
    
    # Check if any photos meet different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4]
    print(f"\n📈 THRESHOLD ANALYSIS:")
    for threshold in thresholds:
        count = sum(1 for r in results if r['final_score'] >= threshold)
        print(f"   Score ≥ {threshold}: {count} photos")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_scoring(sys.argv[1])
    else:
        print("Usage: python debug_scoring.py /path/to/your/photos")