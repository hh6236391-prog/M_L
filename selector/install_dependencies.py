import subprocess
import sys
import os

def install_requirements():
    """Install requirements with error handling"""
    requirements = [
        "opencv-python==4.8.1.78",
        "numpy==1.24.3", 
        "Pillow==10.0.1",
        "scikit-image==0.21.0",
        "mediapipe==0.10.21",  # or remove this line for OpenCV-only version
        "PyYAML==6.0.1",
        "matplotlib==3.7.2"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            
            # Try without version specifier
            try:
                package_name = package.split('==')[0]
                print(f"Trying to install {package_name} without version constraint...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"✓ {package_name} installed successfully")
            except:
                print(f"✗ Could not install {package_name}")

if __name__ == "__main__":
    install_requirements()
    print("\nInstallation completed!")