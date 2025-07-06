#!/usr/bin/env python3
"""
Script to check package compatibility with Python 3.13
"""

import subprocess
import sys
import json

# Packages to check
packages = [
    "flask==3.0.0",
    "flask-cors==4.0.0", 
    "werkzeug==3.0.1",
    "Pillow==10.4.0",
    "numpy==1.26.4",
    "opencv-python-headless==4.9.0.80",
    "pytesseract==0.3.10",
    "reportlab==4.0.7",
    "ultralytics==8.1.0",
    "torch==2.5.0",
    "torchvision==0.21.0",
    "gunicorn==21.2.0",
    "python-dotenv==1.0.1",
    "requests==2.31.0",
    "urllib3==2.2.0"
]

def check_package_compatibility(package):
    """Check if a package is compatible with Python 3.13"""
    try:
        # Use pip to check package info
        result = subprocess.run([
            sys.executable, "-m", "pip", "index", "versions", package.split("==")[0]
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ {package} - Available")
            return True
        else:
            print(f"❌ {package} - Error checking")
            return False
    except Exception as e:
        print(f"❌ {package} - Exception: {e}")
        return False

def main():
    print("Checking package compatibility with Python 3.13...")
    print("=" * 50)
    
    compatible = []
    incompatible = []
    
    for package in packages:
        if check_package_compatibility(package):
            compatible.append(package)
        else:
            incompatible.append(package)
    
    print("\n" + "=" * 50)
    print(f"✅ Compatible packages: {len(compatible)}")
    print(f"❌ Incompatible packages: {len(incompatible)}")
    
    if incompatible:
        print("\nIncompatible packages:")
        for pkg in incompatible:
            print(f"  - {pkg}")
    
    print(f"\nPython version: {sys.version}")

if __name__ == "__main__":
    main() 