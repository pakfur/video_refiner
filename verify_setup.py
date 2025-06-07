#!/usr/bin/env python3
"""
Verification script to ensure all components are properly installed
"""

import sys
import os
from pathlib import Path
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Python 3.8+ is recommended")
        return False
    return True

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            print(f"✓ {first_line}")
            return True
    except FileNotFoundError:
        pass
    print("✗ ffmpeg not found - install with: brew install ffmpeg")
    return False

def check_python_packages():
    """Check required Python packages"""
    required_packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'gfpgan': 'gfpgan',
        'facexlib': 'facexlib',
        'basicsr': 'basicsr'
    }
    
    all_installed = True
    for module, package in required_packages.items():
        try:
            if module == 'cv2':
                import cv2
                print(f"✓ {package} ({cv2.__version__})")
            elif module == 'PIL':
                import PIL
                print(f"✓ {package} ({PIL.__version__})")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'installed')
                print(f"✓ {package} ({version})")
        except ImportError:
            print(f"✗ {package} not installed")
            all_installed = False
    
    # Special check for GFPGAN with compatibility fix
    try:
        # Apply the compatibility fix
        class FunctionalTensorRedirect:
            def __getattr__(self, name):
                import torchvision.transforms.functional as F
                return getattr(F, name)
        sys.modules['torchvision.transforms.functional_tensor'] = FunctionalTensorRedirect()
        
        from gfpgan import GFPGANer
        print("✓ GFPGAN import test passed (with compatibility fix)")
    except Exception as e:
        print(f"✗ GFPGAN import failed: {e}")
        all_installed = False
    
    return all_installed

def check_models():
    """Check if required models are downloaded"""
    models_dir = Path("./models")
    
    required_files = [
        ("GFPGAN model", models_dir / "GFPGANv1.4.pth"),
        ("Real-ESRGAN binary", models_dir / "realesrgan" / "realesrgan-ncnn-vulkan"),
        ("Real-ESRGAN x4plus model", models_dir / "realesrgan" / "models" / "RealESRGAN_x4plus.pth"),
        ("Real-ESRGAN x4plus net model", models_dir / "realesrgan" / "models" / "RealESRNet_x4plus.pth"),
        ("RIFE binary", models_dir / "rife" / "rife-ncnn-vulkan-20221029-macos" / "rife-ncnn-vulkan"),
        ("RIFE v4.6 flownet", models_dir / "rife" / "rife-ncnn-vulkan-20221029-macos" / "rife-v4.6" / "flownet.bin"),
        ("RIFE v4.6 params", models_dir / "rife" / "rife-ncnn-vulkan-20221029-macos" / "rife-v4.6" / "flownet.param"),
    ]
    
    all_present = True
    for name, path in required_files:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {name} ({size_mb:.1f} MB)")
            
            # Check if binaries are executable
            if path.suffix == "" and "binary" in name.lower():
                if os.access(path, os.X_OK):
                    print(f"  ✓ Executable permissions set")
                else:
                    print(f"  ✗ Not executable - run: chmod +x {path}")
                    all_present = False
        else:
            print(f"✗ {name} not found at {path}")
            all_present = False
    
    return all_present

def main():
    print("Video Refiner Setup Verification")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("FFmpeg", check_ffmpeg),
        ("Python Packages", check_python_packages),
        ("Model Files", check_models),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All checks passed! You're ready to use Video Refiner.")
        print("\nExample usage:")
        print("  python video_refiner.py input.mp4 -o output.mp4")
    else:
        print("❌ Some checks failed. Please address the issues above.")
        print("\nTo download missing models, run:")
        print("  python video_refiner.py --setup")
        print("  # or")
        print("  ./setup_models.sh")

if __name__ == "__main__":
    main()