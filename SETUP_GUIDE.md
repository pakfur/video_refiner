# Video Refiner Setup Guide

This guide ensures proper installation of all dependencies and models for the Video Refiner tool.

## Prerequisites

1. **macOS** (optimized for Apple Silicon)
2. **Python 3.8+**
3. **ffmpeg** installed: `brew install ffmpeg`
4. **Sufficient disk space** (~2GB for models)

## Installation Steps

### 1. Clone and Setup Virtual Environment

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd video

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Apple Silicon (if on M1/M2/M3 Mac)
pip install torch torchvision torchaudio

# Install all requirements
pip install -r requirements.txt
```

### 3. Download Models

#### Option A: Using the Python Script
```bash
python video_refiner.py --setup
```

#### Option B: Using the Shell Script
```bash
chmod +x setup_models.sh
./setup_models.sh
```

### 4. Verify Installation

```bash
# Test if all dependencies are installed correctly
python -c "
import cv2
import torch
import numpy as np
print('OpenCV:', cv2.__version__)
print('PyTorch:', torch.__version__)
print('NumPy:', np.__version__)

# Test GFPGAN import with compatibility fix
import sys
class FunctionalTensorRedirect:
    def __getattr__(self, name):
        import torchvision.transforms.functional as F
        return getattr(F, name)
sys.modules['torchvision.transforms.functional_tensor'] = FunctionalTensorRedirect()
from gfpgan import GFPGANer
print('GFPGAN: Import successful')
"
```

## Model Structure

After setup, your models directory should look like:

```
models/
├── GFPGANv1.4.pth
├── realesrgan/
│   ├── realesrgan-ncnn-vulkan
│   └── models/
│       ├── RealESRGAN_x4plus.pth
│       └── RealESRNet_x4plus.pth
└── rife/
    └── rife-ncnn-vulkan-20221029-macos/
        ├── rife-ncnn-vulkan
        └── rife-v4.6/
            ├── flownet.bin
            └── flownet.param
```

## Troubleshooting

### GFPGAN Import Error
If you see `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`, this is due to newer torchvision versions. The video_refiner.py script includes an automatic fix for this.

### RIFE Model Not Found
Ensure the RIFE model is extracted to the correct path: `models/rife/rife-ncnn-vulkan-20221029-macos/rife-v4.6/`

### Permission Denied
Make sure the binary files are executable:
```bash
chmod +x models/realesrgan/realesrgan-ncnn-vulkan
chmod +x models/rife/rife-ncnn-vulkan-20221029-macos/rife-ncnn-vulkan
```

### Apple Silicon Specific
For M1/M2/M3 Macs, ensure you're using ARM64 compatible binaries and PyTorch with MPS support.

## Usage

After successful setup:

```bash
# Process a single video
python video_refiner.py input.mp4 -o output.mp4

# With custom options
python video_refiner.py input.mp4 -o output.mp4 --upscale 4 --face --interpolate 2
```