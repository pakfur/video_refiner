# Video Refiner

Advanced post-processing tool for AI-generated videos using Real-ESRGAN, GFPGAN, and RIFE.

## Requirements

- Python 3.11+ (3.11.12 recommended)
- macOS (Apple Silicon supported)
- ffmpeg

## Setup

### Option 1: Use the pre-configured environment

```bash
# Activate the Python 3.11 environment with all dependencies
source activate.sh

# Setup models (first time only)
python video_refiner.py --setup
```

### Option 2: Create your own environment

```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup models
python video_refiner.py --setup
```

## Fixed Issues

✅ **GFPGAN Compatibility**: Resolved conflicting "GFPGAN setup complete" vs "WARNING - GFPGAN not installed" messages

✅ **Python 3.12 Compatibility**: Updated to Python 3.11 to avoid numpy/setuptools conflicts

✅ **Torchvision Compatibility**: Added automatic patch for `torchvision.transforms.functional_tensor` import issues

## Usage

```bash
# Process a single video
python video_refiner.py input.mp4 -o output.mp4 --upscale 4 --face --interpolate 2

# Batch process multiple videos
python video_refiner.py videos/*.mp4 -o refined/ --upscale 2

# Upscale only (no face enhancement or interpolation)
python video_refiner.py video.mp4 -o upscaled.mp4 --upscale 4 --no-face --no-interpolate
```

## Package Versions

The requirements.txt has been updated with compatible versions:
- Python 3.11.12
- PyTorch 2.7.1
- Torchvision 0.22.1
- GFPGAN 1.3.8
- All other dependencies updated to latest compatible versions

## Features

- **Upscaling**: 2x, 4x, or 8x upscaling using Real-ESRGAN
- **Face Enhancement**: Restore and enhance faces using GFPGAN
- **Frame Interpolation**: Smooth motion with 2x, 4x, or 8x frame interpolation using RIFE
- **Batch Processing**: Process multiple videos in parallel
- **Apple Silicon Optimized**: Automatically uses Metal Performance Shaders on M-series Macs

## Command Line Options

- `input`: Input video file(s) or pattern (supports wildcards)
- `-o, --output`: Output file path (single video) or directory (batch)
- `--setup`: Download and setup required models (first-time only)
- `-u, --upscale`: Upscaling factor (1, 2, 4, or 8; default: 4)
- `--face/--no-face`: Enable/disable face enhancement (default: enabled)
- `-i, --interpolate`: Frame interpolation factor (1, 2, 4, or 8; default: 2)
- `--no-interpolate`: Disable frame interpolation
- `--fps`: Override input FPS during frame extraction
- `--models-dir`: Directory for downloaded models (default: ./models)
- `--device`: Processing device (auto, cpu, or mps; default: auto)

## Troubleshooting

If you see torchvision import errors, the script will automatically apply a compatibility patch. This is normal and expected.

### Common Issues

- **"ffmpeg not found"**: Install with `brew install ffmpeg`
- **GFPGAN import errors**: Use Python 3.11 and the updated requirements.txt
- **Model download fails**: Check internet connection and retry `--setup`
- **Out of memory**: Reduce upscaling factor or process shorter segments

## License

This tool uses the following open-source projects:
- Real-ESRGAN (BSD 3-Clause License)
- GFPGAN (Apache 2.0 License)
- RIFE (MIT License)
