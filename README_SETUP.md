# Video Refiner Setup Guide

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   For Apple Silicon Macs, you may want to install PyTorch separately for better performance:
   ```bash
   pip3 install torch torchvision torchaudio
   ```

2. **Download required models:**
   ```bash
   python video_refiner.py --setup
   ```

   This will download:
   - Real-ESRGAN models and binary
   - RIFE models and binary
   - GFPGAN models (installed via pip)

3. **Verify installation:**
   ```bash
   # Process a test video
   python video_refiner.py input.mp4 -o output.mp4 --upscale 2 --no-face --no-interpolate
   ```

## System Requirements

- Python 3.8 or higher
- FFmpeg (must be installed separately)
- macOS (for included binaries) or Linux
- 8GB+ RAM recommended
- GPU recommended for faster processing

## Notes

- The `.gitignore` file excludes large model files and video files from version control
- Model binaries are downloaded to the `models/` directory
- Temporary files are created in system temp directory (or preserved with `--debug` flag)
- Processing time depends on video resolution and length (~15 seconds per frame for 2x upscaling)

## Troubleshooting

If Real-ESRGAN crashes on macOS:
1. Check that the binary has execute permissions: `chmod +x models/realesrgan/realesrgan-ncnn-vulkan`
2. Remove quarantine attribute: `xattr -d com.apple.quarantine models/realesrgan/realesrgan-ncnn-vulkan`
3. Use `--debug` flag to see detailed error messages