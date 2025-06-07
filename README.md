# Video Refiner

Advanced video post-processing tool for AI-generated videos (WAN 2.1 and others). Combines multiple state-of-the-art techniques for video enhancement.

## Features

- **Upscaling**: 2x, 4x, or 8x upscaling using Real-ESRGAN
- **Face Enhancement**: Restore and enhance faces using GFPGAN
- **Frame Interpolation**: Smooth motion with 2x, 4x, or 8x frame interpolation using RIFE
- **Batch Processing**: Process multiple videos in parallel
- **Apple Silicon Optimized**: Automatically uses Metal Performance Shaders on M-series Macs

## Requirements

- macOS (tested on M4 Pro with 24GB RAM)
- Python 3.8+
- ffmpeg (install via Homebrew: `brew install ffmpeg`)

## Installation

1. Clone or download this repository:
```bash
cd /Users/jk/training/video
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download models and binaries (one-time setup):
```bash
python video_refiner.py --setup
```

This will download:
- Real-ESRGAN models and binaries (~100MB)
- GFPGAN face restoration model (~350MB)
- RIFE frame interpolation models (~50MB)

## Usage

### Basic Usage

Process a single video with default settings (4x upscale, face enhancement, 2x interpolation):
```bash
python video_refiner.py input.mp4 -o output.mp4
```

### Advanced Options

```bash
# Upscale only (no face enhancement or interpolation)
python video_refiner.py input.mp4 -o output.mp4 --upscale 4 --no-face --no-interpolate

# Maximum quality (8x upscale, face enhancement, 4x interpolation)
python video_refiner.py input.mp4 -o output.mp4 --upscale 8 --interpolate 4

# Batch process all MP4 files in a directory
python video_refiner.py videos/*.mp4 -o refined/

# Custom FPS extraction (useful for high frame rate videos)
python video_refiner.py input.mp4 -o output.mp4 --fps 60
```

### Command Line Options

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

## Processing Pipeline

1. **Frame Extraction**: Uses ffmpeg to extract frames as lossless PNG images
2. **Upscaling**: Applies Real-ESRGAN for high-quality upscaling
3. **Face Enhancement**: Detects and enhances faces using GFPGAN
4. **Frame Interpolation**: Generates intermediate frames using RIFE
5. **Video Compilation**: Reassembles frames into final video with H.264 encoding

## Performance Tips

- The tool automatically uses all available CPU cores for parallel processing
- On Apple Silicon Macs, GPU acceleration is used when available
- Processing time depends on video resolution, length, and selected enhancements
- Typical processing: ~2-5 minutes per minute of 1080p video with 4x upscaling

## Troubleshooting

### "ffmpeg not found" error
Install ffmpeg using Homebrew:
```bash
brew install ffmpeg
```

### Model download fails
1. Check your internet connection
2. Try running setup again: `python video_refiner.py --setup`
3. Models are downloaded to `./models` directory by default

### Out of memory errors
- Reduce upscaling factor
- Process shorter video segments
- Close other applications to free up RAM

### GFPGAN import error
Ensure all requirements are installed:
```bash
pip install -r requirements.txt
```

## Examples

### Quick Test
```bash
# Download a sample video (optional)
# Process with default settings
python video_refiner.py sample.mp4 -o enhanced.mp4
```

### Production Pipeline
```bash
# High quality processing for final output
python video_refiner.py raw_video.mp4 -o final_video.mp4 --upscale 4 --interpolate 2
```

### Batch Processing
```bash
# Process all videos in a folder
python video_refiner.py /path/to/videos/*.mp4 -o /path/to/output/
```

## Notes

- Input videos are not modified; always outputs to a new file
- Supports common video formats: MP4, AVI, MOV, MKV
- Output is always MP4 with H.264 encoding for compatibility
- Temporary files are automatically cleaned up after processing

## License

This tool uses the following open-source projects:
- Real-ESRGAN (BSD 3-Clause License)
- GFPGAN (Apache 2.0 License)
- RIFE (MIT License)