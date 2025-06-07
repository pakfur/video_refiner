#!/bin/bash
# Alternative setup script for downloading models manually

echo "Video Refiner - Model Setup Script"
echo "=================================="

# Create directories
mkdir -p models/realesrgan/models
mkdir -p models/rife
mkdir -p models/gfpgan

# Download Real-ESRGAN
echo "Downloading Real-ESRGAN..."
if [[ $(uname -m) == "arm64" ]]; then
    curl -L "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip" -o models/realesrgan.zip
else
    echo "Note: This script is optimized for Apple Silicon. Intel Mac users may need different binaries."
fi

unzip -o models/realesrgan.zip -d models/realesrgan/
chmod +x models/realesrgan/realesrgan-ncnn-vulkan
rm models/realesrgan.zip

# Download Real-ESRGAN models
echo "Downloading Real-ESRGAN models..."
curl -L "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -o models/realesrgan/models/RealESRGAN_x4plus.pth
curl -L "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth" -o models/realesrgan/models/RealESRNet_x4plus.pth

# Download RIFE
echo "Downloading RIFE..."
curl -L "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-macos.zip" -o models/rife.zip
unzip -o models/rife.zip -d models/rife/
chmod +x models/rife/rife-ncnn-vulkan-20221029-macos/rife-ncnn-vulkan
rm models/rife.zip

# Download RIFE model
echo "Downloading RIFE model..."
curl -L "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-v4.6.tar.gz" -o models/rife-v4.6.tar.gz
# Extract to the correct subdirectory where the binary expects it
tar -xzf models/rife-v4.6.tar.gz -C models/rife/rife-ncnn-vulkan-20221029-macos/
rm models/rife-v4.6.tar.gz

# Download GFPGAN model
echo "Downloading GFPGAN model..."
curl -L "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -o models/GFPGANv1.4.pth

echo ""
echo "Setup complete! Models downloaded to ./models/"
echo "You can now run: python video_refiner.py input.mp4 -o output.mp4"