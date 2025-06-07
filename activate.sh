#!/bin/bash
# Activation script for video_refiner environment

echo "Activating video_refiner Python 3.11 environment..."
source video_refiner_env/bin/activate
echo "✓ Environment activated. Python version:"
python --version
echo ""
echo "You can now run video_refiner.py without the conflicting GFPGAN messages!"
echo "Example: python video_refiner.py --setup"
echo ""

