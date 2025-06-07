#!/usr/bin/env python3
"""
Video Refiner CLI - Advanced post-processing for WAN 2.1 generated videos
Uses Real-ESRGAN for upscaling, GFPGAN for face restoration, and RIFE for frame interpolation
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import json
import urllib.request
import tarfile
import zipfile
from tqdm import tqdm
import logging

# Fix for newer torchvision versions - must be applied before importing GFPGAN/BasicSR
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    try:
        import torchvision.transforms.functional as F
        sys.modules['torchvision.transforms.functional_tensor'] = F
        print("Applied torchvision compatibility patch")
    except ImportError:
        pass  # torchvision not installed yet


class VideoRefiner:
    def __init__(self, 
                 models_dir: Path = Path("./models"),
                 temp_dir: Optional[Path] = None,
                 device: str = "auto",
                 debug: bool = False):
        # Setup logging first to track initialization
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.logger.debug("Initializing VideoRefiner...")
        self.logger.debug(f"Parameters: models_dir={models_dir}, temp_dir={temp_dir}, device={device}, debug={debug}")
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = temp_dir
        self.device = device
        self.debug = debug
        
        self.logger.debug("Setting up model URLs...")
        
        # Model URLs
        self.model_urls = {
            "realesrgan": {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip",
                "models": {
                    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
                }
            },
            "gfpgan": {
                "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                "detection": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
                "parsing": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
            },
            "rife": {
                "url": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-macos.zip",
                "model": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-v4.6.tar.gz"
            }
        }
        
        # Binary paths
        self.realesrgan_bin = None
        self.rife_bin = None
        
        self.logger.debug("VideoRefiner initialization complete")
        
    def setup_environment(self) -> bool:
        """Download and setup all required models and binaries."""
        self.logger.info("Setting up environment...")
        
        # Check for required system tools
        if not self._check_ffmpeg():
            self.logger.error("ffmpeg not found. Please install: brew install ffmpeg")
            return False
        
        # Download models
        if not self._download_realesrgan():
            return False
            
        if not self._download_rife():
            return False
            
        # We'll use Python-based GFPGAN
        if not self._setup_gfpgan():
            return False
            
        self.logger.info("Environment setup complete!")
        return True
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is installed."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _download_file(self, url: str, dest: Path, desc: str) -> bool:
        """Download a file with progress bar."""
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Get file size
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
            
            # Download with progress
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                def hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    pbar.update(downloaded - pbar.n)
                
                urllib.request.urlretrieve(url, dest, reporthook=hook)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to download {desc}: {e}")
            return False
    
    def _download_realesrgan(self) -> bool:
        """Download Real-ESRGAN binary and models."""
        bin_dir = self.models_dir / "realesrgan"
        bin_path = bin_dir / "realesrgan-ncnn-vulkan"
        
        if bin_path.exists():
            self.realesrgan_bin = bin_path
            self.logger.info("Real-ESRGAN already installed")
            return True
        
        self.logger.info("Downloading Real-ESRGAN...")
        
        # Download binary
        zip_path = self.models_dir / "realesrgan.zip"
        if not self._download_file(self.model_urls["realesrgan"]["url"], zip_path, "Real-ESRGAN binary"):
            return False
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(bin_dir)
        
        # Make executable
        bin_path.chmod(0o755)
        self.realesrgan_bin = bin_path
        
        # Clean up
        zip_path.unlink()
        
        # Download models
        models_dir = bin_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_url in self.model_urls["realesrgan"]["models"].items():
            model_path = models_dir / f"{model_name}.pth"
            if not model_path.exists():
                if not self._download_file(model_url, model_path, f"Model: {model_name}"):
                    return False
        
        self.logger.info("Real-ESRGAN setup complete")
        return True
    
    def _download_rife(self) -> bool:
        """Download RIFE binary and models."""
        bin_dir = self.models_dir / "rife"
        # The actual binary is in the extracted subdirectory
        bin_path = bin_dir / "rife-ncnn-vulkan-20221029-macos" / "rife-ncnn-vulkan"
        
        if bin_path.exists():
            self.rife_bin = bin_path
            self.logger.info("RIFE already installed")
            return True
        
        self.logger.info("Downloading RIFE...")
        
        # Download binary
        zip_path = self.models_dir / "rife.zip"
        if not self._download_file(self.model_urls["rife"]["url"], zip_path, "RIFE binary"):
            return False
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(bin_dir)
        
        # Make executable
        bin_path.chmod(0o755)
        self.rife_bin = bin_path
        
        # Clean up
        zip_path.unlink()
        
        # Download model
        model_archive = self.models_dir / "rife-v4.6.tar.gz"
        if not self._download_file(self.model_urls["rife"]["model"], model_archive, "RIFE model"):
            return False
        
        # Extract model to the correct subdirectory
        extract_path = bin_dir / "rife-ncnn-vulkan-20221029-macos"
        with tarfile.open(model_archive, 'r:gz') as tar:
            tar.extractall(extract_path)
        
        model_archive.unlink()
        
        self.logger.info("RIFE setup complete")
        return True
    
    def _setup_gfpgan(self) -> bool:
        """Setup GFPGAN (will be used via Python)."""
        self.logger.info("Checking GFPGAN setup...")
        
        # Download GFPGAN model if needed
        model_path = self.models_dir / "GFPGANv1.4.pth"
        if not model_path.exists():
            self.logger.info("Downloading GFPGAN model...")
            # Use the correct URL from GFPGAN v1.3.0 release (v1.3.8 doesn't have the model file)
            model_url = self.model_urls["gfpgan"]["url"]
            if not self._download_file(model_url, model_path, "GFPGAN model"):
                # Try alternative download method
                self.logger.warning("Direct download failed, trying gdown...")
                try:
                    import gdown
                    # Alternative: Google Drive link
                    gdown.download("https://drive.google.com/uc?id=1sCZFVBwFZrmFn3fNQQfaWzqOuC59Vbm9", str(model_path), quiet=False)
                except Exception as e:
                    self.logger.error(f"Failed to download GFPGAN model: {e}")
                    return False
        
        self.logger.info("GFPGAN setup complete")
        return True
    
    def process_video(self,
                     input_path: Path,
                     output_path: Path,
                     upscale: int = 4,
                     enhance_face: bool = True,
                     interpolate: int = 2,
                     fps: Optional[float] = None,
                     upscale_model: Optional[str] = None,
                     strip_metadata: bool = True) -> bool:
        """Process a single video with all enhancements."""
        temp_dir_obj = None
        temp_path = None
        
        try:
            # Initialize temp directory
            if self.debug:
                # Create a persistent temp directory for debugging
                temp_path = Path(tempfile.mkdtemp(prefix="video_refiner_debug_"))
                self.logger.info(f"Debug mode: Using temp directory: {temp_path}")
                self.logger.info("This directory will NOT be deleted automatically.")
                temp_dir_obj = None
            else:
                temp_dir_obj = tempfile.TemporaryDirectory(prefix="video_refiner_")
                temp_path = Path(temp_dir_obj.name)
                self.logger.debug(f"Using temp directory: {temp_path}")
            
            # Log start of processing
            self.logger.info(f"Starting video processing: {input_path}")
            self.logger.debug(f"Options: upscale={upscale}, enhance_face={enhance_face}, interpolate={interpolate}, fps={fps}, strip_metadata={strip_metadata}")
            
            # Step 1: Extract frames
            self.logger.info(f"Extracting frames from {input_path.name}...")
            frames_dir = temp_path / "frames"
            frames_dir.mkdir()
            
            if not self._extract_frames(input_path, frames_dir, fps, strip_metadata):
                return False
            
            # Get video info
            video_info = self._get_video_info(input_path)
            self.logger.debug(f"Video info: {video_info}")
            
            # Step 2: Upscale frames
            if upscale > 1:
                self.logger.info(f"Upscaling frames {upscale}x...")
                upscaled_dir = temp_path / "upscaled"
                upscaled_dir.mkdir()
                
                if not self._upscale_frames(frames_dir, upscaled_dir, upscale, upscale_model):
                    return False
                
                frames_dir = upscaled_dir
            
            # Step 3: Enhance faces
            if enhance_face:
                self.logger.info("Enhancing faces...")
                enhanced_dir = temp_path / "enhanced"
                enhanced_dir.mkdir()
                
                if not self._enhance_faces(frames_dir, enhanced_dir):
                    return False
                
                frames_dir = enhanced_dir
            
            # Step 4: Interpolate frames
            if interpolate > 1:
                self.logger.info(f"Interpolating frames {interpolate}x...")
                interpolated_dir = temp_path / "interpolated"
                interpolated_dir.mkdir()
                
                if not self._interpolate_frames(frames_dir, interpolated_dir, interpolate):
                    return False
                
                frames_dir = interpolated_dir
                # Adjust FPS for interpolated video
                video_info["fps"] *= interpolate
            
            # Step 5: Compile video
            self.logger.info("Compiling final video...")
            if not self._compile_video(frames_dir, output_path, video_info):
                return False
            
            self.logger.info(f"✓ Successfully processed: {output_path}")
            
            # Clean up temp directory if not in debug mode
            if temp_dir_obj:
                temp_dir_obj.cleanup()
                
            return True
                
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            if self.debug and temp_path:
                self.logger.error(f"Debug mode: Temp directory preserved at: {temp_path}")
                self.logger.error("You can inspect the extracted frames and intermediate results there.")
            return False
    
    def _extract_frames(self, video_path: Path, output_dir: Path, fps: Optional[float] = None, strip_metadata: bool = True) -> bool:
        """Extract frames from video using ffmpeg."""
        try:
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-qscale:v", "1",
                "-qmin", "1",
                "-qmax", "1",
                "-vsync", "0"
            ]
            
            if fps:
                cmd.extend(["-r", str(fps)])
            
            # Add metadata stripping if enabled
            if strip_metadata:
                cmd.extend([
                    "-map_metadata", "-1",  # Remove all metadata
                    "-fflags", "+bitexact",  # Ensure deterministic output
                    "-flags:v", "+bitexact",
                    "-flags:a", "+bitexact"
                ])
                self.logger.debug("Stripping metadata from extracted frames")
            
            cmd.append(str(output_dir / "frame_%08d.png"))
            
            self.logger.debug(f"Extracting frames with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
                
            # Count extracted frames
            frame_count = len(list(output_dir.glob("frame_*.png")))
            self.logger.info(f"Extracted {frame_count} frames to {output_dir}")
            
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract frames: {e.stderr}")
            return False
    
    def _get_video_info(self, video_path: Path) -> dict:
        """Get video information using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                "-of", "json",
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            stream = info["streams"][0]
            
            # Parse frame rate
            fps_parts = stream["r_frame_rate"].split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1])
            
            return {
                "width": int(stream["width"]),
                "height": int(stream["height"]),
                "fps": fps,
                "codec": stream["codec_name"]
            }
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            # Return defaults
            return {"width": 1920, "height": 1080, "fps": 30.0, "codec": "h264"}
    
    def _upscale_frames(self, input_dir: Path, output_dir: Path, scale: int, model_override: Optional[str] = None) -> bool:
        """Upscale frames using Real-ESRGAN."""
        try:
            # Count input frames
            input_frames = sorted(list(input_dir.glob("*.png")))
            self.logger.debug(f"Found {len(input_frames)} frames to upscale in {input_dir}")
            
            if not input_frames:
                self.logger.error("No PNG frames found in input directory")
                return False
            
            # Get the model path
            model_path = self.models_dir / "realesrgan" / "models" / "realesrgan-x4plus.param"
            if not model_path.exists():
                # Try alternative model path structure
                model_path = self.models_dir / "realesrgan" / "realesrgan-x4plus.param"
                if not model_path.exists():
                    self.logger.error(f"Model file not found at {model_path}")
                    return False
            
            # Process frames individually for better reliability
            self.logger.info(f"Processing {len(input_frames)} frames individually...")
            
            # Progress tracking
            successful_frames = 0
            failed_frames = 0
            keep_debug_frame = self.debug  # Flag to keep one failed frame for debugging
            
            for i, frame_path in enumerate(tqdm(input_frames, desc="Upscaling frames")):
                output_frame_path = output_dir / frame_path.name
                
                # Build command for individual frame
                # Use appropriate model based on scale factor
                if model_override:
                    model_name = model_override
                elif scale == 2:
                    # Use animevideo model for 2x scaling - better for videos
                    model_name = "realesr-animevideov3"
                elif scale == 3:
                    model_name = "realesr-animevideov3"
                else:
                    # Use x4plus for 4x and above
                    model_name = "realesrgan-x4plus"
                
                cmd = [
                    "./realesrgan-ncnn-vulkan",
                    "-i", str(frame_path),
                    "-o", str(output_frame_path),
                    "-n", model_name,
                    "-s", str(scale),
                    "-f", "png",
                    "-t", "0"  # Auto tile size for better memory management
                ]
                
                # Add model path if it exists
                if model_path.exists():
                    # Since we're running from models/realesrgan directory, use relative path
                    cmd.extend(["-m", "models"])
                
                # Let Real-ESRGAN auto-detect GPU on macOS
                # The -g flag can cause issues on Apple Silicon
                if self.device == "cpu":
                    # Force CPU mode if explicitly requested
                    cmd.extend(["-g", "-1"])
                
                # Run command with current working directory set to models directory
                # This helps the binary find its dependencies
                cwd = self.models_dir / "realesrgan"
                
                # Always log the exact command for the first frame
                if i == 0:
                    self.logger.info(f"Real-ESRGAN command: {' '.join(cmd)}")
                    self.logger.info(f"Working directory: {cwd}")
                
                try:
                    # Run with timeout to prevent hanging
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True,
                        cwd=str(cwd),
                        timeout=60  # 60 seconds timeout per frame
                    )
                    
                    if result.returncode != 0:
                        # Log detailed error information
                        self.logger.error(f"Real-ESRGAN failed for frame {frame_path.name}")
                        self.logger.error(f"Exit code: {result.returncode}")
                        self.logger.error(f"STDOUT: {result.stdout}")
                        self.logger.error(f"STDERR: {result.stderr}")
                        
                        # If debug mode and this is the first failure, keep the frame for debugging
                        if keep_debug_frame and failed_frames == 0:
                            debug_frame_path = self.models_dir / f"debug_failed_frame_{frame_path.name}"
                            shutil.copy2(frame_path, debug_frame_path)
                            self.logger.info(f"Debug: Saved failed frame to {debug_frame_path}")
                            self.logger.info("Debug: You can test Real-ESRGAN manually with this frame")
                            keep_debug_frame = False  # Only keep one frame
                        
                        failed_frames += 1
                        # Resize original frame to match target resolution as fallback
                        self._resize_frame_fallback(frame_path, output_frame_path, scale)
                    else:
                        # Log stdout if debug mode
                        if self.debug and result.stdout.strip():
                            self.logger.debug(f"Real-ESRGAN output: {result.stdout}")
                            
                        # Verify output was created
                        if output_frame_path.exists():
                            successful_frames += 1
                        else:
                            self.logger.warning(f"Output not created for frame {frame_path.name}")
                            self.logger.warning(f"STDOUT: {result.stdout}")
                            self.logger.warning(f"STDERR: {result.stderr}")
                            failed_frames += 1
                            # Resize original frame to match target resolution as fallback
                            self._resize_frame_fallback(frame_path, output_frame_path, scale)
                            
                except subprocess.TimeoutExpired:
                    self.logger.error(f"Timeout processing frame {frame_path.name} (>60s)")
                    failed_frames += 1
                    # Resize original frame to match target resolution as fallback
                    self._resize_frame_fallback(frame_path, output_frame_path, scale)
                except Exception as e:
                    self.logger.error(f"Unexpected error processing frame {frame_path.name}: {type(e).__name__}: {e}")
                    failed_frames += 1
                    # Resize original frame to match target resolution as fallback
                    self._resize_frame_fallback(frame_path, output_frame_path, scale)
            
            # Check results
            output_frames = list(output_dir.glob("*.png"))
            self.logger.info(f"Upscaling complete: {successful_frames} succeeded, {failed_frames} failed")
            self.logger.info(f"Total output frames: {len(output_frames)}")
            
            # Consider it successful if we processed at least 90% of frames
            success_rate = successful_frames / len(input_frames) if input_frames else 0
            if success_rate >= 0.9:
                return True
            else:
                self.logger.error(f"Too many frames failed ({failed_frames}/{len(input_frames)})")
                return False
                
        except Exception as e:
            self.logger.error(f"Unexpected error during upscaling: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _resize_frame_fallback(self, input_path: Path, output_path: Path, scale: int) -> None:
        """Resize a frame to match the target upscaled resolution."""
        try:
            import cv2
            
            # Read the original frame
            img = cv2.imread(str(input_path))
            if img is None:
                self.logger.error(f"Failed to read image for resizing: {input_path}")
                # As last resort, copy the original
                shutil.copy2(input_path, output_path)
                return
            
            # Calculate target dimensions
            height, width = img.shape[:2]
            new_width = width * scale
            new_height = height * scale
            
            # Resize using cv2 (bicubic interpolation)
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Save the resized frame
            cv2.imwrite(str(output_path), resized)
            self.logger.debug(f"Resized fallback frame from {width}x{height} to {new_width}x{new_height}")
            
        except Exception as e:
            self.logger.error(f"Failed to resize frame {input_path}: {e}")
            # As last resort, copy the original
            shutil.copy2(input_path, output_path)
    
    def _enhance_faces(self, input_dir: Path, output_dir: Path) -> bool:
        """Enhance faces in frames using GFPGAN."""
        try:
            # Import GFPGAN (assuming it's installed)
            from gfpgan import GFPGANer
            import cv2
            
            # Initialize GFPGAN
            restorer = GFPGANer(
                model_path=str(self.models_dir / "GFPGANv1.4.pth"),
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
            # Process each frame
            frames = sorted(input_dir.glob("*.png"))
            for frame_path in tqdm(frames, desc="Enhancing faces"):
                img = cv2.imread(str(frame_path))
                
                # Enhance
                _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False)
                
                # Save
                output_path = output_dir / frame_path.name
                cv2.imwrite(str(output_path), output)
            
            return True
        except ImportError:
            self.logger.warning("GFPGAN not installed. Copying frames without face enhancement.")
            # Copy frames without enhancement
            for frame in input_dir.glob("*.png"):
                shutil.copy2(frame, output_dir / frame.name)
            return True
        except Exception as e:
            self.logger.error(f"Failed to enhance faces: {e}")
            return False
    
    def _interpolate_frames(self, input_dir: Path, output_dir: Path, factor: int) -> bool:
        """Interpolate frames using RIFE."""
        try:
            # Count input frames to calculate target frame count
            input_frames = list(input_dir.glob("*.png"))
            target_frame_count = len(input_frames) * factor
            
            cmd = [
                str(self.rife_bin),
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-m", str(self.models_dir / "rife" / "rife-ncnn-vulkan-20221029-macos" / "rife-v4.6"),
                "-n", str(target_frame_count),
                "-f", "%08d.png"
            ]
            
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to interpolate frames: {e}")
            return False
    
    def _compile_video(self, frames_dir: Path, output_path: Path, video_info: dict) -> bool:
        """Compile frames back into video using ffmpeg."""
        try:
            # Check for available frames and determine naming pattern
            frame_patterns = [
                ("frame_%08d.png", "frame_*.png"),  # Original extraction pattern
                ("%08d.png", "[0-9]*.png")           # RIFE interpolation pattern
            ]
            
            input_pattern = None
            frame_count = 0
            
            for pattern, glob_pattern in frame_patterns:
                frames = list(frames_dir.glob(glob_pattern))
                if frames:
                    input_pattern = pattern
                    frame_count = len(frames)
                    self.logger.info(f"Found {frame_count} frames with pattern: {pattern}")
                    break
            
            if not input_pattern or frame_count == 0:
                self.logger.error(f"No frames found in {frames_dir}")
                self.logger.error(f"Directory contents: {list(frames_dir.iterdir())[:10]}...")  # Show first 10 files
                return False
            
            # Get actual dimensions from the processed frames
            sample_frame = next(frames_dir.glob("*.png"), None)
            if sample_frame:
                # Use ffprobe to get the actual frame dimensions
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "json",
                    str(sample_frame)
                ]
                
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                info = json.loads(result.stdout)
                if info.get("streams"):
                    actual_width = int(info["streams"][0]["width"])
                    actual_height = int(info["streams"][0]["height"])
                    self.logger.info(f"Using actual frame dimensions: {actual_width}x{actual_height}")
                else:
                    actual_width = video_info["width"]
                    actual_height = video_info["height"]
            else:
                actual_width = video_info["width"]
                actual_height = video_info["height"]
            
            cmd = [
                "ffmpeg", "-y",
                "-r", str(video_info["fps"]),
                "-i", str(frames_dir / input_pattern),
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                # Ensure output dimensions match the processed frames
                "-vf", f"scale={actual_width}:{actual_height}:force_original_aspect_ratio=decrease,pad={actual_width}:{actual_height}:(ow-iw)/2:(oh-ih)/2",
                str(output_path)
            ]
            
            self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"FFmpeg failed with exit code: {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                self.logger.error(f"STDOUT: {result.stdout}")
                return False
                
            self.logger.info(f"Successfully compiled video to: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to compile video: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"FFmpeg error: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during video compilation: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def process_batch(self, input_files: List[Path], output_dir: Path, **kwargs) -> None:
        """Process multiple videos."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = 0
        failed = 0
        
        for input_file in input_files:
            self.logger.info(f"\nProcessing {input_file.name} ({success + failed + 1}/{len(input_files)})")
            
            output_file = output_dir / f"refined_{input_file.stem}.mp4"
            
            if self.process_video(input_file, output_file, **kwargs):
                success += 1
            else:
                failed += 1
        
        self.logger.info(f"\nBatch complete: {success} succeeded, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Video Refiner - Advanced post-processing for AI-generated videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup (download models)
  %(prog)s --setup

  # Process single video with all enhancements
  %(prog)s input.mp4 -o output.mp4 --upscale 4 --face --interpolate 2

  # Batch process directory
  %(prog)s videos/*.mp4 -o refined/ --upscale 2

  # Upscale only
  %(prog)s video.mp4 -o upscaled.mp4 --upscale 4 --no-face --no-interpolate
        """
    )
    
    parser.add_argument('input', nargs='*', 
                        help='Input video file(s) or pattern')
    parser.add_argument('-o', '--output',
                        help='Output file (single) or directory (batch)')
    parser.add_argument('--setup', action='store_true',
                        help='Download and setup required models')
    parser.add_argument('-u', '--upscale', type=int, default=4, choices=[1, 2, 4, 8],
                        help='Upscaling factor (default: 4)')
    parser.add_argument('--face', action='store_true', default=True,
                        help='Enable face enhancement (default)')
    parser.add_argument('--no-face', dest='face', action='store_false',
                        help='Disable face enhancement')
    parser.add_argument('-i', '--interpolate', type=int, default=2, choices=[1, 2, 4, 8],
                        help='Frame interpolation factor (default: 2)')
    parser.add_argument('--no-interpolate', dest='interpolate', action='store_const', const=1,
                        help='Disable frame interpolation')
    parser.add_argument('--fps', type=float,
                        help='Override input FPS during extraction')
    parser.add_argument('--models-dir', type=Path, default=Path("./models"),
                        help='Directory for downloaded models')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto',
                        help='Processing device (default: auto)')
    parser.add_argument('--upscale-model', choices=['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime', 'realesrnet-x4plus'],
                        help='Upscaling model to use (default: auto-select based on scale)')
    parser.add_argument('--strip-metadata', action='store_true', default=True,
                        help='Strip metadata from frames before upscaling (default: True)')
    parser.add_argument('--no-strip-metadata', dest='strip_metadata', action='store_false',
                        help='Keep metadata in frames')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (preserves temp directories and shows verbose output)')
    
    args = parser.parse_args()
    
    # Initialize refiner
    print(f"Initializing VideoRefiner with debug={args.debug}")
    refiner = VideoRefiner(
        models_dir=args.models_dir,
        device=args.device,
        debug=args.debug
    )
    print("VideoRefiner initialized successfully")
    
    # Setup mode
    if args.setup:
        if refiner.setup_environment():
            print("\n✓ Setup complete! You can now process videos.")
        else:
            print("\n✗ Setup failed. Please check the errors above.")
        return
    
    # Check if models are set up
    if not refiner.realesrgan_bin or not refiner.rife_bin:
        print("Models not yet initialized, setting up environment...")
        if not refiner.setup_environment():
            print("\n✗ Failed to setup environment. Run with --setup first.")
            sys.exit(1)
    
    # Validate inputs
    if not args.input:
        parser.error("No input files specified. Use --setup for first-time setup.")
    
    if not args.output:
        parser.error("Output path required")
    
    # Expand input files
    input_files = []
    for pattern in args.input:
        path = Path(pattern)
        if path.is_file():
            input_files.append(path)
        else:
            parent = path.parent if path.parent.exists() else Path('.')
            matches = list(parent.glob(path.name))
            input_files.extend(matches)
    
    if not input_files:
        parser.error("No input files found")
    
    # Process based on input count
    if len(input_files) == 1:
        # Single file mode
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / f"refined_{input_files[0].stem}.mp4"
        
        success = refiner.process_video(
            input_files[0],
            output_path,
            upscale=args.upscale,
            enhance_face=args.face,
            interpolate=args.interpolate,
            fps=args.fps,
            upscale_model=args.upscale_model,
            strip_metadata=args.strip_metadata
        )
        
        if not success:
            sys.exit(1)
    else:
        # Batch mode
        output_dir = Path(args.output)
        if output_dir.exists() and not output_dir.is_dir():
            parser.error("Output must be a directory for batch processing")
        
        refiner.process_batch(
            input_files,
            output_dir,
            upscale=args.upscale,
            enhance_face=args.face,
            interpolate=args.interpolate,
            fps=args.fps,
            upscale_model=args.upscale_model,
            strip_metadata=args.strip_metadata
        )


if __name__ == "__main__":
    main()