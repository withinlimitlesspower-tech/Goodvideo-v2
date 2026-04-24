```py
"""
FFmpeg Handler Module for AI Video Generator

This module provides a robust wrapper around FFmpeg commands to generate
videos from sequences of images and audio files. It handles video composition,
timing, transitions, and output formatting.

Features:
- Generate videos from image sequences with configurable duration
- Add audio/voiceover tracks to video
- Support for various output formats and codecs
- Progress tracking and error handling
- Input validation and security sanitization
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Custom exception for FFmpeg-related errors."""
    pass


class FFmpegHandler:
    """
    Wrapper class for FFmpeg video generation operations.
    
    Handles the creation of videos from image sequences and audio files,
    with support for various configurations and error handling.
    """

    # Default FFmpeg arguments for high-quality output
    DEFAULT_ENCODER_ARGS = [
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest"
    ]

    # Allowed image extensions for security validation
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    # Allowed audio extensions
    ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.ogg', '.m4a'}

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        temp_dir: Optional[str] = None,
        max_file_size_mb: int = 500,
        timeout_seconds: int = 300
    ):
        """
        Initialize the FFmpeg handler.

        Args:
            ffmpeg_path: Path to FFmpeg executable (default: 'ffmpeg')
            ffprobe_path: Path to FFprobe executable (default: 'ffprobe')
            temp_dir: Directory for temporary files (default: system temp)
            max_file_size_mb: Maximum output file size in MB
            timeout_seconds: Maximum execution time for FFmpeg commands

        Raises:
            FileNotFoundError: If FFmpeg executables are not found
        """
        self.ffmpeg_path = self._validate_executable(ffmpeg_path)
        self.ffprobe_path = self._validate_executable(ffprobe_path)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.max_file_size_mb = max_file_size_mb
        self.timeout_seconds = timeout_seconds

        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"FFmpegHandler initialized with ffmpeg={self.ffmpeg_path}, "
            f"ffprobe={self.ffprobe_path}"
        )

    @staticmethod
    def _validate_executable(path: str) -> str:
        """
        Validate that an executable exists and is accessible.

        Args:
            path: Path to the executable

        Returns:
            Validated path string

        Raises:
            FileNotFoundError: If executable not found
        """
        # Check if it's a full path or just a command name
        if os.path.isabs(path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Executable not found: {path}")
        else:
            # Check if command is available in PATH
            if not shutil.which(path):
                raise FileNotFoundError(
                    f"Executable '{path}' not found in PATH. "
                    f"Please install FFmpeg or provide the full path."
                )
        
        return path

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal attacks.

        Args:
            filename: Input filename to sanitize

        Returns:
            Sanitized filename with only safe characters
        """
        # Remove any directory components
        filename = Path(filename).name
        
        # Remove any characters that aren't alphanumeric, dash, underscore, or dot
        sanitized = re.sub(r'[^\w\-.]', '_', filename)
        
        # Ensure filename isn't empty after sanitization
        if not sanitized:
            sanitized = "output.mp4"
        
        return sanitized

    def _validate_image_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that a file is a supported image format and exists.

        Args:
            file_path: Path to the image file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension is not supported
        """
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        if path.suffix.lower() not in self.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. "
                f"Allowed formats: {self.ALLOWED_IMAGE_EXTENSIONS}"
            )
        
        return path

    def _validate_audio_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that a file is a supported audio format and exists.

        Args:
            file_path: Path to the audio file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension is not supported
        """
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        if path.suffix.lower() not in self.ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{path.suffix}'. "
                f"Allowed formats: {self.ALLOWED_AUDIO_EXTENSIONS}"
            )
        
        return path

    async def _run_ffmpeg_command(
        self,
        args: List[str],
        description: str = "FFmpeg operation"
    ) -> Tuple[bool, str]:
        """
        Execute an FFmpeg command asynchronously.

        Args:
            args: List of command arguments (excluding executable)
            description: Human-readable description for logging

        Returns:
            Tuple of (success: bool, output/error message: str)

        Raises:
            FFmpegError: If command fails or times out
        """
        cmd = [self.ffmpeg_path] + args
        
        logger.info(f"Running {description}: {' '.join(cmd)}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds
                )
                
                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')
                
                if process.returncode != 0:
                    error_msg = (
                        f"{description} failed with return code {process.returncode}.\n"
                        f"STDERR: {stderr_str[:500]}"
                    )
                    logger.error(error_msg)
                    return False, error_msg
                
                logger.info(f"{description} completed successfully")
                return True, stdout_str or stderr_str[:200]
                
            except asyncio.TimeoutError:
                process.kill()
                error_msg = f"{description} timed out after {self.timeout_seconds} seconds"
                logger.error(error_msg)
                raise FFmpegError(error_msg)
                
        except FileNotFoundError as e:
            error_msg = f"FFmpeg executable not found: {e}"
            logger.error(error_msg)
            raise FFmpegError(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error during {description}: {str(e)}"
            logger.error(error_msg)
            raise FFmpegError(error_msg)

    async def get_media_duration(self, file_path: Union[str, Path]) -> float:
        """
        Get the duration of a media file using FFprobe.

        Args:
            file_path: Path to the media file

        Returns:
            Duration in seconds as float

        Raises:
            FFmpegError: If unable to get duration
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise FFmpegError(
                    f"Failed to get duration for {path}: {result.stderr}"
                )
            
            duration = float(result.stdout.strip())
            logger.info(f"Duration of {path.name}: {duration:.2f}s")
            
            return duration
            
        except subprocess.TimeoutExpired:
            raise FFmpegError("FFprobe command timed out")
        
        except ValueError as e:
            raise FFmpegError(f"Invalid duration value received: {e}")

    async def create_video_from_images(
        self,
        image_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        audio_path: Optional[Union[str, Path]] = None,
        image_duration: float = 5.0,
        transition_duration: float = 0.5,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 24,
        custom_args: Optional[List[str]] = None,
    ) -> Path:
        """
        Create a video from a list of images with optional audio.

        Uses FFmpeg's concat demuxer for smooth transitions between images.

        Args:
            image_paths: List of paths to input images (in order)
            output_path: Path for the output video file
            audio_path: Optional path to audio file for voiceover/music
            image_duration: Duration in seconds for each image (default: 5.0)
            transition_duration: Crossfade duration between images (default: 0.5)
            resolution: Output video resolution as (width, height) (default: 1920x1080)
            fps: Frames per second for output video (default: 24)
            custom_args: Additional FFmpeg arguments to append

        Returns:
            Path to the generated video file

        Raises:
            ValueError: If no images provided or invalid parameters
            FileNotFoundError: If input files don't exist
            FFmpegError: If video generation fails
        """
        
        # Validate inputs
        if not image_paths:
            raise ValueError("At least one image must be provided")
        
        if image_duration <= 0:
            raise ValueError("Image duration must be positive")
        
        if transition_duration < 0:
            raise ValueError("Transition duration cannot be negative")
        
        if len(resolution) != 2 or resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError("Resolution must be a tuple of positive integers (width, height)")
        
        # Validate and resolve all paths
        validated_images = []
        
        for img_path in image_paths:
            validated_img = self._validate_image_file(img_path)
            
            # Check file size (warn if too large)
            file_size_mb = validated_img.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 50:
                logger.warning(
                    f"Large image file ({file_size_mb:.1f} MB): {validated_img.name}"
                )
            
            validated_images.append(validated_img)
        
        # Sanitize output path
        output_dir = Path(output_path).parent.resolve()
        
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_output_name = self._sanitize_filename(Path(output_path).name)
        
        final_output_path = output_dir / safe_output_name
        
        # Validate audio if provided
        validated_audio = None
        
        if audio_path:
            validated_audio = self._validate_audio_file(audio_path)
            
            # Check audio duration against total video duration estimate
            try:
                audio_duration = await self.get_media_duration(validated_audio)
                
                total_image_duration = len(validated_images) * image_duration
                
                if audio_duration > total_image_duration + 10:
                    logger.warning(
                        f"Audio duration ({audio_duration:.1f}s) significantly exceeds "
                        f"total image duration ({total_image_duration:.1f}s)"
                    )
                    
                    # Adjust image duration to match audio length
                    adjusted_duration = audio_duration / len(validated_images)
                    
                    logger.info(
                        f"Adjusting image duration from {image_duration:.1f}s "
                        f"to {adjusted_duration:.1f}s to match audio"
                    )
                    
                    image_duration = adjusted_duration
                    
            except Exception as e:
                logger.warning(f"Could not check audio duration: {e}")
        
        # Create temporary concat file for images with durations