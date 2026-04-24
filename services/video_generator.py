```py
"""
Video Generator Service

This module provides functionality to combine media files (images/videos) with
voiceover audio into a final video using FFmpeg. It handles various media formats,
timing, transitions, and ensures proper synchronization between audio and visuals.

The service is designed to work with the AI Video Generator app's pipeline:
1. Script generation (DeepSeek AI)
2. Media fetching (Pixabay)
3. Voiceover creation (ElevenLabs)
4. Video composition (this module)
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoGeneratorError(Exception):
    """Base exception for video generation errors."""
    pass


class FFmpegNotFoundError(VideoGeneratorError):
    """Raised when FFmpeg is not found in the system."""
    pass


class MediaProcessingError(VideoGeneratorError):
    """Raised when there's an error processing media files."""
    pass


class VideoCompositionError(VideoGeneratorError):
    """Raised when video composition fails."""
    pass


class VideoGenerator:
    """
    Service class for combining media and voiceover into a final video.
    
    This class handles the entire video generation pipeline including:
    - Validating input media and audio files
    - Processing different media formats (images, videos)
    - Synchronizing audio with visual content
    - Applying transitions and effects
    - Generating the final video file
    
    Attributes:
        output_dir: Directory for storing generated videos
        temp_dir: Directory for temporary files during processing
        ffmpeg_path: Path to FFmpeg executable
        ffprobe_path: Path to FFprobe executable
        max_video_duration: Maximum allowed video duration in seconds
        supported_image_formats: List of supported image formats
        supported_video_formats: List of supported video formats
        supported_audio_formats: List of supported audio formats
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "output/videos",
        temp_dir: Union[str, Path] = "temp",
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        max_video_duration: int = 300  # 5 minutes default
    ):
        """
        Initialize the VideoGenerator service.
        
        Args:
            output_dir: Directory for storing generated videos
            temp_dir: Directory for temporary files
            ffmpeg_path: Custom path to FFmpeg executable
            ffprobe_path: Custom path to FFprobe executable
            max_video_duration: Maximum allowed video duration in seconds
            
        Raises:
            FFmpegNotFoundError: If FFmpeg is not found in the system
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.max_video_duration = max_video_duration
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set FFmpeg paths
        self.ffmpeg_path = ffmpeg_path or self._find_executable("ffmpeg")
        self.ffprobe_path = ffprobe_path or self._find_executable("ffprobe")
        
        # Validate FFmpeg availability
        if not self._check_ffmpeg():
            raise FFmpegNotFoundError(
                "FFmpeg is not installed or not found in system PATH. "
                "Please install FFmpeg and ensure it's accessible."
            )
        
        # Supported formats
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        self.supported_video_formats = {'.mp4', '.webm', '.avi', '.mov', '.mkv'}
        self.supported_audio_formats = {'.mp3', '.wav', '.ogg', '.aac', '.m4a'}
        
        logger.info(
            f"VideoGenerator initialized. Output: {self.output_dir}, "
            f"FFmpeg: {self.ffmpeg_path}"
        )
    
    @staticmethod
    def _find_executable(name: str) -> str:
        """
        Find an executable in the system PATH.
        
        Args:
            name: Name of the executable to find
            
        Returns:
            Path to the executable or the name itself if not found
        """
        import shutil
        path = shutil.which(name)
        return path or name
    
    def _check_ffmpeg(self) -> bool:
        """
        Check if FFmpeg is available and working.
        
        Returns:
            True if FFmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _get_media_duration(self, file_path: Union[str, Path]) -> float:
        """
        Get the duration of a media file using FFprobe.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            Duration in seconds
            
        Raises:
            MediaProcessingError: If unable to get duration
        """
        try:
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise MediaProcessingError(
                    f"Failed to get duration for {file_path}: {result.stderr}"
                )
            
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            
            return duration
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise MediaProcessingError(
                f"Failed to parse duration for {file_path}: {str(e)}"
            )
    
    def _get_video_dimensions(self, file_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get the dimensions of a video file.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            MediaProcessingError: If unable to get dimensions
        """
        try:
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise MediaProcessingError(
                    f"Failed to get dimensions for {file_path}: {result.stderr}"
                )
            
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            width = int(stream['width'])
            height = int(stream['height'])
            
            return (width, height)
            
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            raise MediaProcessingError(
                f"Failed to parse dimensions for {file_path}: {str(e)}"
            )
    
    def _validate_media_file(self, file_path: Union[str, Path]) -> str:
        """
        Validate a media file exists and has a supported format.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            File extension in lowercase
            
        Raises:
            MediaProcessingError: If file is invalid or unsupported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise MediaProcessingError(f"Media file not found: {file_path}")
        
        if not path.is_file():
            raise MediaProcessingError(f"Path is not a file: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_image_formats | self.supported_video_formats:
            raise MediaProcessingError(
                f"Unsupported media format: {extension}. "
                f"Supported formats: {self.supported_image_formats | self.supported_video_formats}"
            )
        
        return extension
    
    def _validate_audio_file(self, file_path: Union[str, Path]) -> None:
        """
        Validate an audio file exists and has a supported format.
        
        Args:
            file_path: Path to the audio file
            
        Raises:
            MediaProcessingError: If file is invalid or unsupported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise MediaProcessingError(f"Audio file not found: {file_path}")
        
        if not path.is_file():
            raise MediaProcessingError(f"Path is not a file: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_audio_formats:
            raise MediaProcessingError(
                f"Unsupported audio format: {extension}. "
                f"Supported formats: {self.supported_audio_formats}"
            )
    
    def _create_image_video(
        self,
        image_path: Union[str, Path],
        duration: float,
        output_path: Union[str, Path],
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30
    ) -> None:
        """
        Create a video from a single image with specified duration.
        
        Args:
            image_path: Path to the input image
            duration: Duration of the output video in seconds
            output_path: Path for the output video
            resolution: Output video resolution (width, height)
            fps: Frames per second for output video
            
        Raises:
            MediaProcessingError: If video creation fails
        """
        try:
            # Add zoom effect for more dynamic presentation
            cmd = [
                self.ffmpeg_path,
                "-loop", "1",
                "-i", str(image_path),
                "-c:v", "libx264",
                "-t", str(duration),
                "-pix_fmt", "yuv420p",
                "-vf", (
                    f"scale={resolution[0]}:{resolution[1]}:force_original_aspect_ratio=decrease,"
                    f"pad={resolution[0]}:{resolution[1]}:(ow-iw)/2:(oh-ih)/2,"
                    f"zoompan=z='if(lte(zoom,1.0),1.5,zoom+0.0025)':d={int(fps * duration)}"
                ),
                "-r", str(fps),
                "-y",
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout for large images
            )
            
            if result.returncode != 0:
                raise MediaProcessingError(
                    f"Failed to create video from image {image_path}: {result.stderr}"
                )
                
            logger.debug(f"Created video from image: {output_path}")
            
        except subprocess.TimeoutExpired:
            raise MediaProcessingError(
                f"Timeout while creating video from image {image_path}"
            )
    
    def _concatenate_videos(
        self,
        video_files: List[Union[str, Path]],
        output_path: Union[str, Path],
    ) -> None:
        """
        Concatenate multiple video files into one.
        
        Args:
            video_files: List of video file paths to concatenate
            output_path: Path for the concatenated output video
            
        Raises:
            VideoCompositionError: If concatenation fails
        """
        if not video_files:
            raise VideoCompositionError("No video files to concatenate")
        
        # Create a temporary file list for FFmpeg concat demuxer
        temp_file_list = None
        
        try:
            # Create temporary file with list of videos
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                delete=False,
                dir=self.temp_dir
            ) as f:
                temp_file_list = f.name
                
                for video_file in video_files:
                    # Ensure absolute path for FFmpeg compatibility
                    abs_path = Path(video_file).resolve()
                    f.write(f"file '{abs_path}'\n")
            
            # Concatenate videos using FFmpeg concat demuxer
            cmd = [
                self.ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", temp_file_list,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                "-y",
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for large videos
            )
            
            if result.returncode != 0:
                raise VideoCompositionError(
                    f"Failed to concatenate videos: {result.stderr}"
                )
                
            logger.debug(f"Concatenated {len(video_files)} videos into {output_path}")
            
        except subprocess.TimeoutExpired:
            raise VideoCompositionError("Timeout while concatenating videos")
            
        finally:
            # Clean up temporary file list
            if temp_file_list and os.path.exists(temp_file_list):
                os.unlink(temp_file_list)
    
    def _add_audio_to_video(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """
        Add audio track to a video file.
        
        Args:
            video_path: Path to the input video (without audio)
            audio_path: Path to the audio file to add
            output_path: Path for the output video with audio
            
        Raises:
            VideoCompositionError: If adding audio fails
        """
        try:
            # Get durations for synchronization
            video_duration = self._get_media_duration(video_path)
            
            cmd = [
                self.ffmpeg_path,
                "-i", str(video_path),
                "-i", str(audio_path),
                
                # Map streams correctly
                "-map", "0:v:0",  # Video from first input
                
                # Handle audio based on duration comparison
                *(["-map", "1:a:0"] if audio_path else []),  # Audio from second input if exists
                
                # Audio encoding settings
                "-c:v", "libx264",