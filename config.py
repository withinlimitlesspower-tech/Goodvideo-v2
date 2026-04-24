"""
Configuration module for the AI Video Generator application.

Loads environment variables and provides configuration constants for:
- DeepSeek AI API (script generation)
- Pixabay API (media fetching)
- ElevenLabs API (voiceover generation)
- Application settings (database, server, etc.)

All sensitive data is loaded from environment variables with proper validation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


def load_environment_variables() -> None:
    """
    Load environment variables from .env file if it exists.
    
    Searches for .env file in the project root directory and loads
    it into the environment. This allows for local development without
    setting system-wide environment variables.
    """
    # Determine project root directory (parent of this config file)
    project_root = Path(__file__).resolve().parent
    
    # Look for .env file in project root
    env_path = project_root / '.env'
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("No .env file found. Using system environment variables.")


def get_env_variable(var_name: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable with validation.
    
    Args:
        var_name: Name of the environment variable
        required: If True, raises ValueError when variable is missing
        default: Default value if variable is not set and not required
    
    Returns:
        The value of the environment variable or default
    
    Raises:
        ValueError: If required variable is missing and no default provided
    """
    value = os.environ.get(var_name)
    
    if value is None or value.strip() == '':
        if required:
            raise ValueError(
                f"Required environment variable '{var_name}' is not set. "
                f"Please set it in your .env file or system environment."
            )
        return default
    
    return value.strip()


# Load environment variables on module import
load_environment_variables()


class Config:
    """
    Application configuration loaded from environment variables.
    
    Provides typed access to all configuration constants with validation.
    """
    
    # --------------------------------------------------------------------------
    # API Keys (Required - application will not function without these)
    # --------------------------------------------------------------------------
    
    DEEPSEEK_API_KEY: str = get_env_variable("DEEPSEEK_API_KEY", required=True)
    """API key for DeepSeek AI text generation service."""
    
    PIXABAY_API_KEY: str = get_env_variable("PIXABAY_API_KEY", required=True)
    """API key for Pixabay media search service."""
    
    ELEVENLABS_API_KEY: str = get_env_variable("ELEVENLABS_API_KEY", required=True)
    """API key for ElevenLabs text-to-speech service."""
    
    # --------------------------------------------------------------------------
    # API Endpoints
    # --------------------------------------------------------------------------
    
    DEEPSEEK_API_URL: str = get_env_variable(
        "DEEPSEEK_API_URL",
        required=False,
        default="https://api.deepseek.com/v1/chat/completions"
    )
    """Base URL for DeepSeek AI API."""
    
    PIXABAY_API_URL: str = get_env_variable(
        "PIXABAY_API_URL",
        required=False,
        default="https://pixabay.com/api/"
    )
    """Base URL for Pixabay API."""
    
    ELEVENLABS_API_URL: str = get_env_variable(
        "ELEVENLABS_API_URL",
        required=False,
        default="https://api.elevenlabs.io/v1/text-to-speech"
    )
    """Base URL for ElevenLabs API."""
    
    # --------------------------------------------------------------------------
    # Application Settings
    # --------------------------------------------------------------------------
    
    DATABASE_URL: str = get_env_variable(
        "DATABASE_URL",
        required=False,
        default="sqlite:///./video_generator.db"
    )
    """Database connection string. Defaults to local SQLite file."""
    
    DATABASE_PATH: str = get_env_variable(
        "DATABASE_PATH",
        required=False,
        default="./video_generator.db"
    )
    """Path to SQLite database file (used when DATABASE_URL is SQLite)."""
    
    SERVER_HOST: str = get_env_variable(
        "SERVER_HOST",
        required=False,
        default="0.0.0.0"
    )
    """Host address for the FastAPI server."""
    
    SERVER_PORT: int = int(get_env_variable(
        "SERVER_PORT",
        required=False,
        default="8000"
    ))
    """Port number for the FastAPI server."""
    
    DEBUG_MODE: bool = get_env_variable(
        "DEBUG_MODE",
        required=False,
        default="False"
    ).lower() in ("true", "1", "yes")
    """Enable debug mode for detailed error messages and logging."""
    
    LOG_LEVEL: str = get_env_variable(
        "LOG_LEVEL",
        required=False,
        default="INFO"
    )
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
    
    MAX_SESSION_AGE_HOURS: int = int(get_env_variable(
        "MAX_SESSION_AGE_HOURS",
        required=False,
        default="24"
    ))
    """Maximum session age in hours before cleanup."""
    
    MAX_MEDIA_PER_TOPIC: int = int(get_env_variable(
        "MAX_MEDIA_PER_TOPIC",
        required=False,
        default="10"
    ))
    """Maximum number of media items to fetch per topic."""
    
    MAX_SCRIPT_LENGTH_CHARS: int = int(get_env_variable(
        "MAX_SCRIPT_LENGTH_CHARS",
        required=False,
        default="2000"
    ))
    """Maximum character length for generated scripts."""
    
    # --------------------------------------------------------------------------
    # ElevenLabs Voice Settings
    # --------------------------------------------------------------------------
    
    ELEVENLABS_VOICE_ID: str = get_env_variable(
        "ELEVENLABS_VOICE_ID",
        required=False,
        default="21m00Tcm4TlvDq8ikWAM"  # Default Rachel voice
    )
    """Voice ID to use for ElevenLabs text-to-speech."""
    
    ELEVENLABS_MODEL_ID: str = get_env_variable(
        "ELEVENLABS_MODEL_ID",
        required=False,
        default="eleven_monolingual_v1"
    )
    """Model ID for ElevenLabs text-to-speech."""
    
    ELEVENLABS_STABILITY: float = float(get_env_variable(
        "ELEVENLABS_STABILITY",
        required=False,
        default="0.5"
    ))
    """Voice stability parameter (0.0 to 1.0)."""
    
    ELEVENLABS_SIMILARITY_BOOST: float = float(get_env_variable(
        "ELEVENLABS_SIMILARITY_BOOST",
        required=False,
        default="0.75"
    ))
    """Voice similarity boost parameter (0.0 to 1.0)."""
    
    # --------------------------------------------------------------------------
    # DeepSeek Model Settings
    # --------------------------------------------------------------------------
    
    DEEPSEEK_MODEL: str = get_env_variable(
        "DEEPSEEK_MODEL",
        required=False,
        default="deepseek-chat"
    )
    """Model name for DeepSeek AI."""
    
    DEEPSEEK_TEMPERATURE: float = float(get_env_variable(
        "DEEPSEEK_TEMPERATURE",
        required=False,
        default="0.7"
    ))
    """Temperature for text generation (0.0 to 2.0)."""
    
    DEEPSEEK_MAX_TOKENS: int = int(get_env_variable(
        "DEEPSEEK_MAX_TOKENS",
        required=False,
        default="2048"
    ))
    """Maximum tokens for DeepSeek response."""
    
    # --------------------------------------------------------------------------
    # Pixabay Search Settings
    # --------------------------------------------------------------------------
    
    PIXABAY_SAFE_SEARCH: bool = get_env_variable(
        "PIXABAY_SAFE_SEARCH",
        required=False,
        default="True"
    ).lower() in ("true", "1", "yes")
    """Enable safe search for Pixabay media."""
    
    PIXABAY_IMAGE_TYPE: str = get_env_variable(
        "PIXABAY_IMAGE_TYPE",
        required=False,
        default="photo"
    )
    """Type of images to fetch (photo, illustration, vector)."""
    
    PIXABAY_VIDEO_TYPE: str = get_env_variable(
        "PIXABAY_VIDEO_TYPE",
        required=False,
        default="film"
    )
    """Type of videos to fetch (film, animation)."""
    
    PIXABAY_MIN_WIDTH: int = int(get_env_variable(
        "PIXABAY_MIN_WIDTH",
        required=False,
        default="1920"
    ))
    """Minimum width for fetched media in pixels."""
    
    PIXABAY_MIN_HEIGHT: int = int(get_env_variable(
        "PIXABAY_MIN_HEIGHT",
        required=False,
        default="1080"
    ))
    """Minimum height for fetched media in pixels."""
    
    # --------------------------------------------------------------------------
    # Output Settings
    # --------------------------------------------------------------------------
    
    OUTPUT_DIRECTORY: str = get_env_variable(
        "OUTPUT_DIRECTORY",
        required=False,
        default="./output"
    )
    """Directory where generated videos are stored."""
    
    TEMP_DIRECTORY: str = get_env_variable(
        "TEMP_DIRECTORY",
        required=False,
        default="./temp"
    )
    """Directory for temporary files during video generation."""
    
    VIDEO_FPS: int = int(get_env_variable(
        "VIDEO_FPS",
        required=False,
        default="24"
    ))
    """Frames per second for output video."""
    
    VIDEO_RESOLUTION_WIDTH: int = int(get_env_variable(
        "VIDEO_RESOLUTION_WIDTH",
        required=False,
        default="1920"
    ))
    """Output video width in pixels."""
    
    VIDEO_RESOLUTION_HEIGHT: int = int(get_env_variable(
        "VIDEO_RESOLUTION_HEIGHT",
        required=False,
        default="1080"
    ))
    """Output video height in pixels."""
    
    VIDEO_CODEC: str = get_env_variable(
        "VIDEO_CODEC",
        required=False,
        default="libx264"
    )
    """Video codec for encoding."""
    
    AUDIO_CODEC: str = get_env_variable(
        "VIDEO_AUDIO_CODEC",
        required=False,
        default="aac"
    )
    
    
def validate_config() -> Dict[str, Any]:
   
   
   """
   Validate that all critical configuration values are properly set.
   
   Returns:
       Dictionary with validation results containing 'valid' boolean
       and 'errors' list of error messages.
   """
   errors = []
   
   # Check critical API keys
   if not Config.DEEPSEEK_API_KEY:
       errors.append("DEEPSEEK_API_KEY is missing or empty")
   
   if not Config.PIXABAY_API_KEY:
       errors.append("PIXABAY_API_KEY is missing or empty")
   
   if not Config.ELEVENLABS_API_KEY:
       errors.append("ELEVENLABS_API_KEY is missing or empty")
   
   # Validate numeric ranges
   if not (0 <= Config.ELEVENLABS_STABILITY <= 1):
       errors.append("ELEVENLABS_STABILITY must be between 0 and 1")
   
   if not (0 <= Config.ELEVENLABS_SIMILARITY_BOOST <= 1):
       errors.append("ELEVENLABS_SIMILARITY_BOOST must be between 0 and 1")
   
   if not (0 <= Config.DEEPSEEK_TEMPERATURE <= 2):
       errors.append("DEEPSEEK_TEMPERATURE must be between 0 and 2")
   
   if Config.MAX_MEDIA_PER_TOPIC < 1:
       errors.append("MAX_MEDIA_PER_TOPIC must be at least 1")
   
   if Config.MAX_SCRIPT_LENGTH_CHARS < 100:
       errors.append("MAX_SCRIPT_LENGTH_CHARS must be at least 100")
   
   if Config.SERVER_PORT < 1024 or Config.SERVER_PORT > 65535:
       errors.append("SERVER_PORT must be between 1024 and 65535")
   
   # Validate directory paths exist or can be created
   for dir_path in [Config.OUTPUT_DIRECTORY, Config.TEMP_DIRECTORY]:
       path = Path(dir_path)
       try:
           path.mkdir(parents=True, exist_ok=True)
       except PermissionError:
           errors.append(f"Cannot create directory '{dir_path}': Permission denied")
       except OSError as e:
           errors.append(f"Cannot create directory '{dir_path}': {e}")
   
   return {
       "valid": len(errors) == 0,
       "errors": errors
   }


# Perform validation on module import and warn about issues
_config_validation = validate_config()
if not _config_validation["valid"]:
   import warnings
   warnings.warn(
       f"Configuration validation failed with {len(_config_validation['errors'])} error(s):\n"
       + "\n".join(f"  - {err}" for err in _config_validation["errors"])
   )


# Export commonly used constants at module level for convenience
DEEPSEEK_API_KEY = Config.DEEPSEEK_API_KEY
PIXABAY_API_KEY = Config.PIXABAY_API_KEY
ELEVENLABS_API_KEY = Config.ELEVENLABS_API_KEY

DATABASE_URL = Config.DATABASE_URL
SERVER_HOST = Config.SERVER_HOST
SERVER_PORT = Config.SERVER_PORT
DEBUG_MODE = Config.DEBUG_MODE

__all__ = [
   'Config',
   'validate_config',
   'load_environment_variables',
   'get_env_variable',
   'DEEPSEEK_API_KEY',
   'PIXABAY_API_KEY',
   'ELEVENLABS_API_KEY',
   'DATABASE_URL',
   'SERVER_HOST',
   'SERVER_PORT',
   'DEBUG_MODE',
]