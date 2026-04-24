"""
ElevenLabs Voiceover Service

This module provides a service to generate voiceover audio using the ElevenLabs API.
It handles text-to-speech conversion with configurable voice settings and proper
error handling for production use.

Typical usage:
    service = ElevenLabsService(api_key="your_key")
    audio_content = service.generate_voiceover("Hello world", voice_id="21m00Tcm4TlvDq8ikWAM")
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure logging
logger = logging.getLogger(__name__)


class ElevenLabsError(Exception):
    """Custom exception for ElevenLabs service errors."""
    pass


class ElevenLabsService:
    """
    Service class for interacting with the ElevenLabs Text-to-Speech API.

    Attributes:
        api_key (str): The ElevenLabs API key.
        base_url (str): Base URL for the ElevenLabs API.
        default_voice_id (str): Default voice ID to use when none specified.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retry attempts for failed requests.
    """

    BASE_URL = "https://api.elevenlabs.io/v1"
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    DEFAULT_MODEL = "eleven_monolingual_v1"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the ElevenLabs service.

        Args:
            api_key: ElevenLabs API key. If None, loads from ELEVENLABS_API_KEY env var.
            voice_id: Default voice ID to use. If None, uses DEFAULT_VOICE_ID.
            model: Model to use for generation. If None, uses DEFAULT_MODEL.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.

        Raises:
            ElevenLabsError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ElevenLabsError(
                "ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.voice_id = voice_id or self.DEFAULT_VOICE_ID
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

        # Validate voice ID format (basic check)
        if not self.voice_id or len(self.voice_id) < 10:
            logger.warning(f"Voice ID '{self.voice_id}' appears invalid. Using default.")
            self.voice_id = self.DEFAULT_VOICE_ID

        logger.info(
            f"ElevenLabsService initialized with voice_id={self.voice_id}, "
            f"model={self.model}"
        )

    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests.

        Returns:
            Dictionary of headers including authorization.
        """
        return {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }

    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        method: str = "POST",
    ) -> bytes:
        """
        Make an HTTP request to the ElevenLabs API with retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/text-to-speech/{voice_id}").
            payload: JSON payload for the request.
            method: HTTP method (default: POST).

        Returns:
            Raw audio content as bytes.

        Raises:
            ElevenLabsError: If the request fails after all retries.
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Request attempt {attempt}/{self.max_retries} to {url}")
                
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                
                response.raise_for_status()
                
                # Check content type to ensure we got audio
                content_type = response.headers.get("Content-Type", "")
                if "audio" not in content_type and "octet-stream" not in content_type:
                    logger.warning(
                        f"Unexpected content type: {content_type}. "
                        f"Response: {response.text[:200]}"
                    )
                
                logger.info(
                    f"Successfully generated audio ({len(response.content)} bytes)"
                )
                return response.content

            except Timeout as e:
                logger.warning(f"Request timeout (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    raise ElevenLabsError(
                        f"Request timed out after {self.max_retries} attempts"
                    )

            except ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    raise ElevenLabsError(
                        f"Connection failed after {self.max_retries} attempts: {e}"
                    )

            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                error_body = e.response.text[:500] if e.response else ""
                
                if status_code == 401:
                    raise ElevenLabsError(
                        "Authentication failed. Check your ElevenLabs API key."
                    )
                elif status_code == 422:
                    raise ElevenLabsError(
                        f"Invalid request parameters: {error_body}"
                    )
                elif status_code == 429:
                    logger.warning(f"Rate limited (attempt {attempt}). Retrying...")
                    if attempt == self.max_retries:
                        raise ElevenLabsError(
                            "Rate limit exceeded. Please try again later."
                        )
                else:
                    raise ElevenLabsError(
                        f"HTTP {status_code} error: {error_body}"
                    )

            except RequestException as e:
                logger.error(f"Request failed (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    raise ElevenLabsError(f"Request failed: {e}")

        # This should not be reached, but just in case
        raise ElevenLabsError("Unexpected error in request logic")

    def generate_voiceover(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        speaker_boost: bool = False,
    ) -> bytes:
        """
        Generate voiceover audio from text using ElevenLabs TTS.

        Args:
            text: The text to convert to speech (max 5000 characters).
            voice_id: Voice ID to use. If None, uses the default voice.
            model: Model to use for generation. If None, uses the default model.
            stability: Voice stability (0.0 to 1.0). Higher values are more stable.
            similarity_boost: Voice similarity boost (0.0 to 1.0).
            style: Style exaggeration (0.0 to 1.0).
            speaker_boost: Whether to boost the speaker's presence.

        Returns:
            Raw MP3 audio content as bytes.

        Raises:
            ElevenLabsError: If text is invalid or generation fails.
            ValueError: If parameters are out of valid ranges.
        """
        # Validate input text
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Strip and limit text length
        text = text.strip()
        
        # Check for empty text after stripping
        if not text:
            raise ValueError("Text cannot be empty after stripping whitespace")
        
        # Enforce character limit (ElevenLabs allows up to 5000)
        max_chars = 5000
        if len(text) > max_chars:
            logger.warning(
                f"Text exceeds {max_chars} characters ({len(text)}). Truncating."
            )
            text = text[:max_chars]
        
        # Validate numeric parameters
        for param_name, param_value, min_val, max_val in [
            ("stability", stability, 0.0, 1.0),
            ("similarity_boost", similarity_boost, 0.0, 1.0),
            ("style", style, 0.0, 1.0),
        ]:
            if not min_val <= param_value <= max_val:
                raise ValueError(
                    f"{param_name} must be between {min_val} and {max_val}, "
                    f"got {param_value}"
                )
        
        # Use provided values or defaults
        active_voice_id = voice_id or self.voice_id
        active_model = model or self.model
        
        # Build the endpoint URL
        endpoint = f"/text-to-speech/{active_voice_id}"
        
        # Build the request payload
        payload = {
            "text": text,
            "model_id": active_model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": speaker_boost,
            },
        }
        
        logger.info(
            f"Generating voiceover for text ({len(text)} chars) "
            f"with voice={active_voice_id}, model={active_model}"
        )
        
        # Make the API request
        audio_content = self._make_request(endpoint, payload)
        
        return audio_content

    def save_audio_to_file(
        self,
        audio_content: bytes,
        filepath: Union[str, Path],
    ) -> Path:
        """
        Save audio content to a file.

        Args:
            audio_content: Raw audio bytes to save.
            filepath: Path where to save the audio file.

        Returns:
            Path object pointing to the saved file.

        Raises:
            ElevenLabsError: If saving fails.
            ValueError: If audio_content is empty.
        """
        if not audio_content:
            raise ValueError("Audio content is empty")
        
        filepath = Path(filepath)
        
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write audio file
            with open(filepath, "wb") as f:
                f.write(audio_content)
            
            logger.info(f"Audio saved to {filepath} ({len(audio_content)} bytes)")
            
            return filepath
            
        except OSError as e:
            raise ElevenLabsError(f"Failed to save audio file: {e}")

    def list_voices(self) -> Dict[str, Any]:
        """
         Fetch available voices from ElevenLabs.

         Returns:
             Dictionary containing voice information.

         Raises:
             ElevenLabsError: If the API request fails.
         """
         endpoint = "/voices"
         
         try:
             response = requests.get(
                 f"{self.BASE_URL}{endpoint}",
                 headers={"xi-api-key": self.api_key},
                 timeout=self.timeout,
             )
             response.raise_for_status()
             
             voices_data = response.json()
             logger.info(f"Retrieved {len(voices_data.get('voices', []))} voices")
             
             return voices_data
             
         except RequestException as e:
             raise ElevenLabsError(f"Failed to fetch voices: {e}")


# Convenience function for quick use
def generate_speech(
    text: str,
    api_key: Optional[str] = None,
    voice_id: Optional[str] = None,
    **kwargs,
) -> bytes:
    """
    Quick function to generate speech without instantiating the service class.

    Args:
         text: Text to convert to speech.
         api_key: ElevenLabs API key (or use ELEVENLABS_API_KEY env var).
         voice_id: Voice ID to use.
         **kwargs: Additional parameters passed to generate_voiceover().

     Returns:
         Raw MP3 audio bytes.
     """
     service = ElevenLabsService(api_key=api_key, voice_id=voice_id)
     return service.generate_voiceover(text, **kwargs)


# Example usage (commented out)
if __name__ == "__main__":
     # This block runs when the script is executed directly
     import sys
    
     # Example usage
     try:
         service = ElevenLabsService()
         audio = service.generate_voiceover(
             "Hello! This is a test of the ElevenLabs text-to-speech service.",
             stability=0.7,
             similarity_boost=0.8,
         )
         service.save_audio_to_file(audio, "output/test_voiceover.mp3")
         print("Voiceover generated successfully!")
         
     except (ElevenLabsError, ValueError) as e:
         print(f"Error: {e}", file=sys.stderr)
         sys.exit(1)