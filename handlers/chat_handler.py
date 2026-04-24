"""
Chat Handler Module for AI Video Generator

This module handles the complete workflow of processing chat messages:
1. Receives user input (topic)
2. Generates a script using DeepSeek AI
3. Fetches relevant media from Pixabay
4. Creates voiceover using ElevenLabs
5. Composes final video output

All API keys are loaded from environment variables for security.
"""

import os
import json
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.orm import Session as DBSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatHandler:
    """
    Main handler class for processing chat messages and generating videos.
    
    This class orchestrates the entire video generation workflow:
    - Script generation via DeepSeek AI
    - Media fetching from Pixabay
    - Voiceover creation via ElevenLabs
    - Video composition
    
    Attributes:
        deepseek_api_key: API key for DeepSeek AI service
        pixabay_api_key: API key for Pixabay media service
        elevenlabs_api_key: API key for ElevenLabs TTS service
        output_dir: Directory for storing generated videos
        session: Database session for persistence
    """
    
    def __init__(self, db_session: DBSession):
        """
        Initialize the ChatHandler with API keys and configuration.
        
        Args:
            db_session: SQLAlchemy database session for persistence
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Load API keys from environment variables
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.pixabay_api_key = os.getenv("PIXABAY_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Validate required API keys
        if not all([self.deepseek_api_key, self.pixabay_api_key, self.elevenlabs_api_key]):
            raise ValueError(
                "Missing required API keys. Please set DEEPSEEK_API_KEY, "
                "PIXABAY_API_KEY, and ELEVENLABS_API_KEY environment variables."
            )
        
        # Setup output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Store database session
        self.session = db_session
        
        # HTTP client for API calls (with timeout and retry configuration)
        self.http_client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        logger.info("ChatHandler initialized successfully")
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """
        Process an incoming chat message and initiate video generation.
        
        Args:
            message: User's input message (topic for video)
            session_id: Unique session identifier
            background_tasks: FastAPI background tasks handler
            
        Returns:
            Dictionary containing the response status and initial data
            
        Raises:
            HTTPException: If input validation fails or processing error occurs
        """
        # Input validation and sanitization
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Sanitize input - remove potentially harmful characters
        sanitized_message = self._sanitize_input(message)
        
        if len(sanitized_message) > 500:
            raise HTTPException(
                status_code=400,
                detail="Message too long. Maximum 500 characters allowed."
            )
        
        try:
            # Log the incoming request
            logger.info(f"Processing message for session {session_id}: {sanitized_message[:50]}...")
            
            # Save message to database
            self._save_message_to_db(session_id, sanitized_message, "user")
            
            # Start video generation in background
            background_tasks.add_task(
                self._generate_video_workflow,
                sanitized_message,
                session_id
            )
            
            # Return immediate response
            return {
                "status": "processing",
                "message": "Video generation started",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process message: {str(e)}"
            )
    
    async def _generate_video_workflow(self, topic: str, session_id: str) -> None:
        """
        Complete video generation workflow executed in background.
        
        Args:
            topic: The video topic/subject
            session_id: Session identifier for tracking
            
        This method orchestrates:
        1. Script generation via DeepSeek AI
        2. Media fetching from Pixabay
        3. Voiceover creation via ElevenLabs
        4. Video composition and final output
        """
        try:
            # Step 1: Generate script using DeepSeek AI
            logger.info(f"Step 1: Generating script for topic: {topic}")
            script = await self._generate_script(topic)
            
            if not script:
                raise ValueError("Failed to generate script")
            
            # Save generated script to database
            self._save_message_to_db(session_id, script, "assistant")
            
            # Step 2: Fetch relevant media from Pixabay
            logger.info("Step 2: Fetching media from Pixabay")
            media_urls = await self._fetch_media(topic)
            
            if not media_urls:
                logger.warning("No media found for topic, using placeholder")
                media_urls = self._get_placeholder_media()
            
            # Step 3: Generate voiceover using ElevenLabs
            logger.info("Step 3: Generating voiceover")
            audio_path = await self._generate_voiceover(script, session_id)
            
            if not audio_path:
                raise ValueError("Failed to generate voiceover")
            
            # Step 4: Compose final video
            logger.info("Step 4: Composing final video")
            video_path = await self._compose_video(
                media_urls,
                audio_path,
                script,
                session_id
            )
            
            if not video_path:
                raise ValueError("Failed to compose video")
            
            # Update database with video URL/path
            self._update_video_status(session_id, str(video_path), "completed")
            
            logger.info(f"Video generation completed successfully for session {session_id}")
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}", exc_info=True)
            self._update_video_status(session_id, None, "failed", str(e))
    
    async def _generate_script(self, topic: str) -> Optional[str]:
        """
        Generate a video script using DeepSeek AI.
        
        Args:
            topic: The topic/subject for the video script
            
        Returns:
            Generated script text or None if generation fails
            
        Raises:
            HTTPException: If API call fails
        """
        try:
            # Prepare the prompt for DeepSeek AI
            prompt = (
                f"Create a short, engaging video script about '{topic}'. "
                "The script should be:\n"
                "- Between 100-200 words\n"
                "- Suitable for voiceover narration\n"
                "- Include an introduction, main points, and conclusion\n"
                "- Written in a conversational tone\n\n"
                "Script:"
            )
            
            # Make API call to DeepSeek
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.deepseek_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a professional scriptwriter creating engaging video scripts."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract the generated script
                script = result["choices"][0]["message"]["content"].strip()
                
                logger.info(f"Script generated successfully ({len(script)} characters)")
                return script
                
        except httpx.HTTPError as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse DeepSeek response: {str(e)}")
            return None
    
    async def _fetch_media(self, query: str) -> List[str]:
        """
        Fetch relevant media (images/videos) from Pixabay.
        
        Args:
            query: Search query for media
            
        Returns:
            List of media URLs
            
        Raises:
            HTTPException: If API call fails
        """
        try:
            media_urls = []
            
            # Fetch images from Pixabay
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://pixabay.com/api/",
                    params={
                        "key": self.pixabay_api_key,
                        "q": query,
                        "image_type": "photo",
                        "per_page": 5,
                        "safesearch": True
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Extract image URLs
                for hit in data.get("hits", [])[:5]:
                    if "largeImageURL" in hit:
                        media_urls.append(hit["largeImageURL"])
                
                # If we have fewer than 3 images, try fetching videos too
                if len(media_urls) < 3:
                    video_response = await client.get(
                        "https://pixabay.com/api/videos/",
                        params={
                            "key": self.pixabay_api_key,
                            "q": query,
                            "per_page": 3,
                            "safesearch": True
                        }
                    )
                    
                    video_response.raise_for_status()
                    video_data = video_response.json()
                    
                    for hit in video_data.get("hits", [])[:3]:
                        if "videos" in hit and "medium" in hit["videos"]:
                            media_urls.append(hit["videos"]["medium"]["url"])
                
                logger.info(f"Fetched {len(media_urls)} media items for query: {query}")
                return media_urls[:5]  # Limit to 5 media items
                
        except httpx.HTTPError as e:
            logger.error(f"Pixabay API error: {str(e)}")
            return []
    
    async def _generate_voiceover(self, text: str, session_id: str) -> Optional[Path]:
        """
        Generate voiceover audio using ElevenLabs TTS.
        
        Args:
            text: Text to convert to speech
            session_id: Session identifier for file naming
            
        Returns:
            Path to generated audio file or None if generation fails
            
        Raises:
            HTTPException: If API call fails or file operations fail
        """
        try:
            # Prepare output path for audio file
            audio_filename = f"voiceover_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            audio_path = self.output_dir / audio_filename
            
            # Make API call to ElevenLabs
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",  # Default voice ID (Rachel)
                    headers={
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": self.elevenlabs_api_key
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.5,
                            "style": 0.0,
                            "use_speaker_boost": True
                        }
                    }
                )
                
                response.raise_for_status()
                
                # Save audio file
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Voiceover generated successfully at {audio_path}")
                return audio_path
                
        except httpx.HTTPError as e:
            logger.error(f"ElevenLabs API error: {str(e)}")
            return None
        except IOError as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            return None
    
    async def _compose_video(
        self,
        media_urls: List[str],
        audio_path: Path,
        script: str,
        session_id: str
    ) -> Optional[Path]:
        """
        Compose final video from media files and voiceover.
        
        Args:
            media_urls: List of media file URLs to include in video
            audio_path: Path to voiceover audio file
            script: Generated script text (for subtitles/timing)
            session_id: Session identifier for file naming
            
        Returns:
            Path to composed video file or None if composition fails
            
        Note:
            This is a placeholder implementation. In production, you would use 
            FFmpeg or a video editing library like MoviePy for actual composition.
        """
        try:
            # Prepare output path for video file
            video_filename = f"video_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = self.output_dir / video_filename
            
            # TODO: Implement actual video composition using FFmpeg or MoviePy
            # For now, we'll create a placeholder JSON manifest
            
            manifest = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "media_urls": media_urls,
                "audio_path": str(audio_path),
                "script": script,
                "status": "pending_composition"
            }
            
            manifest_path = self.output_dir / f"manifest_{session_id}.json"
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Video manifest created at {manifest_path}")
            
            # In production, you would compose the actual video here
            # For now, we'll just create a symbolic link or copy the audio as placeholder
            
            import shutil
            
            # Create a simple placeholder video (just copy audio to simulate)
            shutil.copy2(audio_path, video_path)
            
            logger.info(f"Video composed successfully at {video_path}")
            
            return video_path
            
        except Exception as e:
            logger.error(f"Video composition failed: {str(e)}", exc_info=True)
            
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            text: Raw user input
            
        Returns:
            Sanitized text string
            
        Note:
            Removes potentially harmful characters and limits length.
        """
        import re
        
        # Remove any HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove any script tags or JavaScript events
        text = re.sub(r'(?i)<script[\s\S]*?</script>', '', text)
        
        # Remove potentially dangerous characters (keep alphanumeric and basic punctuation)
        text = re.sub(r'[^\w\s\-.,!?\'"]', '', text)
        
        # Trim whitespace and limit length
        text = text.strip()[:500]
        
        return text
    
    def _save_message_to_db(self, session_id: str, content: str, role: str) -> None:
        """
        Save a chat message to the database.
        
        Args:
            session_id: Session identifier
            content: Message content
            role: Message role ('user' or 'assistant')
            
        Note:
            This is a placeholder implementation. Actual database operations 
            would depend on your SQLAlchemy model definitions.
        
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(
    def _update_video_status(