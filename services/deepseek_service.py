"""
DeepSeek AI Service Module

This module provides a service class for interacting with DeepSeek AI to generate
video scripts. It handles API communication, response parsing, error handling,
and provides structured script generation for the AI Video Generator application.

The service supports:
- Script generation from user topics
- Structured script output with sections
- Error handling and retry logic
- Input validation and sanitization
- Configurable parameters (temperature, max tokens, etc.)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import httpx
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logger = logging.getLogger(__name__)


class ScriptSection(BaseModel):
    """Represents a section of the generated video script."""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)
    duration_seconds: int = Field(default=10, ge=5, le=120)
    suggested_keywords: List[str] = Field(default_factory=list)


class GeneratedScript(BaseModel):
    """Complete generated script with metadata."""
    topic: str = Field(..., min_length=1, max_length=500)
    title: str = Field(..., min_length=1, max_length=300)
    sections: List[ScriptSection] = Field(..., min_items=1, max_items=20)
    total_duration_seconds: int = Field(..., ge=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="deepseek-chat")
    raw_response: Optional[str] = Field(default=None)


class DeepSeekConfig(BaseModel):
    """Configuration for DeepSeek API client."""
    api_key: str = Field(..., min_length=1)
    base_url: str = Field(default="https://api.deepseek.com/v1")
    model: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=100, le=8192)
    timeout_seconds: int = Field(default=30, ge=5)
    max_retries: int = Field(default=3, ge=0)


class DeepSeekServiceError(Exception):
    """Base exception for DeepSeek service errors."""
    pass


class APIKeyError(DeepSeekServiceError):
    """Raised when API key is missing or invalid."""
    pass


class APIRequestError(DeepSeekServiceError):
    """Raised when API request fails."""
    pass


class ScriptGenerationError(DeepSeekServiceError):
    """Raised when script generation fails."""
    pass


class DeepSeekService:
    """
    Service class for interacting with DeepSeek AI to generate video scripts.
    
    This service handles:
    - API authentication and communication
    - Script generation with structured prompts
    - Response parsing and validation
    - Error handling and retry logic
    - Input sanitization
    
    Attributes:
        config (DeepSeekConfig): Configuration for the API client.
        client (httpx.AsyncClient): Async HTTP client for API calls.
    """

    def __init__(self, config: Optional[DeepSeekConfig] = None):
        """
        Initialize the DeepSeek service.
        
        Args:
            config: Optional configuration object. If not provided, loads from environment.
            
        Raises:
            APIKeyError: If no API key is found in config or environment.
        """
        if config is None:
            config = self._load_config_from_env()
        
        if not config.api_key:
            raise APIKeyError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable "
                "or provide it in the configuration."
            )
        
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )
        logger.info(f"DeepSeek service initialized with model: {config.model}")

    def _load_config_from_env(self) -> DeepSeekConfig:
        """
        Load configuration from environment variables.
        
        Returns:
            DeepSeekConfig object with values from environment.
        """
        return DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096")),
            timeout_seconds=int(os.getenv("DEEPSEEK_TIMEOUT", "30")),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
        )

    def _build_script_prompt(self, topic: str) -> str:
        """
        Build a structured prompt for script generation.
        
        Args:
            topic: The user's video topic.
            
        Returns:
            Formatted prompt string for the AI model.
            
        Raises:
            ValueError: If topic is empty or too long.
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if len(topic) > 500:
            raise ValueError(f"Topic too long ({len(topic)} chars). Maximum is 500 characters.")
        
        # Sanitize input - remove potentially harmful characters
        sanitized_topic = topic.strip()[:500]
        
        prompt = f"""You are an expert video script writer. Create a detailed video script about the following topic:

Topic: {sanitized_topic}

Requirements:
1. The script should be engaging and well-structured for a video format
2. Include an introduction, main content sections, and a conclusion
3. Each section should have a clear title and content
4. Suggest relevant keywords for finding media (images/videos) for each section
5. Estimate appropriate duration for each section (in seconds)
6. Total video duration should be between 30 seconds and 5 minutes

Please format your response as a JSON object with the following structure:
{{
    "title": "Video Title",
    "sections": [
        {{
            "title": "Section Title",
            "content": "Section script content...",
            "duration_seconds": 15,
            "suggested_keywords": ["keyword1", "keyword2"]
        }}
    ]
}}

Ensure the JSON is valid and complete. Do not include any text outside the JSON object."""

        return prompt

    async def generate_script(self, topic: str) -> GeneratedScript:
        """
        Generate a video script for the given topic.
        
        Args:
            topic: The topic for the video script.
            
        Returns:
            GeneratedScript object containing the structured script.
            
        Raises:
            ValueError: If topic is invalid.
            APIRequestError: If API request fails.
            ScriptGenerationError: If script parsing fails.
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        prompt = self._build_script_prompt(topic)
        
        try:
            raw_response = await self._make_api_request(prompt)
            script_data = self._parse_script_response(raw_response, topic)
            
            # Calculate total duration
            total_duration = sum(
                section.duration_seconds for section in script_data.sections
            )
            
            return GeneratedScript(
                topic=topic,
                title=script_data.title,
                sections=script_data.sections,
                total_duration_seconds=total_duration,
                raw_response=raw_response,
                model_used=self.config.model,
            )
            
        except ValidationError as e:
            logger.error(f"Script validation error: {e}")
            raise ScriptGenerationError(f"Failed to validate generated script: {str(e)}")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in response: {e}")
            raise ScriptGenerationError(f"Failed to parse AI response as JSON: {str(e)}")

    async def _make_api_request(self, prompt: str) -> str:
        """
        Make an API request to DeepSeek with retry logic.
        
        Args:
            prompt: The formatted prompt to send.
            
        Returns:
            Response text from the API.
            
        Raises:
            APIRequestError: If all retry attempts fail.
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"API request attempt {attempt + 1}/{self.config.max_retries + 1}")
                
                response = await self.client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert video script writer. Always respond with valid JSON."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    }
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                # Extract the assistant's message content
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0].get("message", {}).get("content", "")
                    if content:
                        return content
                
                raise APIRequestError("Empty response from API")
                
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e.response.status_code}")
                
                if e.response.status_code == 401:
                    raise APIKeyError("Invalid or expired DeepSeek API key")
                elif e.response.status_code == 429:
                    logger.warning("Rate limited by DeepSeek API")
                    if attempt < self.config.max_retries:
                        await self._wait_with_backoff(attempt)
                        continue
                elif e.response.status_code >= 500:
                    logger.warning(f"Server error from DeepSeek API")
                    if attempt < self.config.max_retries:
                        await self._wait_with_backoff(attempt)
                        continue
                else:
                    raise APIRequestError(f"API request failed with status {e.response.status_code}: {e.response.text}")
                    
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries:
                    await self._wait_with_backoff(attempt)
                    continue
                    
            except httpx.RequestError as e:
                last_error = e
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries:
                    await self._wait_with_backoff(attempt)
                    continue
        
        raise APIRequestError(f"All API request attempts failed. Last error: {last_error}")

    async def _wait_with_backoff(self, attempt: int) -> None:
        """
        Wait with exponential backoff before retrying.
        
        Args:
            attempt: Current attempt number (0-indexed).
        """
        import asyncio
        
        wait_time = min(2 ** attempt * 1.5, 30)  # Cap at 30 seconds
        logger.debug(f"Waiting {wait_time:.1f} seconds before retry")
        await asyncio.sleep(wait_time)

    def _parse_script_response(self, response_text: str, original_topic: str) -> GeneratedScript:
        """
        Parse and validate the AI response into a structured script.
        
        Args:
            response_text: Raw response text from the AI.
            original_topic: Original user topic for validation.
            
        Returns:
            GeneratedScript object with parsed data.
            
        Raises:
            ScriptGenerationError: If parsing or validation fails.
        """
        # Try to extract JSON from the response (handle markdown code blocks)
        json_str = response_text.strip()
        
        # Remove markdown code block markers if present
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        
        json_str = json_str.strip()
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            
            # Attempt to fix common JSON issues
            fixed_json = self._attempt_json_fix(json_str)
            if fixed_json:
                data = fixed_json
            else:
                raise ScriptGenerationError(
                    f"Failed to parse AI response as JSON. Response preview: {response_text[:200]}"
                )
        
        # Validate required fields
        if not isinstance(data, dict):
            raise ScriptGenerationError("Response is not a JSON object")
        
        if "title" not in data or not data["title"]:
            data["title"] = f"Video about {original_topic}"
        
        if "sections" not in data or not data["sections"]:
            # Create a default section if none provided
            data["sections"] = [
                {
                    "title": "Introduction",
                    "content": data.get("content", f"This video explores {original_topic}."),
                    "duration_seconds": 15,
                    "suggested_keywords": [original_topic]
                }
            ]
        
        # Validate and clean sections
        validated_sections = []
        for i, section in enumerate(data["sections"]):
            if not isinstance(section, dict):
                continue
            
            cleaned_section = {
                "title": str(section.get("title", f"Section {i + 1}"))[:200],
                "content": str(section.get("content", ""))[:5000],
                "duration_seconds": max(5, min(120, int(section.get("duration_seconds", 10)))),
                "suggested_keywords": [
                    str(kw)[:100] for kw in section.get("suggested_keywords", [])
                    if isinstance(kw, (str, int, float))
                ][:20]  # Limit to 20 keywords per section
            }
            
            if cleaned_section["content"]:
                validated_sections.append(ScriptSection(**cleaned_section))
        
        if not validated_sections:
            raise ScriptGenerationError("No valid sections found in response")
        
        return GeneratedScript(
            title=str(data["title"])[:300],
            sections=validated_sections,
            total_duration_seconds=sum(s.duration_seconds for s in validated_sections),
            topic=original_topic,
            raw_response=response_text,
            model_used=self.config.model,
        )

    def _attempt_json_fix(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to fix common JSON formatting issues.
        
        Args:
            json_str: Potentially malformed JSON string.
            
        Returns:
            Parsed dictionary if fix succeeds, None otherwise.
        """
        try:
            # Try to find JSON-like content between braces
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate = json_str[start_idx:end_idx + 1]
                
                # Fix common issues like trailing commas
                import re
                candidate = re.sub(r',\s*}', '}', candidate)
                candidate = re.sub(r',\s*]', ']', candidate)
                
                return json.loads(candidate)
                
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None

    async def health_check(self) -> bool:
        """
        Check if the DeepSeek API is accessible and working.
        
        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            response = await self.client.get("/models")
            
            if response.status_code == 200:
                models_data = response.json()
                
                # Check if our configured model is available
                available_models = [
                    model["id"] for model in models_data.get("data", [])
                    if isinstance(model, dict)
                ]
                
                if self.config.model not in available_models:
                    logger.warning(f"Model '{self.config.model}' not found in available models")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with resource cleanup."""
        await self.close()


# Convenience function for quick script generation
async def generate_video_script(topic: str) -> GeneratedScript:
    """
    Convenience function to generate a video script with default configuration.
    
    Args:
        topic: The topic for the video script.
        
    Returns:
        GeneratedScript object.
        
    Example:
        >>> script = await generate_video_script("Artificial Intelligence")
        >>> print(script.title)
    """
    service = DeepSeekService()
    
    try:
        return await service.generate_script(topic)
    finally:
        await service.close()