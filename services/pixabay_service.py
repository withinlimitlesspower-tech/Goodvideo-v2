"""
Pixabay Media Service for AI Video Generator

This module provides a service to fetch relevant media (images and videos)
from the Pixabay API based on script keywords. It handles API communication,
error handling, rate limiting, and media selection logic.

Dependencies:
    - httpx: Async HTTP client for API requests
    - pydantic: Data validation and settings management
    - typing: Type hints for better code documentation
"""

import os
import logging
from typing import List, Optional, Dict, Any
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logger = logging.getLogger(__name__)


class MediaType(str, Enum):
    """Enumeration for supported media types."""
    IMAGE = "image"
    VIDEO = "video"


class PixabayMedia(BaseModel):
    """Pydantic model for a single media item from Pixabay."""
    id: int
    type: MediaType
    url: str
    preview_url: str = Field(alias="previewURL")
    tags: str
    user: str
    page_url: str = Field(alias="pageURL")
    
    # Image-specific fields
    image_width: Optional[int] = Field(default=None, alias="imageWidth")
    image_height: Optional[int] = Field(default=None, alias="imageHeight")
    large_image_url: Optional[str] = Field(default=None, alias="largeImageURL")
    
    # Video-specific fields
    duration: Optional[int] = Field(default=None)
    video_url: Optional[str] = Field(default=None)
    
    class Config:
        """Pydantic configuration."""
        populate_by_name = True
        arbitrary_types_allowed = True


class PixabaySearchResult(BaseModel):
    """Pydantic model for Pixabay API search response."""
    total: int
    total_hits: int
    hits: List[Dict[str, Any]]


class PixabayServiceError(Exception):
    """Custom exception for Pixabay service errors."""
    pass


class PixabayRateLimitError(PixabayServiceError):
    """Exception raised when API rate limit is exceeded."""
    pass


class PixabayService:
    """
    Service class for interacting with Pixabay API.
    
    This service handles fetching media (images/videos) from Pixabay based on
    search queries derived from script keywords. It includes error handling,
    rate limiting awareness, and media selection logic.
    
    Attributes:
        api_key (str): Pixabay API key from environment variables
        base_url (str): Base URL for Pixabay API
        timeout (int): Request timeout in seconds
        max_retries (int): Maximum number of retry attempts
        client (httpx.AsyncClient): Async HTTP client instance
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://pixabay.com/api",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the Pixabay service.
        
        Args:
            api_key: Pixabay API key. If None, loads from PIXABAY_API_KEY env var.
            base_url: Base URL for Pixabay API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
        
        Raises:
            PixabayServiceError: If API key is not found.
        """
        self.api_key = api_key or os.getenv("PIXABAY_API_KEY")
        if not self.api_key:
            raise PixabayServiceError(
                "Pixabay API key not found. Set PIXABAY_API_KEY environment variable."
            )
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize async HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={"User-Agent": "AIVideoGenerator/1.0"}
        )
        
        logger.info("PixabayService initialized successfully")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client session."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            logger.debug("HTTP client session closed")
    
    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize and validate search query.
        
        Args:
            query: Raw search query string.
        
        Returns:
            Sanitized query string.
        
        Raises:
            ValueError: If query is empty or contains invalid characters.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        # Remove potentially harmful characters and limit length
        sanitized = query.strip()[:100]
        
        # Remove special characters that might cause issues
        import re
        sanitized = re.sub(r'[^\w\s\-]', '', sanitized)
        
        return sanitized.strip()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to Pixabay API with retry logic.
        
        Args:
            endpoint: API endpoint path.
            params: Query parameters for the request.
            retry_count: Current retry attempt number.
        
        Returns:
            JSON response from the API.
        
        Raises:
            PixabayRateLimitError: If rate limit is exceeded.
            PixabayServiceError: For other API errors.
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            params["key"] = self.api_key
            
            logger.debug(f"Making request to {url} with params: {params}")
            
            response = await self.client.get(url, params=params)
            
            if response.status_code == 429:
                raise PixabayRateLimitError(
                    "Pixabay API rate limit exceeded. Please try again later."
                )
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API-level errors
            if "error" in data:
                raise PixabayServiceError(f"Pixabay API error: {data['error']}")
            
            return data
            
        except httpx.TimeoutException as e:
            logger.warning(f"Request timeout (attempt {retry_count + 1}): {e}")
            if retry_count < self.max_retries:
                import asyncio
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await self._make_request(endpoint, params, retry_count + 1)
            raise PixabayServiceError(f"Request timed out after {self.max_retries} retries")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e}")
            if e.response.status_code == 429:
                raise PixabayRateLimitError("Rate limit exceeded")
            raise PixabayServiceError(f"HTTP error: {e}")
            
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise PixabayServiceError(f"Request failed: {e}")
    
    async def search_images(
        self,
        query: str,
        per_page: int = 5,
        image_type: str = "photo",
        orientation: str = "horizontal",
        category: Optional[str] = None,
        min_width: int = 1920,
        min_height: int = 1080,
        safesearch: bool = True
    ) -> List[PixabayMedia]:
        """
        Search for images on Pixabay.
        
        Args:
            query: Search query string.
            per_page: Number of results per page (max 200).
            image_type: Type of image (photo, illustration, vector).
            orientation: Image orientation (horizontal, vertical, all).
            category: Image category filter.
            min_width: Minimum image width.
            min_height: Minimum image height.
            safesearch: Enable safe search filtering.
        
        Returns:
            List of PixabayMedia objects representing images.
        
        Raises:
            ValueError: If query is invalid.
            PixabayServiceError: For API errors.
        """
        try:
            sanitized_query = self._sanitize_query(query)
            
            params = {
                "q": sanitized_query,
                "image_type": image_type,
                "orientation": orientation,
                "per_page": min(per_page, 200),
                "safesearch": str(safesearch).lower(),
                "min_width": min_width,
                "min_height": min_height,
                "order": "popular"
            }
            
            if category:
                params["category"] = category
            
            data = await self._make_request("", params)
            
            # Parse and validate response
            result = PixabaySearchResult(**data)
            
            images = []
            for hit in result.hits[:per_page]:
                try:
                    media = PixabayMedia(
                        id=hit["id"],
                        type=MediaType.IMAGE,
                        url=hit.get("largeImageURL", hit.get("webformatURL", "")),
                        preview_url=hit.get("previewURL", ""),
                        tags=hit.get("tags", ""),
                        user=hit.get("user", ""),
                        page_url=hit.get("pageURL", ""),
                        image_width=hit.get("imageWidth"),
                        image_height=hit.get("imageHeight"),
                        large_image_url=hit.get("largeImageURL")
                    )
                    images.append(media)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Skipping invalid image hit: {e}")
                    continue
            
            logger.info(f"Found {len(images)} images for query '{sanitized_query}'")
            return images
            
        except ValueError as e:
            logger.error(f"Invalid search query: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to search images: {e}")
            raise PixabayServiceError(f"Image search failed: {e}")
    
    async def search_videos(
        self,
        query: str,
        per_page: int = 3,
        video_type: str = "all",
        orientation: str = "horizontal",
        category: Optional[str] = None,
        min_width: int = 1920,
        min_height: int = 1080,
        safesearch: bool = True
    ) -> List[PixabayMedia]:
        """
        Search for videos on Pixabay.
        
        Args:
            query: Search query string.
            per_page: Number of results per page (max 200).
            video_type: Type of video (all, film, animation).
            orientation: Video orientation (horizontal, vertical, all).
            category: Video category filter.
            min_width: Minimum video width.
            min_height: Minimum video height.
            safesearch: Enable safe search filtering.
        
        Returns:
            List of PixabayMedia objects representing videos.
        
        Raises:
            ValueError: If query is invalid.
            PixabayServiceError: For API errors.
        """
        try:
            sanitized_query = self._sanitize_query(query)
            
            params = {
                "q": sanitized_query,
                "video_type": video_type,
                "orientation": orientation,
                "per_page": min(per_page, 200),
                "safesearch": str(safesearch).lower(),
                "min_width": min_width,
                "min_height": min_height,
                "order": "popular"
            }
            
            if category:
                params["category"] = category
            
            data = await self._make_request("videos", params)
            
            # Parse and validate response
            result = PixabaySearchResult(**data)
            
            videos = []
            for hit in result.hits[:per_page]:
                try:
                    # Get the best quality video URL available
                    video_files = hit.get("videos", {}).get("large", {})
                    if not video_files:
                        video_files = hit.get("videos", {}).get("medium", {})
                    if not video_files:
                        video_files = hit.get("videos", {}).get("small", {})
                    
                    video_url = video_files.get("url", "") if video_files else ""
                    
                    media = PixabayMedia(
                        id=hit["id"],
                        type=MediaType.VIDEO,
                        url=video_url,
                        preview_url=hit.get("picture_id", ""),
                        tags=hit.get("tags", ""),
                        user=hit.get("user", ""),
                        page_url=hit.get("pageURL", ""),
                        duration=hit.get("duration"),
                        video_url=video_url,
                        image_width=video_files.get("width") if video_files else None,
                        image_height=video_files.get("height") if video_files else None
                    )
                    videos.append(media)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Skipping invalid video hit: {e}")
                    continue
            
            logger.info(f"Found {len(videos)} videos for query '{sanitized_query}'")
            return videos
            
        except ValueError as e:
            logger.error(f"Invalid search query: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to search videos: {e}")
            raise PixabayServiceError(f"Video search failed: {e}")
    
    async def search_media(
        self,
        keywords: List[str],
        media_type: MediaType = MediaType.IMAGE,
        max_results_per_keyword: int = 2,
        **kwargs
    ) -> List[PixabayMedia]:
        """
        Search for media based on multiple keywords.
        
        This method searches for media using each keyword and combines results,
        removing duplicates by media ID.
        
        Args:
            keywords: List of search keywords from script analysis.
            media_type: Type of media to search for (image or video).
            max_results_per_keyword: Maximum results to fetch per keyword.
            **kwargs: Additional parameters passed to search method.
        
        Returns:
            Combined list of unique PixabayMedia objects.
        
        Raises:
            ValueError: If keywords list is empty.
            PixabayServiceError: For API errors.
        """
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        
        all_media = []
        seen_ids = set()
        
        for keyword in keywords[:5]:  # Limit to first 5 keywords to avoid excessive API calls
            try:
                if media_type == MediaType.IMAGE:
                    results = await self.search_images(
                        query=keyword,
                        per_page=max_results_per_keyword,
                        **kwargs
                    )
                else:
                    results = await self.search_videos(
                        query=keyword,
                        per_page=max_results_per_keyword,
                        **kwargs
                    )
                
                # Add unique results only
                for media in results:
                    if media.id not in seen_ids:
                        all_media.append(media)
                        seen_ids.add(media.id)
                
                logger.debug(f"Found {len(results)} unique results for keyword '{keyword}'")
                
                # Small delay to avoid rate limiting between requests
                import asyncio
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to search for keyword '{keyword}': {e}")
                continue
        
        logger.info(f"Total unique media found across all keywords: {len(all_media)}")
        
        # Sort by relevance (more tags matching keywords is better)
        def relevance_score(media):
            tag_list = [tag.strip().lower() for tag in media.tags.split(",")]
            return sum(1 for kw in keywords if any(kw.lower() in tag for tag in tag_list))
        
        all_media.sort(key=relevance_score, reverse=True)
        
        return all_media
    
    async def get_media_by_id(self, media_id: int, media_type: MediaType) -> Optional[PixabayMedia]:
        """
         Fetch a specific media item by its ID.
        
         Args:
             media_id: The ID of the media item to fetch.
             media_type: Type of media (image or video).
        
         Returns:
             PixabayMedia object if found, None otherwise.
        
         Raises:
             PixabayServiceError: For API errors.
         """
         try:
             endpoint = "" if media_type == MediaType.IMAGE else "videos"
             params = {"id": media_id}
            
             data = await self._make_request(endpoint, params)
            
             if data.get("total_hits", 0) == 0 or not data.get("hits"):
                 logger.warning(f"No media found with ID {media_id}")
                 return None
            
             hit = data["hits"][0]
            
             if media_type == MediaType.IMAGE:
                 return PixabayMedia(
                     id=hit["id"],
                     type=MediaType.IMAGE,
                     url=hit.get("largeImageURL", hit.get("webformatURL", "")),
                     preview_url=hit.get("previewURL", ""),
                     tags=hit.get("tags", ""),
                     user=hit.get("user", ""),
                     page_url=hit.get("pageURL", ""),
                     image_width=hit.get("imageWidth"),
                     image_height=hit.get("imageHeight"),
                     large_image_url=hit.get("largeImageURL")
                 )
             else:
                 video_files = hit.get("videos", {}).get("large", {})
                 if not video_files:
                     video_files = hit.get("videos", {}).get("medium", {})
                 if not video_files:
                     video_files = hit.get("videos", {}).get("small", {})
                
                 video_url = video_files.get("url", "") if video_files else ""
                
                 return PixabayMedia(
                     id=hit["id"],
                     type=MediaType.VIDEO,
                     url=video_url,
                     preview_url=hit.get("picture_id", ""),
                     tags=hit.get("tags", ""),
                     user=hit.get("user", ""),
                     page_url=hit.get("pageURL", ""),
                     duration=hit.get("duration"),
                     video_url=video_url,
                     image_width=video_files.get("width") if video_files else None,
                     image_height=video_files.get("height") if video_files else None
                 )
            
         except Exception as e:
             logger.error(f"Failed to fetch media by ID {media_id}: {e}")
             raise PixabayServiceError(f"Failed to fetch media by ID: {e}")


# Convenience function for creating service instance
async def create_pixabay_service(**kwargs) -> PixabayService:
    """
     Create and return a configured PixabayService instance.
    
     This is a convenience function that can be used as a dependency in FastAPI.
    
     Args:
         **kwargs: Additional arguments passed to PixabayService constructor.
    
     Returns:
         Configured PixabayService instance.
     """
     service = PixabayService(**kwargs)
     return service


# Example usage and testing
if __name__ == "__main__":
     import asyncio
    
     async def test_service():
         """Test the Pixabay service functionality."""
         try:
             async with PixabayService() as service:
                 # Test image search
                 print("Testing image search...")
                 images = await service.search_images(
                     query="nature landscape",
                     per_page=3
                 )
                 print(f"Found {len(images)} images:")
                 for img in images[:2]:
                     print(f"  - Image {img.id}: {img.tags[:50]}...")
                
                 # Test video search
                 print("\nTesting video search...")
                 videos = await service.search_videos(
                     query="ocean waves",
                     per_page=2
                 )
                 print(f"Found {len(videos)} videos:")
                 for vid in videos[:2]:
                     print(f"  - Video {vid.id}: duration={vid.duration}s")
                
                 # Test multi-keyword search
                 print("\nTesting multi-keyword search...")
                 keywords = ["sunset", "beach", "mountains"]
                 combined = await service.search_media(
                     keywords=keywords,
                     media_type=MediaType.IMAGE,
                     max_results_per_keyword=2
                 )
                 print(f"Combined results across keywords: {len(combined)} unique items")
                
         except Exception as e:
             print(f"Test failed with error: {e}")
    
     # Run the test
     asyncio.run(test_service())