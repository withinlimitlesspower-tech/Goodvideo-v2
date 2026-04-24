"""
Media Handler Module

Handles downloading and caching of media files from Pixabay API.
Supports both images and videos with proper error handling and caching mechanisms.
"""

import os
import hashlib
import asyncio
import aiohttp
import aiofiles
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
from urllib.parse import urlparse, quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaHandlerError(Exception):
    """Base exception for media handler errors."""
    pass


class MediaDownloadError(MediaHandlerError):
    """Raised when media download fails."""
    pass


class MediaCacheError(MediaHandlerError):
    """Raised when caching operations fail."""
    pass


class MediaHandler:
    """
    Handles downloading and caching of media files from Pixabay.
    
    Features:
    - Async download with progress tracking
    - File-based caching with TTL
    - Automatic retry on failure
    - Concurrent download support
    - File type validation
    """
    
    def __init__(
        self,
        cache_dir: str = "media_cache",
        cache_ttl_hours: int = 24,
        max_concurrent_downloads: int = 5,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the media handler.
        
        Args:
            cache_dir: Directory for cached media files
            cache_ttl_hours: Time-to-live for cached files in hours
            max_concurrent_downloads: Maximum concurrent downloads
            max_retries: Maximum retry attempts for failed downloads
            retry_delay: Delay between retries in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize semaphore for concurrent downloads
        self._download_semaphore = asyncio.Semaphore(max_concurrent_downloads)
        
        # Supported file extensions
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        self.supported_video_extensions = {'.mp4', '.webm', '.mov', '.avi'}
        
        logger.info(f"Media handler initialized with cache dir: {cache_dir}")
    
    def _get_cache_key(self, url: str) -> str:
        """
        Generate a unique cache key for a URL.
        
        Args:
            url: Media URL to generate key for
            
        Returns:
            str: MD5 hash of the URL as cache key
        """
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, url: str, file_extension: str) -> Path:
        """
        Get the cache file path for a URL.
        
        Args:
            url: Media URL
            file_extension: File extension including dot
            
        Returns:
            Path: Full path to cached file
        """
        cache_key = self._get_cache_key(url)
        return self.cache_dir / f"{cache_key}{file_extension}"
    
    def _validate_file_extension(self, url: str) -> Optional[str]:
        """
        Validate and extract file extension from URL.
        
        Args:
            url: Media URL to validate
            
        Returns:
            Optional[str]: File extension if valid, None otherwise
            
        Raises:
            MediaDownloadError: If file extension is not supported
        """
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            ext = Path(path).suffix
            
            if ext in self.supported_image_extensions or ext in self.supported_video_extensions:
                return ext
            
            logger.warning(f"Unsupported file extension: {ext} for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error validating file extension for URL {url}: {str(e)}")
            return None
    
    def _is_cached_valid(self, cache_path: Path) -> bool:
        """
        Check if a cached file is still valid (not expired).
        
        Args:
            cache_path: Path to cached file
            
        Returns:
            bool: True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        # Check if file is older than TTL
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < self.cache_ttl
    
    async def download_media(
        self,
        url: str,
        session: aiohttp.ClientSession,
        timeout: int = 30
    ) -> Optional[Path]:
        """
        Download media from URL with caching support.
        
        Args:
            url: Media URL to download
            session: aiohttp ClientSession for making requests
            timeout: Request timeout in seconds
            
        Returns:
            Optional[Path]: Path to downloaded/cached file, None on failure
            
        Raises:
            MediaDownloadError: If download fails after all retries
            MediaCacheError: If caching operation fails
        """
        # Validate URL
        if not url or not isinstance(url, str):
            logger.error("Invalid URL provided")
            return None
        
        # Validate and get file extension
        file_extension = self._validate_file_extension(url)
        if not file_extension:
            logger.warning(f"Unsupported media type for URL: {url}")
            return None
        
        # Check cache first
        cache_path = self._get_cache_path(url, file_extension)
        
        if self._is_cached_valid(cache_path):
            logger.info(f"Using cached media: {cache_path}")
            return cache_path
        
        # Download with retry logic
        for attempt in range(self.max_retries):
            try:
                async with self._download_semaphore:
                    logger.info(f"Downloading media (attempt {attempt + 1}/{self.max_retries}): {url}")
                    
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                        if response.status != 200:
                            error_msg = f"HTTP {response.status} error downloading {url}"
                            logger.warning(error_msg)
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        
                        # Read content and save to cache
                        content = await response.read()
                        
                        # Validate content size (prevent empty files)
                        if len(content) == 0:
                            logger.warning(f"Empty content received from {url}")
                            continue
                        
                        # Save to cache directory
                        try:
                            async with aiofiles.open(cache_path, 'wb') as f:
                                await f.write(content)
                            
                            logger.info(f"Successfully downloaded and cached: {cache_path}")
                            return cache_path
                            
                        except IOError as e:
                            raise MediaCacheError(f"Failed to write cache file: {str(e)}")
                
                break  # Success, exit retry loop
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading {url} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
            except aiohttp.ClientError as e:
                logger.error(f"Client error downloading {url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        error_msg = f"Failed to download media after {self.max_retries} attempts: {url}"
        logger.error(error_msg)
        
        # Clean up partial download if exists
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Cleaned up partial download: {cache_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up partial download: {str(e)}")
        
        return None
    
    async def download_multiple(
        self,
        urls: List[str],
        session: aiohttp.ClientSession,
        timeout: int = 30,
        progress_callback=None
    ) -> Dict[str, Optional[Path]]:
        """
        Download multiple media files concurrently.
        
        Args:
            urls: List of media URLs to download
            session: aiohttp ClientSession for making requests
            timeout: Request timeout in seconds per download
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict[str, Optional[Path]]: Mapping of URLs to their downloaded paths (None on failure)
            
        Raises:
            MediaDownloadError: If all downloads fail
        """
        if not urls:
            logger.warning("No URLs provided for download")
            return {}
        
        results = {}
        
        # Create tasks for concurrent downloads
        tasks = []
        
        for i, url in enumerate(urls):
            task = asyncio.create_task(
                self.download_media(url, session, timeout)
            )
            tasks.append((url, task))
        
        # Wait for all downloads to complete with progress updates
        completed = 0
        total = len(tasks)
        
        for url, task in tasks:
            try:
                result = await task
                results[url] = result
                
                completed += 1
                
                # Call progress callback if provided
                if progress_callback and callable(progress_callback):
                    try:
                        progress_callback(completed, total, url, result is not None)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {str(e)}")
                
                logger.info(f"Download progress: {completed}/{total} completed")
                
            except Exception as e:
                logger.error(f"Task failed for URL {url}: {str(e)}")
                results[url] = None
        
        # Log summary
        successful = sum(1 for path in results.values() if path is not None)
        failed = total - successful
        
        logger.info(f"Download complete: {successful} successful, {failed} failed out of {total}")
        
        return results
    
    def clear_expired_cache(self) -> int:
        """
        Clear expired cache files.
        
        Returns:
            int: Number of files cleared
            
        Raises:
            MediaCacheError: If cache clearing fails
        """
        cleared_count = 0
        
        try:
            for cache_file in self.cache_dir.iterdir():
                if cache_file.is_file():
                    try:
                        if not self._is_cached_valid(cache_file):
                            cache_file.unlink()
                            cleared_count += 1
                            logger.info(f"Cleared expired cache file: {cache_file}")
                    except OSError as e:
                        logger.warning(f"Failed to clear cache file {cache_file}: {str(e)}")
            
            logger.info(f"Cleared {cleared_count} expired cache files")
            
        except Exception as e:
            raise MediaCacheError(f"Failed to clear expired cache: {str(e)}")
        
        return cleared_count
    
    def clear_all_cache(self) -> int:
        """
         Clear all cached files.
        
         Returns:
             int: Number of files cleared
        
         Raises:
             MediaCacheError: If cache clearing fails
         """
         cleared_count = 0
        
         try:
             for cache_file in self.cache_dir.iterdir():
                 if cache_file.is_file():
                     try:
                         cache_file.unlink()
                         cleared_count += 1
                         logger.info(f"Cleared cache file: {cache_file}")
                     except OSError as e:
                         logger.warning(f"Failed to clear cache file {cache_file}: {str(e)}")
            
             logger.info(f"Cleared all {cleared_count} cached files")
            
         except Exception as e:
             raise MediaCacheError(f"Failed to clear all cache: {str(e)}")
        
         return cleared_count
    
    def get_cache_stats(self) -> Dict[str, any]:
         """
         Get statistics about the media cache.
        
         Returns:
             Dict[str, any]: Cache statistics including total files, size, etc.
         """
         stats = {
             'total_files': 0,
             'total_size_bytes': 0,
             'expired_files': 0,
             'valid_files': 0,
             'cache_directory': str(self.cache_dir),
             'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600,
             'max_concurrent_downloads': self.max_concurrent_downloads,
             'max_retries': self.max_retries,
             'supported_formats': {
                 'images': list(self.supported_image_extensions),
                 'videos': list(self.supported_video_extensions)
             }
         }
        
         try:
             for cache_file in self.cache_dir.iterdir():
                 if cache_file.is_file():
                     stats['total_files'] += 1
                     stats['total_size_bytes'] += cache_file.stat().st_size
                    
                     if self._is_cached_valid(cache_file):
                         stats['valid_files'] += 1
                     else:
                         stats['expired_files'] += 1
            
             stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
            
         except Exception as e:
             logger.error(f"Failed to get cache stats: {str(e)}")
        
         return stats


# Convenience function for creating a media handler instance with default settings
def create_media_handler(
     cache_dir: str = "media_cache",
     **kwargs
 ) -> MediaHandler:
     """
     Create a MediaHandler instance with default configuration.
    
     Args:
         cache_dir: Directory for cached media files (default: "media_cache")
         **kwargs: Additional arguments passed to MediaHandler constructor
    
     Returns:
         MediaHandler: Configured media handler instance
    
     Example:
         handler = create_media_handler(cache_dir="custom_cache", max_concurrent_downloads=3)
     """
     return MediaHandler(cache_dir=cache_dir, **kwargs)


# Example usage and testing code (commented out)
"""
# Example usage in an async context:

async def main():
     # Create handler with custom settings
     handler = create_media_handler(
         cache_dir="my_cache",
         max_concurrent_downloads=3,
         max_retries=2,
         retry_delay=1,
         cache_ttl_hours=48
     )
    
     async with aiohttp.ClientSession() as session:
         # Download single media file
         result = await handler.download_media(
             "https://example.com/image.jpg",
             session,
             timeout=30
         )
        
         if result:
             print(f"Downloaded to: {result}")
        
         # Download multiple files concurrently with progress tracking
         urls = [
             "https://example.com/video1.mp4",
             "https://example.com/image2.png",
             "https://example.com/video3.webm"
         ]
        
         def progress(current, total, url, success):
             status = "✓" if success else "✗"
             print(f"[{status}] Downloaded {current}/{total}: {url}")
        
         results = await handler.download_multiple(
             urls,
             session,
             progress_callback=progress
         )
        
         # Print results summary
         for url, path in results.items():
             status = f"Cached at {path}" if path else "Failed"
             print(f"{url}: {status}")
    
     # Get cache statistics
     stats = handler.get_cache_stats()
     print(f"\nCache Statistics:")
     print(f"Total files: {stats['total_files']}")
     print(f"Total size: {stats['total_size_mb']} MB")
     print(f"Valid files: {stats['valid_files']}")
     print(f"Expired files: {stats['expired_files']}")
    
     # Clear expired cache entries periodically (e.g., via cron job)
     cleared_count = handler.clear_expired_cache()
     print(f"\nCleared {cleared_count} expired entries")

# Run the example (requires Python 3.7+)
# asyncio.run(main())
"""