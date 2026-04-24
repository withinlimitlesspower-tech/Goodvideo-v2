"""
Utility functions for the AI Video Generator application.

This module provides common helper functions for keyword extraction,
file handling, text processing, and other utility tasks used across
the application.
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from datetime import datetime
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_KEYWORDS = 10
MIN_KEYWORD_LENGTH = 2
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png', '.gif'}
MAX_FILENAME_LENGTH = 255
SAFE_FILENAME_PATTERN = re.compile(r'[^\w\-_\. ]')


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing unsafe characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename with safe characters only
        
    Raises:
        ValueError: If filename is empty or too long after sanitization
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Filename must be a non-empty string")
    
    # Remove path separators and null bytes
    filename = filename.replace('/', '_').replace('\\', '_').replace('\0', '')
    
    # Replace unsafe characters with underscores
    filename = SAFE_FILENAME_PATTERN.sub('_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(filename)
        ext = ext[:10]  # Limit extension length
        max_name_length = MAX_FILENAME_LENGTH - len(ext)
        if max_name_length > 0:
            filename = name[:max_name_length] + ext
        else:
            filename = filename[:MAX_FILENAME_LENGTH]
    
    if not filename:
        raise ValueError("Filename is empty after sanitization")
    
    return filename


def generate_unique_filename(prefix: str = "", extension: str = ".mp4") -> str:
    """
    Generate a unique filename using UUID and timestamp.
    
    Args:
        prefix: Optional prefix for the filename
        extension: File extension (default: .mp4)
        
    Returns:
        Unique filename string
        
    Raises:
        ValueError: If extension is invalid
    """
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    if extension.lower() not in ALLOWED_EXTENSIONS and extension.lower() not in {'.txt', '.json', '.wav', '.mp3'}:
        logger.warning(f"Unusual file extension: {extension}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    
    if prefix:
        prefix = sanitize_filename(prefix)
        filename = f"{prefix}_{timestamp}_{unique_id}{extension}"
    else:
        filename = f"{timestamp}_{unique_id}{extension}"
    
    return filename


def extract_keywords(text: str, max_keywords: int = MAX_KEYWORDS) -> List[str]:
    """
    Extract relevant keywords from text for media search queries.
    
    Uses a combination of techniques:
    - Removes common stop words
    - Extracts noun phrases and important terms
    - Filters by minimum length and relevance
    
    Args:
        text: Input text to extract keywords from
        max_keywords: Maximum number of keywords to return (default: 10)
        
    Returns:
        List of extracted keywords
        
    Raises:
        ValueError: If text is empty or invalid
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    # Common English stop words to filter out
    stop_words: Set[str] = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'shall', 'can', 'need', 'dare', 'ought', 'used',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours',
        'theirs'
    }
    
    # Clean and normalize text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into words and filter
    words = text.split()
    filtered_words = [
        word for word in words 
        if word not in stop_words 
        and len(word) >= MIN_KEYWORD_LENGTH
        and word.isalpha()
    ]
    
    # Count word frequencies
    word_freq: Dict[str, int] = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    
    # Extract top keywords, ensuring diversity (avoid similar words)
    keywords: List[str] = []
    seen_stems: Set[str] = set()
    
    for word, freq in sorted_words:
        if len(keywords) >= max_keywords:
            break
        
        # Simple stemming check to avoid duplicates
        stem = word[:4] if len(word) > 4 else word
        if stem not in seen_stems:
            keywords.append(word)
            seen_stems.add(stem)
    
    return keywords


def extract_keywords_from_topic(topic: str) -> List[str]:
    """
    Extract search keywords specifically from a video topic.
    
    Handles multi-word topics and phrases better than general keyword extraction.
    
    Args:
        topic: The video topic string
        
    Returns:
        List of relevant search keywords
        
    Raises:
        ValueError: If topic is empty or invalid
    """
    if not topic or not isinstance(topic, str):
        raise ValueError("Topic must be a non-empty string")
    
    # Clean the topic
    topic = topic.strip()
    
    # For short topics (1-2 words), use as-is
    words = topic.split()
    if len(words) <= 2:
        return [topic.lower()]
    
    # For longer topics, extract meaningful phrases and keywords
    keywords: List[str] = []
    
    # Add the full topic as a phrase (for specific searches)
    keywords.append(topic.lower())
    
    # Extract individual meaningful words
    individual_keywords = extract_keywords(topic, max_keywords=5)
    
    # Add individual keywords that aren't already covered
    for kw in individual_keywords:
        if kw not in keywords and len(keywords) < MAX_KEYWORDS:
            keywords.append(kw)
    
    return keywords


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
        
    Raises:
        OSError: If directory cannot be created or accessed
        ValueError: If path is invalid
    """
    if not directory_path or not isinstance(directory_path, str):
        raise ValueError("Directory path must be a non-empty string")
    
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        
        if not path.is_dir():
            raise OSError(f"Path exists but is not a directory: {directory_path}")
        
        logger.debug(f"Directory ensured: {directory_path}")
        return path
    
    except PermissionError as e:
        logger.error(f"Permission denied creating directory {directory_path}: {e}")
        raise OSError(f"Cannot create directory {directory_path}: Permission denied")
    
    except OSError as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        raise


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: The filename to extract extension from
        
    Returns:
        File extension with leading dot (e.g., '.mp4')
        
    Raises:
        ValueError: If filename is invalid
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Filename must be a non-empty string")
    
    _, extension = os.path.splitext(filename)
    
    if not extension:
        logger.warning(f"No extension found in filename: {filename}")
    
    return extension.lower()


def is_allowed_file(filename: str, allowed_extensions: Optional[Set[str]] = None) -> bool:
    """
    Check if a file has an allowed extension.
    
    Args:
        filename: The filename to check
        allowed_extensions: Set of allowed extensions (default: common media types)
        
    Returns:
        True if file extension is allowed, False otherwise
        
    Raises:
        ValueError: If filename is invalid
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Filename must be a non-empty string")
    
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_EXTENSIONS
    
    extension = get_file_extension(filename)
    
    return extension in allowed_extensions


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB", "234 KB")
        
    Raises:
        ValueError: If size is negative
    """
    if size_bytes < 0:
        raise ValueError("File size cannot be negative")
    
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def safe_json_loads(json_string: str) -> Optional[Dict[str, Any]]:
    """
     Safely parse a JSON string, returning None on failure.
     
     Args:
         json_string: JSON string to parse
         
     Returns:
         Parsed dictionary or None if parsing fails
     """
     if not json_string or not isinstance(json_string, str):
         return None
    
     try:
         data = json.loads(json_string)
         if isinstance(data, dict):
             return data
         logger.warning(f"JSON parsed but not a dictionary: {type(data)}")
         return None
    
     except json.JSONDecodeError as e:
         logger.error(f"JSON parsing error: {e}")
         return None


def hash_string(input_string: str) -> str:
     """
     Create a SHA-256 hash of a string.
     
     Args:
         input_string: String to hash
         
     Returns:
         Hexadecimal hash string
         
     Raises:
         ValueError: If input is empty or invalid
     """
     if not input_string or not isinstance(input_string, str):
         raise ValueError("Input must be a non-empty string")
    
     return hashlib.sha256(input_string.encode('utf-8')).hexdigest()


def clean_text_for_tts(text: str) -> str:
     """
     Clean and prepare text for Text-to-Speech processing.
     
     Removes markdown formatting, special characters, and normalizes whitespace.
     
     Args:
         text: Raw text to clean
         
     Returns:
         Cleaned text suitable for TTS
        
     Raises:
         ValueError: If text is empty or invalid
     """
     if not text or not isinstance(text, str):
         raise ValueError("Text must be a non-empty string")
    
     # Remove markdown formatting
     text = re.sub(r'[*_~`#]', '', text)
     text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove links but keep text
    
     # Remove URLs
     text = re.sub(r'https?://\S+', '', text)
    
     # Remove HTML tags
     text = re.sub(r'<[^>]+>', '', text)
    
     # Normalize whitespace
     text = re.sub(r'\s+', ' ', text)
     text = re.sub(r'\n+', '. ', text)
    
     # Remove leading/trailing whitespace and punctuation artifacts
     text = text.strip('.,;:!? ')
    
     # Ensure text ends with proper punctuation for natural speech flow
     if text and not text[-1] in '.!?':
         text += '.'
    
     return text


def chunk_text_for_tts(text: str, max_chunk_size: int = 5000) -> List[str]:
     """
     Split long text into chunks suitable for TTS processing.
     
     Splits at sentence boundaries when possible.
     
     Args:
         text: Text to split into chunks
         max_chunk_size: Maximum characters per chunk (default: 5000)
         
     Returns:
         List of text chunks
        
     Raises:
         ValueError: If parameters are invalid
     """
     if not text or not isinstance(text, str):
         raise ValueError("Text must be a non-empty string")
    
     if max_chunk_size < 100:
         raise ValueError("max_chunk_size must be at least 100 characters")
    
     # Clean the text first
     cleaned_text = clean_text_for_tts(text)
    
     if len(cleaned_text) <= max_chunk_size:
         return [cleaned_text]
    
     chunks: List[str] = []
     
     # Split by sentences (periods, exclamation marks, question marks)
     sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
     
     current_chunk = ""
     
     for sentence in sentences:
         # If adding this sentence would exceed the limit, start a new chunk
         if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
             if current_chunk.strip():
                 chunks.append(current_chunk.strip())
             
             # If a single sentence is too long, split it further by commas or spaces
             if len(sentence) > max_chunk_size:
                 parts = re.split(r'(?<=[,;:])\s+|\s{2,}', sentence)
                 current_chunk = ""
                 for part in parts:
                     if len(current_chunk) + len(part) + 1 > max_chunk_size and current_chunk.strip():
                         chunks.append(current_chunk.strip())
                         current_chunk = part + " "
                     else:
                         current_chunk += part + " "
                 current_chunk = current_chunk.strip()
             else:
                 current_chunk = sentence + " "
         else:
             current_chunk += sentence + " "
     
     # Add the last chunk if it exists
     if current_chunk.strip():
         chunks.append(current_chunk.strip())
     
     return chunks


def validate_topic(topic: str) -> bool:
     """
     Validate a video topic string.
     
     Checks for minimum length, allowed characters, and reasonable content.
     
     Args:
         topic: Topic string to validate
         
     Returns:
         True if topic is valid, False otherwise
        
     Raises:
         ValueError: If topic is invalid with specific error message
     """
     if not topic or not isinstance(topic, str):
         raise ValueError("Topic must be a non-empty string")
    
     topic = topic.strip()
    
     if len(topic) < 3:
         raise ValueError("Topic must be at least 3 characters long")
    
     if len(topic) > 500:
         raise ValueError("Topic must be less than 500 characters")
    
     # Check for potentially harmful content (basic XSS prevention)
     dangerous_patterns = [
         r'<script[^>]*>',
         r'javascript\s*:',
         r'on\w+\s*=',
         r'<iframe[^>]*>',
         r'<embed[^>]*>',
         r'<object[^>]*>'
     ]
     
     for pattern in dangerous_patterns:
         if re.search(pattern, topic, re.IGNORECASE):
             raise ValueError("Topic contains potentially harmful content")
    
     return True


def get_timestamp() -> str:
     """
     Get current timestamp in ISO format.
     
     Returns:
         ISO formatted timestamp string (e.g., "2024-01-15T10:30:00.123456")
     """
     return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
     """
     Format duration in seconds to human-readable format.
     
     Args:
         seconds: Duration in seconds (float)
         
     Returns:
         Formatted duration string (e.g., "2m 30s", "1h 15m")
        
     Raises:
         ValueError: If seconds is negative
     """
     if seconds < 0:
         raise ValueError("Duration cannot be negative")
    
     hours = int(seconds // 3600)
     minutes = int((seconds % 3600) // 60)
     secs = int(seconds % 60)
     
     parts = []
     
     if hours > 0:
         parts.append(f"{hours}h")
     
     if minutes > 0 or hours > 0:
         parts.append(f"{minutes}m")
     
     parts.append(f"{secs}s")
     
     return " ".join(parts)


def create_temp_directory(base_path: Optional[str] = None) -> Path:
     """
      Create a temporary directory for processing files.
      
      Args:
          base_path: Base directory for temp files (default: system temp dir)
          
      Returns:
          Path object for the created temporary directory
        
      Raises:
          OSError: If directory cannot be created
      """
      import tempfile
    
      try:
          temp_dir = tempfile.mkdtemp(prefix="video_gen_", dir=base_path)
          path = Path(temp_dir)
          logger.debug(f"Created temporary directory: {temp_dir}")
          return path
    
      except OSError as e:
          logger.error(f"Failed to create temporary directory: {e}")
          raise


def cleanup_temp_files(directory_path: Path) -> bool:
      """
      Clean up temporary files in a directory.
      
      Args:
          directory_path: Path to the directory to clean up
            
      Returns:
          True if cleanup was successful, False otherwise
            
      Raises:
          ValueError: If path is invalid or unsafe for deletion
      """
      import shutil
    
      if not directory_path or not isinstance(directory_path, Path):
          raise ValueError("Directory path must be a valid Path object")
      
      # Safety check - only delete directories that look like temp dirs
      dir_name = directory_path.name
      if not dir_name.startswith("video_gen_"):
          logger.warning(f"Skipping cleanup - directory doesn't look like temp dir: {dir_name}")
          return False
    
      try:
          shutil.rmtree(directory_path)
          logger.debug(f"Cleaned up temporary directory: {directory_path}")
          return True
    
      except PermissionError as e:
          logger.error(f"Permission denied cleaning up {directory_path}: {e}")
          return False
    
      except OSError as e:
          logger.error(f"Error cleaning up {directory_path}: {e}")
          return False


# Export commonly used functions for easy import
__all__ = [
      "sanitize_filename",
      "generate_unique_filename",
      "extract_keywords",
      "extract_keywords_from_topic",
      "ensure_directory",
      "get_file_extension",
      "is_allowed_file",
      "format_file_size",
      "safe_json_loads",
      "hash_string",
      "clean_text_for_tts",
      "chunk_text_for_tts",
      "validate_topic",
      "get_timestamp",
      "format_duration",
      "create_temp_directory",
      "cleanup_temp_files"
]