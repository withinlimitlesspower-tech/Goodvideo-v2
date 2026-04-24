```py
"""
Database module for AI Video Generator application.

This module provides SQLite database setup and session management using SQLAlchemy.
It handles database initialization, session creation, and CRUD operations for
storing user sessions and generated video content metadata.

Dependencies:
    - SQLAlchemy >= 2.0
    - sqlite3 (built-in)
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    Boolean,
    Float,
    ForeignKey,
    UniqueConstraint,
    Index,
    event,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
    relationship,
    scoped_session,
)
from sqlalchemy.exc import (
    SQLAlchemyError,
    IntegrityError,
    OperationalError,
    DatabaseError,
)
from sqlalchemy.pool import QueuePool

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./video_generator.db"
)
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "false").lower() == "true"

# Create SQLAlchemy engine with connection pooling for SQLite
engine = create_engine(
    DATABASE_URL,
    echo=DATABASE_ECHO,
    poolclass=QueuePool,
    pool_size=DATABASE_POOL_SIZE,
    max_overflow=DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,  # Verify connections before using
    connect_args={
        "check_same_thread": False,  # Required for FastAPI async support
        "timeout": 30,  # SQLite timeout in seconds
        "isolation_level": "IMMEDIATE",  # Better concurrency handling
    },
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Create scoped session for thread safety
db_session = scoped_session(SessionLocal)

# Create declarative base
Base = declarative_base()


def get_database_path() -> str:
    """
    Get the database file path from the URL.
    
    Returns:
        str: Path to the SQLite database file
        
    Raises:
        ValueError: If database URL is not a valid SQLite URL
    """
    if not DATABASE_URL.startswith("sqlite:///"):
        raise ValueError(f"Invalid database URL format: {DATABASE_URL}")
    
    db_path = DATABASE_URL.replace("sqlite:///", "")
    
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    return db_path


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    This function should be called once at application startup.
    
    Raises:
        DatabaseError: If table creation fails
        OperationalError: If database connection fails
    """
    try:
        # Import all models to ensure they are registered with Base
        from models import UserSession, VideoGeneration, MediaAsset, Voiceover
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database tables created successfully")
        
        # Enable WAL mode for better concurrent access
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.commit()
            
    except OperationalError as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
    except DatabaseError as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def get_db() -> Session:
    """
    Get a database session.
    
    This function should be used as a dependency in FastAPI endpoints.
    
    Yields:
        Session: SQLAlchemy database session
        
    Raises:
        SQLAlchemyError: If session creation fails
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Session:
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_context() as db:
            db.query(UserSession).all()
    
    Yields:
        Session: SQLAlchemy database session
        
    Raises:
        SQLAlchemyError: If session operations fail
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# Model definitions

class UserSession(Base):
    """
    Represents a user chat session.
    
    Stores session metadata and conversation history for the AI Video Generator.
    
    Attributes:
        id: Primary key
        session_id: Unique session identifier (UUID)
        user_id: Optional user identifier for multi-user support
        title: Session title generated from first message
        status: Session status (active, archived, deleted)
        messages: JSON array of chat messages
        created_at: Timestamp of session creation
        updated_at: Timestamp of last update
        is_active: Boolean flag for active sessions
        metadata: Additional session metadata as JSON
    """
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        String(36),
        unique=True,
        nullable=False,
        index=True,
        comment="UUID v4 session identifier",
    )
    user_id = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Optional user identifier",
    )
    title = Column(
        String(255),
        nullable=True,
        comment="Session title",
    )
    status = Column(
        String(20),
        nullable=False,
        default="active",
        comment="Session status: active, archived, deleted",
    )
    messages = Column(
        JSON,
        nullable=False,
        default=list,
        comment="Array of chat messages with role and content",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Session creation timestamp",
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Last update timestamp",
    )
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether session is currently active",
    )
    metadata = Column(
        JSON,
        nullable=True,
        default=dict,
        comment="Additional session metadata",
    )
    
    # Relationships
    video_generations = relationship(
        "VideoGeneration",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_session_user_status", "user_id", "status"),
        Index("idx_session_created", "created_at"),
        UniqueConstraint("session_id", name="uq_session_id"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status,
            "messages": self.messages or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "metadata": self.metadata or {},
            "video_count": len(self.video_generations) if self.video_generations else 0,
        }


class VideoGeneration(Base):
    """
    Represents a video generation request and its result.
    
    Stores the input topic, generated script, media assets used, voiceover info,
    and the final video output path.
    
    Attributes:
        id: Primary key
        session_id: Foreign key to UserSession
        topic: User-provided topic for video generation
        script: Generated script from DeepSeek AI
        script_length: Length of generated script in characters
        status: Generation status (pending, processing, completed, failed)
        progress: Generation progress percentage (0-100)
        error_message: Error message if generation failed
        output_path: Path to generated video file
        duration: Video duration in seconds
        created_at: Timestamp of generation request
        completed_at: Timestamp of completion/failure
        metadata: Additional generation metadata as JSON
    """
    
    __tablename__ = "video_generations"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("user_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent session identifier",
    )
    topic = Column(
        String(500),
        nullable=False,
        comment="User-provided video topic",
    )
    script = Column(
        Text,
        nullable=True,
        comment="Generated video script",
    )
    script_length = Column(
        Integer,
        nullable=True,
        comment="Script character count",
    )
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        comment="Generation status: pending, processing, completed, failed",
    )
    progress = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Generation progress percentage (0-100)",
    )
    error_message = Column(
        Text,
        nullable=True,
        comment="Error details if generation failed",
    )
    output_path = Column(
        String(500),
        nullable=True,
        comment="Path to generated video file",
    )
    duration = Column(
        Float,
        nullable=True,
        comment="Video duration in seconds",
    )