"""
SQLAlchemy models for the AI Video Generator chat application.

This module defines the database schema for storing chat sessions and messages,
supporting the conversation flow where users input topics and receive AI-generated
video content.
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    JSON,
    Boolean,
    create_engine,
)
from sqlalchemy.dialects.sqlite import TEXT as SQLiteText
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, validates
import enum

Base = declarative_base()


class MessageRole(str, enum.Enum):
    """Enumeration of possible message roles in a chat session."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, enum.Enum):
    """Enumeration of message content types."""

    TEXT = "text"
    SCRIPT = "script"
    VIDEO = "video"
    ERROR = "error"
    STATUS = "status"


class SessionStatus(str, enum.Enum):
    """Enumeration of chat session statuses."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class ChatSession(Base):
    """
    Represents a chat session in the AI Video Generator application.

    Each session corresponds to a conversation where a user inputs topics
    and receives AI-generated video content. Sessions store metadata about
    the generation process and maintain a history of messages.

    Attributes:
        id: Unique identifier for the session (UUID string).
        title: Optional user-provided or auto-generated title.
        status: Current status of the session.
        created_at: Timestamp when the session was created.
        updated_at: Timestamp when the session was last modified.
        completed_at: Timestamp when the session was completed.
        metadata_json: Flexible JSON field for additional session data.
        messages: Relationship to associated chat messages.
    """

    __tablename__ = "chat_sessions"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False,
        comment="Unique session identifier (UUID v4)",
    )
    title = Column(
        String(255),
        nullable=True,
        comment="Session title (auto-generated or user-provided)",
    )
    status = Column(
        SQLEnum(SessionStatus),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="Current session status",
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="Timestamp when session was created",
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="Timestamp when session was last updated",
    )
    completed_at = Column(
        DateTime,
        nullable=True,
        comment="Timestamp when session was completed",
    )
    metadata_json = Column(
        JSON,
        default=dict,
        nullable=True,
        comment="Flexible JSON field for additional metadata",
    )

    # Relationships
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="dynamic",
    )

    @validates("title")
    def validate_title(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate and sanitize the session title."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("Title must be a string")
            # Limit title length and strip whitespace
            value = value.strip()[:255]
            if not value:
                return None
        return value

    @validates("status")
    def validate_status(self, key: str, value: SessionStatus) -> SessionStatus:
        """Validate the session status."""
        if not isinstance(value, SessionStatus):
            raise ValueError(f"Invalid status: {value}")
        return value

    def to_dict(self) -> dict:
        """Convert session to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata_json or {},
            "message_count": self.messages.count() if self.messages else 0,
        }

    def __repr__(self) -> str:
        """String representation of the ChatSession."""
        return (
            f"<ChatSession(id='{self.id}', "
            f"title='{self.title}', "
            f"status='{self.status.value if self.status else None}', "
            f"created_at='{self.created_at}')>"
        )


class ChatMessage(Base):
    """
    Represents a single message within a chat session.

    Messages can be user inputs (topics), assistant responses (scripts, video info),
    system messages (status updates), or error messages. Each message stores the
    content and metadata about the generation process.

    Attributes:
        id: Unique identifier for the message (UUID string).
        session_id: Foreign key to the parent chat session.
        role: Role of the message sender (user, assistant, system).
        message_type: Type of message content.
        content: The actual message content (text, script, etc.).
        metadata_json: Flexible JSON field for additional message data.
        created_at: Timestamp when the message was created.
        session: Relationship to the parent chat session.
    """

    __tablename__ = "chat_messages"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False,
        comment="Unique message identifier (UUID v4)",
    )
    session_id = Column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to parent chat session",
    )
    role = Column(
        SQLEnum(MessageRole),
        nullable=False,
        comment="Role of the message sender",
    )
    message_type = Column(
        SQLEnum(MessageType),
        default=MessageType.TEXT,
        nullable=False,
        comment="Type of message content",
    )
    content = Column(
        SQLiteText().with_variant(Text, "sqlite"),
        nullable=False,
        comment="Message content (text, script, etc.)",
    )
    metadata_json = Column(
        JSON,
        default=dict,
        nullable=True,
        comment="Flexible JSON field for additional metadata",
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="Timestamp when message was created",
    )

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    @validates("role")
    def validate_role(self, key: str, value: MessageRole) -> MessageRole:
        """Validate the message role."""
        if not isinstance(value, MessageRole):
            raise ValueError(f"Invalid role: {value}")
        return value

    @validates("message_type")
    def validate_message_type(self, key: str, value: MessageType) -> MessageType:
        """Validate the message type."""
        if not isinstance(value, MessageType):
            raise ValueError(f"Invalid message type: {value}")
        return value

    @validates("content")
    def validate_content(self, key: str, value: str) -> str:
        """Validate and sanitize message content."""
        if not isinstance(value, str):
            raise ValueError("Content must be a string")
        
        # Remove any null bytes and strip whitespace
        value = value.replace("\x00", "").strip()
        
        if not value:
            raise ValueError("Content cannot be empty")
        
        # Limit content length to prevent abuse (adjust as needed)
        max_length = 100000  # 100k characters
        if len(value) > max_length:
            raise ValueError(f"Content exceeds maximum length of {max_length} characters")
        
        return value

    def to_dict(self) -> dict:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value if self.role else None,
            "message_type": self.message_type.value if self.message_type else None,
            "content": self.content,
            "metadata": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "session_title": self.session.title if self.session else None,
            "session_status": self.session.status.value if self.session else None,
        }

    def __repr__(self) -> str:
        """String representation of the ChatMessage."""
        return (
            f"<ChatMessage(id='{self.id}', "
            f"session_id='{self.session_id}', "
            f"role='{self.role.value if self.role else None}', "
            f"type='{self.message_type.value if self.message_type else None}', "
            f"created_at='{self.created_at}')>"
        )


def init_db(database_url: str = "sqlite:///./video_generator.db") -> Session:
    """
    Initialize the database and create all tables.

    Args:
        database_url: SQLAlchemy database URL (defaults to SQLite).

    Returns:
        SQLAlchemy Session instance.

    Raises:
        ValueError: If database_url is invalid.
        Exception: If database initialization fails.
    """
    try:
        if not database_url:
            raise ValueError("Database URL cannot be empty")

        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
            echo=False,  # Set to True for SQL logging in development
            pool_pre_ping=True,  # Verify connections before using them
            pool_recycle=3600,  # Recycle connections after 1 hour
        )

        # Create all tables
        Base.metadata.create_all(engine)

        # Create a session factory
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        return SessionLocal()

    except Exception as e:
        raise Exception(f"Failed to initialize database: {str(e)}") from e


def get_session(db_session: Session, session_id: str) -> Optional[ChatSession]:
    """
    Retrieve a chat session by its ID.

    Args:
        db_session: SQLAlchemy database session.
        session_id: UUID string of the session to retrieve.

    Returns:
        ChatSession instance if found, None otherwise.

    Raises:
        ValueError: If session_id is invalid.
    """
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Invalid session ID")

    try:
        return db_session.query(ChatSession).filter(ChatSession.id == session_id).first()
    except Exception as e:
        raise Exception(f"Error retrieving session {session_id}: {str(e)}") from e


def get_session_messages(
    db_session: Session, session_id: str, limit: int = 100, offset: int = 0
) -> List[ChatMessage]:
    """
    Retrieve messages for a specific chat session with pagination.

    Args:
        db_session: SQLAlchemy database session.
        session_id: UUID string of the session.
        limit: Maximum number of messages to return (default 100).
        offset: Number of messages to skip (default 0).

    Returns:
        List of ChatMessage instances.

    Raises:
        ValueError: If parameters are invalid.
    """
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Invalid session ID")
    
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000")
    
    if offset < 0:
        raise ValueError("Offset must be non-negative")

    try:
        return (
            db_session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .offset(offset)
            .limit(limit)
            .all()
        )
    except Exception as e:
        raise Exception(f"Error retrieving messages for session {session_id}: {str(e)}") from e


def create_session(db_session: Session, title: Optional[str] = None) -> ChatSession:
    """
    Create a new chat session.

    Args:
        db_session: SQLAlchemy database session.
        title: Optional title for the session.

    Returns:
       Newly created ChatSession instance.

   Raises:
       Exception: If session creation fails.
   """
   try:
       new_session = ChatSession(title=title)
       db_session.add(new_session)
       db_session.commit()
       db_session.refresh(new_session)
       return new_session
   except Exception as e:
       db_session.rollback()
       raise Exception(f"Failed to create chat session: {str(e)}") from e


def add_message(
   db_session: Session,
   session_id: str,
   role: MessageRole,
   content: str,
   message_type: MessageType = MessageType.TEXT,
   metadata_json: Optional[dict] = None,
) -> ChatMessage:
   """
   Add a new message to an existing chat session.

   Args:
       db_session: SQLAlchemy database session.
       session_id: UUID string of the target session.
       role: Role of the message sender.
       content: Message content text.
       message_type: Type of message (default TEXT).
       metadata_json: Optional metadata dictionary.

   Returns:
       Newly created ChatMessage instance.

   Raises:
       ValueError: If parameters are invalid.
       Exception: If message creation fails or session not found.
   """
   # Validate inputs
   if not isinstance(role, MessageRole):
       raise ValueError("Invalid role type")
   
   if not isinstance(message_type, MessageType):
       raise ValueError("Invalid message type")
   
   if not content or not isinstance(content, str):
       raise ValueError("Content must be a non-empty string")

   # Verify session exists
   session = get_session(db_session, session_id)
   if not session:
       raise ValueError(f"Session {session_id} not found")

   try:
       new_message = ChatMessage(
           session_id=session_id,
           role=role,
           message_type=message_type,
           content=content.strip(),
           metadata_json=metadata_json or {},
       )
       
       # Update parent session's updated_at timestamp
       session.updated_at = datetime.utcnow()
       
       db_session.add(new_message)
       db_session.commit()
       db_session.refresh(new_message)
       
       return new_message
   except Exception as e:
       db_session.rollback()
       raise Exception(f"Failed to add message to session {session_id}: {str(e)}") from e


def update_session_status(
   db_session: Session, 
   session_id: str, 
   status: SessionStatus
) -> Optional[ChatSession]:
   """
   Update the status of a chat session.

   Args:
       db_session: SQLAlchemy database session.
       session_id: UUID string of the target session.
       status: New status for the session.

   Returns:
       Updated ChatSession instance or None if not found.

   Raises:
       ValueError: If parameters are invalid.
       Exception: If update fails.
   """
   if not isinstance(status, SessionStatus):
       raise ValueError("Invalid status type")

   try:
       session = get_session(db_session, session_id)
       if not session:
           return None

       # Set completed timestamp if completing/cancelling
       current_time = datetime.utcnow()
       
       if status in (SessionStatus.COMPLETED, SessionStatus.CANCELLED):
           if not session.completed_at:
               session.completed_at = current_time
       
       elif status == SessionStatus.ACTIVE and session.completed_at:
           # Reactivating a completed session - clear completion timestamp
           session.completed_at = None

       session.status = status
       db_session.commit()
       db_session.refresh(session)
       
       return session
   except Exception as e:
       db_session.rollback()
       raise Exception(f"Failed to update status for session {session_id}: {str(e)}") from e


def delete_session(db_session: Session, session_id: str) -> bool:
   """
   Delete a chat session and all its messages.

   Args:
       db_session: SQLAlchemy database session.
       session_id: UUID string of the target session.

   Returns:
       True if deleted successfully, False if not found.

   Raises:
       Exception: If deletion fails.
   """
   try:
       # Use cascade delete - messages will be automatically removed
       rows_deleted = (
           db_session.query(ChatSession)
           .filter(ChatSession.id == session_id)
           .delete()
       )
       
       db_session.commit()
       
       return rows_deleted > 0
   except Exception as e:
       db_session.rollback()
       raise Exception(f"Failed to delete session {session_id}: {str(e)}") from e


def get_active_sessions(db_session: Session) -> List[ChatSession]:
   """
   Retrieve all active chat sessions.

   Args:
       db_session: SQLAlchemy database session.

   Returns:
       List of active ChatSession instances ordered by creation date (newest first).
   """
   try:
       return (
           db_session.query(ChatSession)
           .filter(ChatSession.status == SessionStatus.ACTIVE)
           .order_by(ChatSession.created_at.desc())
           .all()
       )
   except Exception as e:
       raise Exception(f"Error retrieving active sessions: {str(e)}") from e


def cleanup_old_sessions(
   db_session: Session, 
   days_old: int = 30
) -> int:
   """
   Delete sessions older than specified number of days.

   Args:
       db_session: SQLAlchemy database session.
       days_old: Age threshold in days (default 30).

   Returns:
       Number of deleted sessions.

   Raises:
       ValueError: If days_old is invalid.
   """
   if days_old < 1 or days_old > 365:
       raise ValueError("Days old must be between 1 and 365")

   try:
       cutoff_date = datetime.utcnow().replace(
           hour=0, minute=0, second=0, microsecond=0
       )
       
       from datetime import timedelta
       cutoff_date -= timedelta(days=days_old)

       rows_deleted = (
           db_session.query(ChatSession)
           .filter(ChatSession.created_at < cutoff_date)
           .delete(synchronize_session='fetch')
       )
       
       db_session.commit()
       
       return rows_deleted
   except Exception as e:
       db_session.rollback()
       raise Exception(f"Error cleaning up old sessions: {str(e)}") from e


# Export all models and utility functions for use in other modules
__all__ = [
   "Base",
   "ChatSession",
   "ChatMessage",
   "MessageRole",
   "MessageType",
   "SessionStatus",
   "init_db",
   "get_session",
   "get_session_messages",
   "create_session",
   "add_message",
   "update_session_status",
   "delete_session",
   "get_active_sessions",
   "cleanup_old_sessions",
]