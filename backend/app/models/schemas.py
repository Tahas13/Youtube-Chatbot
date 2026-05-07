"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class QueryType(str, Enum):
    SEARCH = "search"
    SUMMARIZE = "summarize"
    CLARIFY = "clarify"
    CONTEXTUALIZE = "contextualize"


class IngestStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"


# ──────────────────────────────────────────────
# Ingest
# ──────────────────────────────────────────────

class IngestRequest(BaseModel):
    video_url: str = Field(..., description="YouTube video URL")


class IngestResponse(BaseModel):
    video_id: str
    title: str
    channel: str = ""
    duration_seconds: float = 0
    chunk_count: int
    status: IngestStatus


class IngestStatusResponse(BaseModel):
    video_id: str
    status: IngestStatus
    title: str = ""
    chunk_count: int = 0


# ──────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────

class Citation(BaseModel):
    text: str = Field(..., description="Relevant excerpt from the transcript")
    timestamp_seconds: float = Field(..., description="Start time in seconds")
    timestamp_display: str = Field(..., description="Human-readable timestamp like '05:23'")


class ChatRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    message: str = Field(..., description="User's message/question")
    session_id: str = Field(..., description="Unique session identifier")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Bot's response grounded in the transcript")
    citations: list[Citation] = Field(default_factory=list, description="Timestamp citations")
    suggested_questions: list[str] = Field(
        default_factory=list,
        description="Follow-up question suggestions"
    )
    query_type: QueryType = Field(default=QueryType.SEARCH, description="Detected query intent")


class SessionDeleteResponse(BaseModel):
    session_id: str
    status: str = "deleted"


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
