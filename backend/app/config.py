"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _BACKEND_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # API Keys
    OPENROUTER_API_KEY: str
    PINECONE_API_KEY: str
    COHERE_API_KEY: str = ""

    # Pinecone
    PINECONE_INDEX_NAME: str = "yt-chatbot"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Model Configuration
    # NOTE: Must be a valid OpenRouter model id. See: https://openrouter.ai/api/v1/models
    LLM_MODEL: str = "openai/gpt-oss-20b:free"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: int = 384 # Changed to match all-MiniLM-L6-v2

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEMANTIC_CHUNKING: bool = True

    # Retrieval
    RETRIEVAL_K: int = 20
    RETRIEVAL_TOP_N: int = 5
    MMR_LAMBDA: float = 0.7
    MMR_FETCH_K: int = 50

    # CORS
    BACKEND_CORS_ORIGINS: str = "chrome-extension://*,http://localhost:5173,http://localhost:3000"

    # Session
    SESSION_TTL_SECONDS: int = 3600  # 1 hour
    SESSION_STORE_BACKEND: str = "auto"  # auto | redis | sqlite | memory
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_SQLITE_PATH: str = "data/sessions.db"

    # Sparse/keyword retrieval
    KEYWORD_INDEX_SQLITE_PATH: str = "data/keyword_index.db"

    # Guardrails
    EVIDENCE_CONFIDENCE_THRESHOLD: float = 0.5

    class Config:
        # Always load backend/.env regardless of current working directory.
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
