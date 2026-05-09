"""
FastAPI application entry point.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.routers import ingest, chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("🚀 YouTube Chatbot API starting up...")

    # Share metadata cache between routers
    from app.routers.ingest import _video_metadata_cache
    chat.set_metadata_cache(_video_metadata_cache)

    # Pre-initialize Qdrant collection
    try:
        from app.services.vectorstore import ensure_index_exists
        await asyncio.wait_for(asyncio.to_thread(ensure_index_exists), timeout=10)
        logger.info("✅ Qdrant collection ready")
    except Exception as e:
        logger.warning(f"⚠️ Qdrant initialization deferred: {e}")

    yield

    logger.info("👋 YouTube Chatbot API shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="YouTube AI Chatbot API",
        description="RAG-powered chatbot for YouTube videos",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    origins = [
        origin.strip()
        for origin in settings.BACKEND_CORS_ORIGINS.split(",")
        if origin.strip()
    ]
    # Also allow any chrome-extension origin
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"chrome-extension://.*",
        allow_origins=origins + ["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(ingest.router)
    app.include_router(chat.router)

    # Health check
    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        return HealthResponse(status="ok", version="1.0.0")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
