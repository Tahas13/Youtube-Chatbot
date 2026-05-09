"""
Chat router — handles conversation with the RAG agent.
"""

import logging
import uuid
import json
import asyncio
import re
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, SessionDeleteResponse, QueryType
from app.rag.graph import run_agent
from app.services.memory import get_session_store
from app.services.vectorstore import check_video_indexed
from app.rag.retriever import full_retrieval_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# Metadata cache reference (shared with ingest router at app level)
_video_metadata_cache: dict[str, dict] = {}


def set_metadata_cache(cache: dict):
    """Set the shared metadata cache reference."""
    global _video_metadata_cache
    _video_metadata_cache = cache


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the RAG chatbot about a specific video.
    """
    video_id = request.video_id
    message = request.message.strip()
    session_id = request.session_id or str(uuid.uuid4())

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID is required")

    index_status = check_video_indexed(video_id)
    chunk_count = int(index_status.get("chunk_count", 0) or 0)
    if not index_status.get("indexed") or chunk_count <= 0:
        return ChatResponse(
            answer=(
                "This video hasn't been indexed yet (no transcript chunks found). "
                "Click 'Re-index video' in the header and wait for indexing to finish, then ask again."
            ),
            citations=[],
            suggested_questions=["Re-index this video", "Try a different video"],
            query_type=QueryType.SEARCH,
        )

    try:
        # Get session store and history
        store = get_session_store()
        chat_history = store.get_context_window(session_id, max_messages=10)

        # Get video title from cache
        cached = _video_metadata_cache.get(video_id, {})
        video_title = cached.get("title", "")

        # Add user message to history
        store.add_message(session_id, "user", message)

        # Run the RAG agent
        logger.info(f"Processing chat for video={video_id}, session={session_id[:8]} - handler enter")
        started = asyncio.get_event_loop().time()
        result = await run_agent(
            query=message,
            video_id=video_id,
            video_title=video_title,
            session_id=session_id,
            chat_history=chat_history,
        )
        finished = asyncio.get_event_loop().time()
        logger.info(
            f"Processing chat for video={video_id}, session={session_id[:8]} - agent finished in {finished-started:.2f}s"
        )

        # Store assistant response in history
        answer = result.get("answer", "")
        store.add_message(session_id, "assistant", answer)

        # Build response
        logger.info(f"Returning chat response for session={session_id[:8]}")
        return ChatResponse(
            answer=answer,
            citations=result.get("citations", []),
            suggested_questions=result.get("suggested_questions", []),
            query_type=QueryType(result.get("query_type", "search")),
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/extractive", response_model=ChatResponse)
async def chat_extractive(request: ChatRequest):
    """Quick extractive-only fallback: returns extractive summary from retrieved docs without calling the LLM."""
    video_id = request.video_id
    message = request.message.strip()
    session_id = request.session_id or str(uuid.uuid4())

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID is required")

    index_status = check_video_indexed(video_id)
    if not index_status.get("indexed"):
        return ChatResponse(
            answer=(
                "This video hasn't been indexed yet (no transcript chunks found). "
                "Click 'Re-index video' in the header and wait for indexing to finish, then ask again."
            ),
            citations=[],
            suggested_questions=["Re-index this video", "Try a different video"],
            query_type=QueryType.SEARCH,
        )

    # Detect simple time-window requests
    m = re.search(r"first\s+(\d+)\s*(?:min|minute|minutes)", message, flags=re.I)
    time_window = None
    if m:
        mins = int(m.group(1))
        time_window = (0.0, float(mins * 60))

    try:
        docs = await full_retrieval_pipeline([message], video_id, query_type="search", domain="other", time_window=time_window)
        # Build extractive answer using nodes helper
        from app.rag.nodes import _extractive_summary_from_docs

        answer = _extractive_summary_from_docs(docs, max_bullets=6, video_title="")
        logger.info(f"Returning extractive fallback response for session={session_id[:8]}")
        return ChatResponse(answer=answer, citations=[], suggested_questions=[], query_type=QueryType.SEARCH)
    except Exception as e:
        logger.error(f"Extractive fallback failed: {e}")
        raise HTTPException(status_code=500, detail="Extractive fallback failed")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_comment(text: str) -> str:
    # SSE comments (lines starting with ':') are ignored by clients,
    # but help force intermediate flushes through some buffers.
    return f": {text}\n\n"


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint (SSE) for extension/web clients.
    Keeps /api/chat behavior unchanged while supporting incremental output.
    """
    video_id = request.video_id
    message = request.message.strip()
    session_id = request.session_id or str(uuid.uuid4())

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID is required")

    index_status = check_video_indexed(video_id)
    chunk_count = int(index_status.get("chunk_count", 0) or 0)
    if not index_status.get("indexed") or chunk_count <= 0:
        async def not_indexed_stream() -> AsyncGenerator[str, None]:
            yield _sse("meta", {"session_id": session_id})
            yield _sse_comment("flush")
            await asyncio.sleep(0)
            answer = (
                "This video hasn't been indexed yet (no transcript chunks found). "
                "Click 'Re-index video' in the header and wait for indexing to finish, then ask again."
            )
            yield _sse("chunk", {"text": answer})
            yield _sse(
                "final",
                {
                    "answer": answer,
                    "citations": [],
                    "suggested_questions": ["Re-index this video", "Try a different video"],
                    "query_type": "search",
                },
            )

        return StreamingResponse(
            not_indexed_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    store = get_session_store()
    chat_history = store.get_context_window(session_id, max_messages=10)
    cached = _video_metadata_cache.get(video_id, {})
    video_title = cached.get("title", "")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            store.add_message(session_id, "user", message)
            yield _sse("meta", {"session_id": session_id})
            yield _sse_comment("flush")
            await asyncio.sleep(0)
            logger.info(f"Streaming chat handler enter for session={session_id[:8]}")
            started = asyncio.get_event_loop().time()
            result = await run_agent(
                query=message,
                video_id=video_id,
                video_title=video_title,
                session_id=session_id,
                chat_history=chat_history,
            )
            finished = asyncio.get_event_loop().time()
            logger.info(
                f"Streaming chat agent finished for session={session_id[:8]} in {finished-started:.2f}s"
            )
            answer = result.get("answer", "")
            store.add_message(session_id, "assistant", answer)

            # Stream the final answer in chunks for UI rendering compatibility.
            chunk_size = 15
            for i in range(0, len(answer), chunk_size):
                yield _sse("chunk", {"text": answer[i:i + chunk_size]})
                await asyncio.sleep(0.02) # Adds a nice typewriter effect

            final_payload = {
                "answer": answer,
                "citations": [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in result.get("citations", [])
                ],
                "suggested_questions": result.get("suggested_questions", []),
                "query_type": result.get("query_type", "search"),
            }
            yield _sse("final", final_payload)
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}", exc_info=True)
            yield _sse("error", {"message": "Chat processing failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/session/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    """Clear a chat session's memory."""
    store = get_session_store()
    store.clear_session(session_id)

    return SessionDeleteResponse(
        session_id=session_id,
        status="deleted",
    )
