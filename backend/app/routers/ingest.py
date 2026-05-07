"""
Ingest router — handles YouTube video transcript ingestion.
"""

import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import IngestRequest, IngestResponse, IngestStatusResponse, IngestStatus
from app.services.transcript import (
    extract_video_id,
    fetch_video_metadata,
    fetch_transcript,
    merge_transcript_entries,
    compute_video_duration,
)
from app.services.chunker import chunk_transcript
from app.services.vectorstore import upsert_documents, check_video_indexed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# In-memory metadata cache (video_id → metadata)
_video_metadata_cache: dict[str, dict] = {}


@router.post("", response_model=IngestResponse)
async def ingest_video(request: IngestRequest):
    """
    Ingest a YouTube video: fetch transcript, chunk, embed, and store.
    """
    # Extract video ID
    video_id = extract_video_id(request.video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        # Check if already indexed
        index_status = check_video_indexed(video_id)
        if index_status.get("indexed") and index_status.get("chunk_count", 0) > 0:
            cached = _video_metadata_cache.get(video_id, {})
            return IngestResponse(
                video_id=video_id,
                title=cached.get("title", "Video"),
                channel=cached.get("channel", ""),
                duration_seconds=cached.get("duration", 0),
                chunk_count=index_status["chunk_count"],
                status=IngestStatus.COMPLETED,
            )

        # Fetch metadata
        metadata = await fetch_video_metadata(video_id)
        title = metadata.get("title", "Unknown Video")
        channel = metadata.get("channel", "Unknown Channel")

        # Fetch transcript
        logger.info(f"Fetching transcript for video: {video_id}")
        raw_entries = fetch_transcript(video_id)

        if not raw_entries:
            raise HTTPException(
                status_code=404,
                detail="No transcript available for this video"
            )

        # Merge short entries
        merged_entries = merge_transcript_entries(raw_entries)
        duration = compute_video_duration(raw_entries)

        # Chunk with timestamp metadata
        logger.info(f"Chunking {len(merged_entries)} transcript entries")
        chunks = chunk_transcript(
            transcript_entries=raw_entries,  # Use raw for better timestamp accuracy
            video_id=video_id,
            video_title=title,
            channel=channel,
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Transcript fetched but produced 0 chunks; cannot index this video.",
            )

        # Upsert to Pinecone
        logger.info(f"Upserting {len(chunks)} chunks to Pinecone")
        count = await upsert_documents(chunks, video_id)

        if count <= 0:
            raise HTTPException(
                status_code=500,
                detail="Index upsert returned 0 vectors; indexing did not complete.",
            )

        # Cache metadata
        _video_metadata_cache[video_id] = {
            "title": title,
            "channel": channel,
            "duration": duration,
            "chunk_count": count,
        }

        return IngestResponse(
            video_id=video_id,
            title=title,
            channel=channel,
            duration_seconds=duration,
            chunk_count=count,
            status=IngestStatus.COMPLETED,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed for {video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/status/{video_id}", response_model=IngestStatusResponse)
async def get_ingest_status(video_id: str):
    """Check if a video has been indexed."""
    try:
        status = check_video_indexed(video_id)
        cached = _video_metadata_cache.get(video_id, {})

        if status.get("indexed"):
            return IngestStatusResponse(
                video_id=video_id,
                status=IngestStatus.COMPLETED,
                title=cached.get("title", ""),
                chunk_count=status.get("chunk_count", 0),
            )
        else:
            return IngestStatusResponse(
                video_id=video_id,
                status=IngestStatus.NOT_FOUND,
            )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return IngestStatusResponse(
            video_id=video_id,
            status=IngestStatus.NOT_FOUND,
        )
