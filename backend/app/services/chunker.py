"""
Semantic and fallback text splitting for YouTube transcripts.
Preserves timestamp metadata for each chunk.
"""

import logging
from typing import Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings
from app.utils.timestamps import seconds_to_display

logger = logging.getLogger(__name__)


def create_timestamped_documents(
    transcript_entries: list[dict],
    video_id: str,
    video_title: str = "",
    channel: str = "",
) -> list[Document]:
    """
    Convert raw transcript entries into LangChain Documents
    with timestamp metadata.

    Each entry becomes a small document to be later split/merged
    by the text splitter.
    """
    documents = []
    for i, entry in enumerate(transcript_entries):
        text = entry.get("text", "").strip()
        start = entry.get("start", 0.0)
        duration = entry.get("duration", 0.0)

        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "video_id": video_id,
                "video_title": video_title,
                "channel": channel,
                "start_time": start,
                "end_time": start + duration,
                "start_display": seconds_to_display(start),
                "end_display": seconds_to_display(start + duration),
                "chunk_index": i,
                "source": f"youtube:{video_id}",
            },
        )
        documents.append(doc)

    return documents


def chunk_transcript(
    transcript_entries: list[dict],
    video_id: str,
    video_title: str = "",
    channel: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Split transcript into semantically coherent chunks.

    Uses RecursiveCharacterTextSplitter with time-window awareness:
    groups transcript entries into overlapping windows, then splits
    each window to maintain context.

    Each resulting chunk has metadata with start/end timestamps.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    if not transcript_entries:
        return []

    # ── Step 1: Group entries into time-based windows ──
    # This preserves natural speech flow better than arbitrary splitting
    windows = _create_time_windows(transcript_entries, window_seconds=60)

    # ── Step 2: Create documents from windows ──
    window_docs = []
    for window in windows:
        text = " ".join(e.get("text", "") for e in window).strip()
        if not text:
            continue

        start_time = window[0].get("start", 0.0)
        last_entry = window[-1]
        end_time = last_entry.get("start", 0.0) + last_entry.get("duration", 0.0)

        doc = Document(
            page_content=text,
            metadata={
                "video_id": video_id,
                "video_title": video_title,
                "channel": channel,
                "start_time": start_time,
                "end_time": end_time,
                "start_display": seconds_to_display(start_time),
                "end_display": seconds_to_display(end_time),
                "source": f"youtube:{video_id}",
            },
        )
        window_docs.append(doc)

    # ── Step 3: Split windows with RecursiveCharacterTextSplitter ──
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
        length_function=len,
    )

    final_chunks = []
    for doc in window_docs:
        splits = text_splitter.split_documents([doc])
        for j, split in enumerate(splits):
            # Inherit timestamp metadata from parent window
            split.metadata["chunk_index"] = len(final_chunks)
            split.metadata["window_sub_index"] = j
            final_chunks.append(split)

    logger.info(
        f"Chunked video {video_id}: {len(transcript_entries)} entries "
        f"→ {len(windows)} windows → {len(final_chunks)} chunks"
    )

    return final_chunks


def _create_time_windows(
    entries: list[dict],
    window_seconds: float = 60,
    overlap_seconds: float = 10,
) -> list[list[dict]]:
    """
    Group transcript entries into overlapping time windows.
    This creates natural topical boundaries.
    """
    if not entries:
        return []

    windows = []
    current_window = []
    window_start = entries[0].get("start", 0.0)

    for entry in entries:
        entry_start = entry.get("start", 0.0)

        # If this entry is beyond the current window, start a new one
        if entry_start - window_start >= window_seconds and current_window:
            windows.append(current_window)

            # Start new window with overlap
            overlap_start = entry_start - overlap_seconds
            current_window = [
                e for e in current_window
                if e.get("start", 0.0) >= overlap_start
            ]
            window_start = current_window[0].get("start", 0.0) if current_window else entry_start

        current_window.append(entry)

    # Don't forget the last window
    if current_window:
        windows.append(current_window)

    return windows
