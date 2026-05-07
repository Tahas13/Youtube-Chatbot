"""
Timestamp parsing and formatting utilities.
"""

import re
from typing import Optional


def seconds_to_display(seconds: float) -> str:
    """Convert seconds to human-readable timestamp (HH:MM:SS or MM:SS)."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def display_to_seconds(display: str) -> Optional[float]:
    """Convert a timestamp string (HH:MM:SS or MM:SS) to seconds."""
    parts = display.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except (ValueError, IndexError):
        return None
    return None


def extract_timestamps_from_text(text: str) -> list[dict]:
    """
    Extract timestamp references from LLM-generated text.
    Matches patterns like [05:23], [1:02:30], [timestamp: 05:23], etc.
    """
    # Match [HH:MM:SS] or [MM:SS] patterns
    pattern = r'\[(?:timestamp:\s*)?(\d{1,2}:\d{2}(?::\d{2})?)\]'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    timestamps = []
    for match in matches:
        ts_display = match.group(1)
        ts_seconds = display_to_seconds(ts_display)
        if ts_seconds is not None:
            timestamps.append({
                "display": ts_display,
                "seconds": ts_seconds,
                "original": match.group(0),
            })

    return timestamps


def find_timestamp_for_chunk(
    chunk_text: str,
    transcript_entries: list[dict],
) -> tuple[float, float]:
    """
    Find the start and end timestamps for a chunk by matching it
    against the raw transcript entries.

    Returns (start_seconds, end_seconds).
    """
    if not transcript_entries:
        return (0.0, 0.0)

    # Simple approach: find the first transcript entry whose text
    # appears in the chunk
    chunk_lower = chunk_text.lower()
    start_time = None
    end_time = 0.0

    for entry in transcript_entries:
        entry_text = entry.get("text", "").lower().strip()
        if entry_text and entry_text in chunk_lower:
            entry_start = entry.get("start", 0.0)
            entry_duration = entry.get("duration", 0.0)
            if start_time is None:
                start_time = entry_start
            end_time = entry_start + entry_duration

    return (start_time or 0.0, end_time)
