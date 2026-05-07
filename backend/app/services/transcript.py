"""app.services.transcript

YouTube transcript fetching and metadata extraction.

Primary backend: `youtube-transcript-api`.
Fallback backend: `yt-dlp` (parses caption VTT URLs) for cases where
YouTubeTranscriptApi returns empty/blocked XML (e.g. "no element found").
"""

import re
import logging
import os
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi
import httpx

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch video metadata using YouTube's oEmbed API.
    Returns title, channel name, and thumbnail URL.
    """
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(oembed_url)
            response.raise_for_status()
            data = response.json()
            return {
                "title": data.get("title", "Unknown Video"),
                "channel": data.get("author_name", "Unknown Channel"),
                "thumbnail_url": data.get("thumbnail_url", ""),
            }
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {video_id}: {e}")
        return {
            "title": "Unknown Video",
            "channel": "Unknown Channel",
            "thumbnail_url": "",
        }


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetch the transcript for a YouTube video.

    Returns a list of transcript entries:
    [{"text": "...", "start": 0.0, "duration": 3.5}, ...]
    """
    def _normalize_entries(entries: list) -> list[dict]:
        out: list[dict] = []
        for entry in entries:
            if isinstance(entry, dict):
                out.append(
                    {
                        "text": entry.get("text", ""),
                        "start": float(entry.get("start", 0.0) or 0.0),
                        "duration": float(entry.get("duration", 0.0) or 0.0),
                    }
                )
            else:
                out.append(
                    {
                        "text": getattr(entry, "text", str(entry)),
                        "start": float(getattr(entry, "start", 0.0) or 0.0),
                        "duration": float(getattr(entry, "duration", 0.0) or 0.0),
                    }
                )
        return out

    try:
        transcript = None
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Prefer English varieties, then Hindi
            preferred_langs = ["en", "en-US", "en-GB", "en-IN", "hi"]

            try:
                transcript = transcript_list.find_manually_created_transcript(preferred_langs)
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(preferred_langs)
                except Exception:
                    # Try any available language and translate
                    for t in transcript_list:
                        try:
                            transcript = t.translate("en")
                            break
                        except Exception:
                            continue
                    else:
                        raise ValueError(f"No transcript available for video {video_id}")
            
            # If we found a Hindi transcript (or any non-en), translate it into English
            if transcript is not None and not transcript.language_code.startswith("en"):
                try:
                    # Use YouTube's built-in translation to English
                    transcript = transcript.translate("en")
                except Exception as e:
                    logger.warning(f"Could not translate transcript to english: {e}")

        except Exception as list_err:
            logger.warning(
                "Transcript list failed for %s, will attempt direct transcript fetch: %s",
                video_id,
                list_err,
            )

        if transcript is not None:
            entries = transcript.fetch()
        else:
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi"])

        normalized = _normalize_entries(entries)
        if normalized:
            return normalized

        raise ValueError("Empty transcript response")

    except Exception as e:
        # Fallback: yt-dlp subtitles / auto-captions (VTT)
        try:
            logger.warning(
                "youtube-transcript-api failed for %s (%s); trying yt-dlp fallback",
                video_id,
                e,
            )
            return _fetch_transcript_with_ytdlp(video_id)
        except Exception as ytdlp_err:
            logger.error(f"Failed to fetch transcript for {video_id}: {ytdlp_err}")
            raise ValueError(f"Could not fetch transcript: {str(e)}")


def _fetch_transcript_with_ytdlp(video_id: str) -> list[dict]:
    """Fetch transcript using yt-dlp caption URLs and parse WebVTT into entries."""
    try:
        from yt_dlp import YoutubeDL
    except Exception as e:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with `pip install yt-dlp` to enable transcript fallback."
        ) from e

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }

    # Optional: allow using browser cookies for restricted captions.
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cookies_from_browser:
        # Examples: "chrome", "edge", "firefox"
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    cookiefile = os.getenv("YTDLP_COOKIEFILE", "").strip()
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    vtt_url = _pick_caption_url(info)
    if not vtt_url:
        raise ValueError("No captions/auto-captions URL found")

    resp = httpx.get(vtt_url, timeout=20)
    resp.raise_for_status()
    vtt_text = resp.text or ""
    entries = _parse_vtt_to_entries(vtt_text)
    if not entries:
        raise ValueError("Empty captions after parsing")
    return entries


def _pick_caption_url(info: dict) -> Optional[str]:
    """Pick a VTT caption URL from yt-dlp info dict."""
    def _candidates(subs: dict) -> list[dict]:
        # Prefer exact 'en', otherwise any key starting with 'en'
        if not subs:
            return []
        if "en" in subs:
            return subs.get("en") or []
        for k in subs.keys():
            if str(k).lower().startswith("en"):
                return subs.get(k) or []
        return []

    for bucket_name in ("subtitles", "automatic_captions"):
        bucket = info.get(bucket_name) or {}
        fmts = _candidates(bucket)
        if not fmts:
            continue

        # Prefer vtt, then any format url.
        vtt = next((f for f in fmts if str(f.get("ext", "")).lower() == "vtt" and f.get("url")), None)
        if vtt:
            return vtt.get("url")
        any_url = next((f for f in fmts if f.get("url")), None)
        if any_url:
            return any_url.get("url")
    return None


def _parse_vtt_timestamp(ts: str) -> float:
    """Parse VTT timestamp like HH:MM:SS.mmm or MM:SS.mmm into seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        raise ValueError(f"Invalid VTT timestamp: {ts}")

    if "." in s:
        sec, ms = s.split(".", 1)
        s_val = int(sec)
        ms_val = int((ms + "000")[:3])
    else:
        s_val = int(s)
        ms_val = 0

    return int(h) * 3600 + int(m) * 60 + s_val + (ms_val / 1000.0)


def _parse_vtt_to_entries(vtt_text: str) -> list[dict]:
    """Very small WebVTT parser to transcript entries."""
    lines = [ln.rstrip("\n") for ln in (vtt_text or "").splitlines()]
    entries: list[dict] = []
    i = 0

    # Skip header
    while i < len(lines) and (not lines[i].strip() or lines[i].strip().upper().startswith("WEBVTT")):
        i += 1

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Optional cue identifier line (non-timestamp)
        if "-->" not in line and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        timing = line
        start_raw, rest = timing.split("-->", 1)
        end_raw = rest.strip().split(" ", 1)[0]
        try:
            start_s = _parse_vtt_timestamp(start_raw)
            end_s = _parse_vtt_timestamp(end_raw)
        except Exception:
            i += 1
            continue

        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            txt = lines[i].strip()
            # Strip HTML-ish tags
            txt = re.sub(r"<[^>]+>", "", txt)
            text_lines.append(txt)
            i += 1

        text = " ".join(t for t in text_lines if t).strip()
        if text:
            entries.append(
                {
                    "text": text,
                    "start": float(start_s),
                    "duration": float(max(0.0, end_s - start_s)),
                }
            )

        i += 1

    # Deduplicate consecutive identical captions (common in auto-captions)
    deduped: list[dict] = []
    last_text = None
    for e in entries:
        if e["text"] == last_text:
            continue
        deduped.append(e)
        last_text = e["text"]
    return deduped


def merge_transcript_entries(
    entries: list[dict],
    min_chars: int = 50,
) -> list[dict]:
    """
    Merge short transcript entries into longer coherent segments
    while preserving timestamp mapping.

    Each merged entry keeps the start time of the first sub-entry
    and the cumulative duration.
    """
    if not entries:
        return []

    merged = []
    current_text = ""
    current_start = entries[0].get("start", 0.0)
    current_duration = 0.0

    for entry in entries:
        text = entry.get("text", "").strip()
        start = entry.get("start", 0.0)
        duration = entry.get("duration", 0.0)

        if not text:
            continue

        if current_text:
            current_text += " " + text
        else:
            current_text = text
            current_start = start

        current_duration = (start + duration) - current_start

        # Flush when we have enough content and hit a sentence boundary
        if len(current_text) >= min_chars and current_text[-1] in '.?!':
            merged.append({
                "text": current_text.strip(),
                "start": current_start,
                "duration": current_duration,
            })
            current_text = ""
            current_duration = 0.0

    # Don't forget the last segment
    if current_text.strip():
        merged.append({
            "text": current_text.strip(),
            "start": current_start,
            "duration": current_duration,
        })

    return merged


def get_full_transcript_text(entries: list[dict]) -> str:
    """Combine all transcript entries into a single text string."""
    return " ".join(entry.get("text", "") for entry in entries).strip()


def compute_video_duration(entries: list[dict]) -> float:
    """Compute total video duration from transcript entries."""
    if not entries:
        return 0.0
    last = entries[-1]
    return last.get("start", 0.0) + last.get("duration", 0.0)
