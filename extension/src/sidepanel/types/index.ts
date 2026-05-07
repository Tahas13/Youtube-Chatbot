/* ── Types for the YouTube Chatbot Extension ── */

export interface Citation {
  text: string;
  timestamp_seconds: number;
  timestamp_display: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  suggested_questions?: string[];
  timestamp: number;
}

export interface VideoInfo {
  videoId: string;
  title: string;
  channel: string;
  thumbnailUrl: string;
  duration: number;
}

export interface IngestResponse {
  video_id: string;
  title: string;
  channel: string;
  duration_seconds: number;
  chunk_count: number;
  status: 'processing' | 'completed' | 'failed' | 'not_found';
}

export interface IngestStatusResponse {
  video_id: string;
  status: 'processing' | 'completed' | 'failed' | 'not_found';
  title: string;
  chunk_count: number;
}

export interface ChatRequest {
  video_id: string;
  message: string;
  session_id: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  suggested_questions: string[];
  query_type: 'search' | 'summarize' | 'clarify' | 'contextualize';
}

export type AppState = 'idle' | 'detecting' | 'indexing' | 'ready' | 'error';

/* ── Chrome message types ── */
export interface ChromeMessage {
  type: string;
  payload?: unknown;
}

export interface VideoDetectedMessage extends ChromeMessage {
  type: 'VIDEO_DETECTED';
  payload: {
    videoId: string;
    url: string;
  };
}

export interface SeekVideoMessage extends ChromeMessage {
  type: 'SEEK_VIDEO';
  payload: {
    seconds: number;
  };
}

export interface GetVideoIdMessage extends ChromeMessage {
  type: 'GET_VIDEO_ID';
}
