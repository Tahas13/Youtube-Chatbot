/**
 * API service layer for communicating with the FastAPI backend.
 */

import type { IngestResponse, IngestStatusResponse, ChatResponse } from '../types';

const API_BASE =
  ((import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_API_BASE ?? '').trim() ||
  'http://localhost:8001';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    let detail = errorText;
    try {
      const errorJson = JSON.parse(errorText);
      detail = errorJson.detail || errorText;
    } catch {
      // keep raw text
    }
    throw new ApiError(response.status, detail);
  }

  return response.json();
}

/**
 * Ingest a YouTube video — fetches transcript, chunks, and indexes.
 */
export async function ingestVideo(videoUrl: string): Promise<IngestResponse> {
  return request<IngestResponse>('/api/ingest', {
    method: 'POST',
    body: JSON.stringify({ video_url: videoUrl }),
  });
}

/**
 * Check if a video has already been indexed.
 */
export async function checkVideoStatus(videoId: string): Promise<IngestStatusResponse> {
  return request<IngestStatusResponse>(`/api/ingest/status/${videoId}`);
}

/**
 * Send a chat message about a video.
 */
export async function sendMessage(
  videoId: string,
  message: string,
  sessionId: string
): Promise<ChatResponse> {
  return request<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      video_id: videoId,
      message,
      session_id: sessionId,
    }),
  });
}

export interface ChatStreamHandlers {
  onMeta?: (meta: { session_id?: string }) => void;
  onChunk?: (chunkText: string) => void;
  onFinal?: (response: ChatResponse) => void;
}

export function processSseEvent(rawEvent: string, handlers: ChatStreamHandlers): void {
  const lines = rawEvent
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  const eventLine = lines.find((line) => line.startsWith('event:'));
  const dataLine = lines.find((line) => line.startsWith('data:'));
  if (!eventLine || !dataLine) return;

  const event = eventLine.replace('event:', '').trim();
  const dataRaw = dataLine.replace('data:', '').trim();
  let payload: unknown;
  try {
    payload = JSON.parse(dataRaw);
  } catch {
    return;
  }

  if (event === 'meta' && typeof payload === 'object' && payload) {
    handlers.onMeta?.(payload as { session_id?: string });
    return;
  }

  if (event === 'chunk' && typeof payload === 'object' && payload) {
    const text = (payload as { text?: string }).text ?? '';
    handlers.onChunk?.(text);
    return;
  }

  if (event === 'final' && typeof payload === 'object' && payload) {
    handlers.onFinal?.(payload as ChatResponse);
    return;
  }

  if (event === 'error' && typeof payload === 'object' && payload) {
    const messageText = (payload as { message?: string }).message ?? 'Streaming failed';
    throw new Error(messageText);
  }
}

export function processSseBuffer(
  buffer: string,
  incomingChunk: string,
  handlers: ChatStreamHandlers
): string {
  const combined = `${buffer}${incomingChunk}`;
  const events = combined.split('\n\n');
  const remaining = events.pop() ?? '';
  for (const rawEvent of events) {
    processSseEvent(rawEvent, handlers);
  }
  return remaining;
}

/**
 * Stream a chat response over SSE from /api/chat/stream.
 */
export async function sendMessageStream(
  videoId: string,
  message: string,
  sessionId: string,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
  timeoutMs: number = 60000
): Promise<void> {
  let finalReceived = false;
  const wrappedHandlers: ChatStreamHandlers = {
    onMeta: (meta) => handlers.onMeta?.(meta),
    onChunk: (chunkText) => handlers.onChunk?.(chunkText),
    onFinal: (response) => {
      finalReceived = true;
      handlers.onFinal?.(response);
    },
  };

  const timeoutController = new AbortController();
  const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);

  const onExternalAbort = () => timeoutController.abort();
  if (signal) {
    if (signal.aborted) {
      timeoutController.abort();
    } else {
      signal.addEventListener('abort', onExternalAbort, { once: true });
    }
  }
  try {
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({
        video_id: videoId,
        message,
        session_id: sessionId,
      }),
      signal: timeoutController.signal,
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new ApiError(response.status, errorText || 'Streaming request failed');
    }

    if (!response.body) {
      throw new Error('Streaming response body is empty');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer = processSseBuffer(buffer, decoder.decode(value, { stream: true }), wrappedHandlers);
      if (finalReceived) {
        try {
          await reader.cancel();
        } catch {
          // ignore
        }
        break;
      }
    }

    if (buffer.trim()) {
      processSseEvent(buffer, wrappedHandlers);
    }
    clearTimeout(timeoutId);
    return;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (signal) {
      signal.removeEventListener('abort', onExternalAbort);
    }
    // If aborted due to timeout, request extractive fallback and return it
    if (err && (err.name === 'AbortError' || err.message === 'The user aborted a request.')) {
      try {
        const fallback = await requestExtractiveFallback(videoId, message, sessionId);
        handlers.onFinal?.(fallback);
        return;
      } catch (e) {
        throw new Error('Streaming aborted and extractive fallback failed');
      }
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
    if (signal) {
      signal.removeEventListener('abort', onExternalAbort);
    }
  }
}


/**
 * Helper: request extractive fallback from server when stream times out.
 */
export async function requestExtractiveFallback(videoId: string, message: string, sessionId: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat/extractive`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id: videoId, message, session_id: sessionId }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json();
}

/**
 * Clear a chat session's memory.
 */
export async function clearSession(sessionId: string): Promise<void> {
  await request(`/api/chat/session/${sessionId}`, {
    method: 'DELETE',
  });
}

/**
 * Check if the backend is healthy.
 */
export async function healthCheck(): Promise<boolean> {
  try {
    await request('/api/health');
    return true;
  } catch {
    return false;
  }
}
