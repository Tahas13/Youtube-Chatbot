/**
 * useVideoInfo hook — detects current YouTube video and manages indexing.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { VideoInfo, AppState } from '../types';
import { ingestVideo, checkVideoStatus } from '../services/api';

interface UseVideoInfoReturn {
  videoInfo: VideoInfo | null;
  appState: AppState;
  errorMessage: string | null;
  sessionId: string;
  reindex: () => Promise<void>;
}

function generateSessionId(videoId: string): string {
  return `session_${videoId}_${Date.now()}`;
}

export function useVideoInfo(): UseVideoInfoReturn {
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [appState, setAppState] = useState<AppState>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>('');
  const currentVideoIdRef = useRef<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /**
   * Index a video via the backend.
   */
  const indexVideo = useCallback(async (videoId: string) => {
    setAppState('indexing');
    setErrorMessage(null);

    try {
      const result = await ingestVideo(
        `https://www.youtube.com/watch?v=${videoId}`
      );

      if (!result.chunk_count || result.chunk_count <= 0) {
        throw new Error(
          'Indexing produced 0 transcript chunks. Try re-indexing, or the video may not have captions available.'
        );
      }

      setVideoInfo({
        videoId: result.video_id,
        title: result.title,
        channel: result.channel,
        thumbnailUrl: `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`,
        duration: result.duration_seconds,
      });

      setSessionId(generateSessionId(videoId));
      setAppState('ready');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to index video';
      setErrorMessage(msg);
      setAppState('error');
    }
  }, []);

  /**
   * Detect video from the active tab.
   */
  const detectVideo = useCallback(async () => {
    try {
      // Ask background for current video
      const response = await chrome.runtime.sendMessage({
        type: 'GET_CURRENT_VIDEO',
      });

      const videoId = response?.videoId;

      if (videoId && videoId !== currentVideoIdRef.current) {
        currentVideoIdRef.current = videoId;
        setAppState('detecting');

        // Check if already indexed
        try {
          const status = await checkVideoStatus(videoId);
          if (status.status === 'completed' && status.chunk_count > 0) {
            setVideoInfo({
              videoId,
              title: status.title || 'YouTube Video',
              channel: '',
              thumbnailUrl: `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`,
              duration: 0,
            });
            setSessionId(generateSessionId(videoId));
            setAppState('ready');
            return;
          }
        } catch {
          // Backend might not be running, try to index anyway
        }

        // Not indexed yet, start indexing
        await indexVideo(videoId);
      } else if (!videoId) {
        currentVideoIdRef.current = null;
        setVideoInfo(null);
        setAppState('idle');
      }
    } catch {
      // Extension API might fail if context is invalidated
    }
  }, [indexVideo]);

  /**
   * Force re-index the current video.
   */
  const reindex = useCallback(async () => {
    if (currentVideoIdRef.current) {
      await indexVideo(currentVideoIdRef.current);
    }
  }, [indexVideo]);

  // ── Poll for video changes ──
  useEffect(() => {
    // Initial detection
    detectVideo();

    // Poll every 2 seconds for video changes
    pollingRef.current = setInterval(detectVideo, 2000);

    // Listen for storage changes (from background worker)
    const handleStorageChange = () => {
      detectVideo();
    };
    chrome.storage.session?.onChanged.addListener(handleStorageChange);

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      chrome.storage.session?.onChanged.removeListener(handleStorageChange);
    };
  }, [detectVideo]);

  return {
    videoInfo,
    appState,
    errorMessage,
    sessionId,
    reindex,
  };
}
