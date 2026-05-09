/**
 * useChat hook — manages chat state and API communication.
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import type { ChatMessage, Citation } from '../types';
import { sendMessage as apiSendMessage, sendMessageStream as apiSendMessageStream } from '../services/api';

const STREAMING_ENABLED =
  ((import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_CHAT_STREAMING ?? '')
    .toLowerCase() === 'true' ||
  ((import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_CHAT_STREAMING ?? '') === '1';
const STREAMING_OVERRIDE_STORAGE_KEY = 'chat_streaming_enabled_override';

function generateId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<void>;
  clearMessages: () => void;
  suggestedQuestions: string[];
  streamingEnabled: boolean;
  setStreamingEnabled: (enabled: boolean) => void;
}

export function useChat(videoId: string, sessionId: string): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([
    'Summarize this video',
    'What are the main points?',
    'List all topics covered',
  ]);
  const [streamingOverride, setStreamingOverride] = useState<boolean | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const streamingEnabled = useMemo(
    () => streamingOverride ?? STREAMING_ENABLED,
    [streamingOverride]
  );

  useEffect(() => {
    chrome.storage.local
      .get(STREAMING_OVERRIDE_STORAGE_KEY)
      .then((result) => {
        const saved = result[STREAMING_OVERRIDE_STORAGE_KEY];
        if (typeof saved === 'boolean') {
          setStreamingOverride(saved);
        }
      })
      .catch(() => {
        // Ignore storage access errors and keep env-based behavior.
      });
  }, []);

  const setStreamingEnabled = useCallback((enabled: boolean) => {
    setStreamingOverride(enabled);
    chrome.storage.local
      .set({ [STREAMING_OVERRIDE_STORAGE_KEY]: enabled })
      .catch(() => {
        // Ignore persistence errors; runtime toggle still works for this session.
      });
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || !videoId || isLoading) return;

      // Cancel any in-flight request
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      const userMessage: ChatMessage = {
        id: generateId(),
        role: 'user',
        content: content.trim(),
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setError(null);

      try {
        if (streamingEnabled) {
          const assistantMessageId = generateId();
          let streamedAnswer = '';
          let receivedChunks = false;
          let receivedAnyEvent = false;
          let finalResponse = null as null | {
            answer: string;
            citations: Citation[];
            suggested_questions: string[];
          };

          setMessages((prev) => [
            ...prev,
            {
              id: assistantMessageId,
              role: 'assistant',
              content: '',
              timestamp: Date.now(),
            },
          ]);

          const watchdog = window.setTimeout(() => {
            // If nothing arrives, abort the stream so we can fall back to non-streaming.
            if (!receivedAnyEvent) {
              abortRef.current?.abort();
            }
          }, 30000);

          const overallTimeout = window.setTimeout(() => {
            // Avoid hanging forever if the backend/LLM stalls.
            abortRef.current?.abort();
          }, 90_000);

          try {
            await apiSendMessageStream(
              videoId,
              content.trim(),
              sessionId,
              {
                onMeta: () => {
                  receivedAnyEvent = true;
                },
                onChunk: (chunkText) => {
                  receivedAnyEvent = true;
                  receivedChunks = true;
                  streamedAnswer += chunkText;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, content: streamedAnswer }
                        : msg
                    )
                  );
                },
                onFinal: (payload) => {
                  receivedAnyEvent = true;
                  finalResponse = payload;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            content: payload.answer || streamedAnswer,
                            citations: payload.citations,
                            suggested_questions: payload.suggested_questions,
                          }
                        : msg
                    )
                  );
                },
              },
              abortRef.current.signal
            );

            const suggestions = finalResponse?.suggested_questions ?? [];
            if (suggestions.length > 0) {
              setSuggestedQuestions(suggestions);
            }
            return;
          } catch (streamErr) {
            if (receivedChunks) {
              throw streamErr;
            }
            setMessages((prev) =>
              prev.filter((msg) => msg.id !== assistantMessageId)
            );
          } finally {
            window.clearTimeout(watchdog);
            window.clearTimeout(overallTimeout);
          }
        }

        const response = await apiSendMessage(videoId, content.trim(), sessionId);

        const botMessage: ChatMessage = {
          id: generateId(),
          role: 'assistant',
          content: response.answer,
          citations: response.citations,
          suggested_questions: response.suggested_questions,
          timestamp: Date.now(),
        };

        setMessages((prev) => [...prev, botMessage]);

        if (response.suggested_questions?.length > 0) {
          setSuggestedQuestions(response.suggested_questions);
        }
      } catch (err) {
        const errorMsg =
          err instanceof Error ? err.message : 'Failed to get response';
        setError(errorMsg);

        const errorMessage: ChatMessage = {
          id: generateId(),
          role: 'assistant',
          content: `⚠️ ${errorMsg}. Please make sure the backend server is running at http://localhost:8000`,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    },
    [videoId, sessionId, isLoading, streamingEnabled]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    setSuggestedQuestions([
      'Summarize this video',
      'What are the main points?',
      'List all topics covered',
    ]);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    suggestedQuestions,
    streamingEnabled,
    setStreamingEnabled,
  };
}
