/**
 * Main App component for the YouTube AI Chatbot side panel.
 */

import { useVideoInfo } from './hooks/useVideoInfo';
import { useChat } from './hooks/useChat';
import VideoInfo from './components/VideoInfo';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import SuggestedQuestions from './components/SuggestedQuestions';
import LoadingIndicator from './components/LoadingIndicator';
import { MessageSquare, Sparkles, AlertCircle } from 'lucide-react';

export default function App() {
  const { videoInfo, appState, errorMessage, sessionId, reindex } = useVideoInfo();
  const {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    suggestedQuestions,
    streamingEnabled,
    setStreamingEnabled,
  } = useChat(videoInfo?.videoId || '', sessionId);

  // ── Idle State: No video detected ──
  if (appState === 'idle') {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-brand-500/20 to-accent-500/20 flex items-center justify-center mb-6 animate-pulse-soft">
          <MessageSquare className="w-8 h-8 text-brand-400" />
        </div>
        <h1 className="text-xl font-semibold gradient-text mb-3">
          YouTube AI Chatbot
        </h1>
        <p className="text-surface-400 text-sm leading-relaxed max-w-[260px]">
          Navigate to a YouTube video to start chatting. I can summarize, search for moments, and answer your questions.
        </p>
        <div className="mt-8 flex items-center gap-2 text-surface-500 text-xs">
          <div className="w-2 h-2 rounded-full bg-surface-600 animate-pulse-soft" />
          Waiting for a YouTube video...
        </div>
      </div>
    );
  }

  // ── Detecting State ──
  if (appState === 'detecting') {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <LoadingIndicator size="lg" />
        <p className="text-surface-400 text-sm mt-4">Detecting video...</p>
      </div>
    );
  }

  // ── Indexing State ──
  if (appState === 'indexing') {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="relative mb-6">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-brand-500/20 to-accent-500/20 flex items-center justify-center">
            <Sparkles className="w-8 h-8 text-brand-400 animate-pulse-soft" />
          </div>
          <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-surface-900 flex items-center justify-center">
            <div className="w-4 h-4 rounded-full border-2 border-brand-400 border-t-transparent animate-spin" />
          </div>
        </div>
        <h2 className="text-lg font-semibold text-white mb-2">
          Analyzing Video
        </h2>
        <p className="text-surface-400 text-sm leading-relaxed max-w-[260px]">
          Fetching transcript, creating embeddings, and building the search index...
        </p>
        <div className="mt-6 w-48 h-1.5 bg-surface-800 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-brand-500 to-accent-500 rounded-full shimmer" style={{ width: '60%' }} />
        </div>
      </div>
    );
  }

  // ── Error State ──
  if (appState === 'error') {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 rounded-2xl bg-red-500/10 flex items-center justify-center mb-6">
          <AlertCircle className="w-8 h-8 text-red-400" />
        </div>
        <h2 className="text-lg font-semibold text-white mb-2">
          Something went wrong
        </h2>
        <p className="text-surface-400 text-sm leading-relaxed max-w-[280px] mb-6">
          {errorMessage || 'Failed to process this video. Please try again.'}
        </p>
        <button
          onClick={reindex}
          className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium rounded-xl transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  // ── Ready State: Chat Interface ──
  return (
    <div className="h-full flex flex-col bg-surface-950">
      {/* Header */}
      {videoInfo && (
        <VideoInfo
          videoInfo={videoInfo}
          onReindex={reindex}
          onClear={clearMessages}
          streamingEnabled={streamingEnabled}
          onStreamingToggle={setStreamingEnabled}
        />
      )}

      {/* Chat Area */}
      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        <ChatWindow messages={messages} isLoading={isLoading}>
          {/* Show suggestions when no messages */}
          {messages.length === 0 && (
            <div className="flex-1 flex flex-col items-center justify-center p-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-brand-500/15 to-accent-500/15 flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6 text-brand-400" />
              </div>
              <p className="text-surface-400 text-sm text-center mb-6 max-w-[240px]">
                Ask me anything about this video. I'll find the answer with timestamps!
              </p>
              <SuggestedQuestions
                questions={suggestedQuestions}
                onSelect={sendMessage}
              />
            </div>
          )}
        </ChatWindow>

        {/* Suggestions after messages */}
        {messages.length > 0 && !isLoading && (
          <div className="px-3 pb-1">
            <SuggestedQuestions
              questions={suggestedQuestions}
              onSelect={sendMessage}
              compact
            />
          </div>
        )}

        {/* Input */}
        <ChatInput
          onSend={sendMessage}
          isLoading={isLoading}
          disabled={appState !== 'ready'}
        />
      </div>
    </div>
  );
}
