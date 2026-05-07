/**
 * VideoInfo — header showing current video metadata.
 */

import type { VideoInfo as VideoInfoType } from '../types';
import { RefreshCw, Trash2, Youtube } from 'lucide-react';

interface VideoInfoProps {
  videoInfo: VideoInfoType;
  onReindex: () => void;
  onClear: () => void;
  streamingEnabled: boolean;
  onStreamingToggle: (enabled: boolean) => void;
}

export default function VideoInfo({
  videoInfo,
  onReindex,
  onClear,
  streamingEnabled,
  onStreamingToggle,
}: VideoInfoProps) {
  return (
    <div className="glass-surface border-b border-white/5 p-3">
      <div className="flex items-start gap-3">
        {/* Thumbnail */}
        <div className="relative w-16 h-10 rounded-lg overflow-hidden flex-shrink-0 bg-surface-800">
          <img
            src={videoInfo.thumbnailUrl}
            alt={videoInfo.title}
            className="w-full h-full object-cover"
            loading="lazy"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent" />
          <div className="absolute bottom-0.5 right-0.5">
            <Youtube className="w-3 h-3 text-red-400" />
          </div>
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <h2
            className="text-xs font-semibold text-surface-200 truncate leading-tight"
            title={videoInfo.title}
          >
            {videoInfo.title}
          </h2>
          {videoInfo.channel && (
            <p className="text-[10px] text-surface-500 truncate mt-0.5">
              {videoInfo.channel}
            </p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={onClear}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-surface-500 hover:text-surface-300 hover:bg-surface-700/50 transition-colors"
            title="Clear chat"
            aria-label="Clear chat history"
            id="clear-chat-button"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onReindex}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-surface-500 hover:text-surface-300 hover:bg-surface-700/50 transition-colors"
            title="Re-index video"
            aria-label="Re-index video"
            id="reindex-button"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      <div className="mt-2 flex items-center justify-between rounded-lg border border-white/5 bg-surface-900/30 px-2 py-1.5">
        <div>
          <p className="text-[10px] font-medium text-surface-300">Streaming mode</p>
          <p className="text-[9px] text-surface-500">Live chunks from backend SSE</p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={streamingEnabled}
          onClick={() => onStreamingToggle(!streamingEnabled)}
          className={`relative h-5 w-9 rounded-full transition-colors ${
            streamingEnabled ? 'bg-brand-500' : 'bg-surface-700'
          }`}
          title={`Streaming mode ${streamingEnabled ? 'on' : 'off'}`}
        >
          <span
            className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
              streamingEnabled ? 'translate-x-4' : 'translate-x-0.5'
            }`}
          />
        </button>
      </div>
    </div>
  );
}
