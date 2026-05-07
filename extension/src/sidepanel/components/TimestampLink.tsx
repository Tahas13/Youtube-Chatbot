/**
 * TimestampLink — clickable timestamp that seeks the YouTube player.
 */

import { Clock } from 'lucide-react';

interface TimestampLinkProps {
  seconds: number;
  display: string;
  tooltip?: string;
}

export default function TimestampLink({ seconds, display, tooltip }: TimestampLinkProps) {
  const handleClick = () => {
    // Send seek message through the background worker to the content script
    try {
      chrome.runtime.sendMessage({
        type: 'SEEK_VIDEO',
        payload: { seconds },
      });
    } catch {
      // Fallback: try direct tab message
      console.warn('Failed to send seek message via runtime');
    }
  };

  return (
    <button
      onClick={handleClick}
      className="timestamp-link"
      title={tooltip || `Jump to ${display}`}
      aria-label={`Jump to ${display} in video`}
    >
      <Clock className="w-3 h-3" />
      <span>{display}</span>
    </button>
  );
}
