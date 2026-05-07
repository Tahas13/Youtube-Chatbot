/**
 * LoadingIndicator — animated typing dots and cycling text.
 */

import { useState, useEffect } from 'react';

interface LoadingIndicatorProps {
  size?: 'sm' | 'md' | 'lg';
  text?: boolean;
}

const LOADING_MESSAGES = [
  "Analyzing transcript...",
  "Searching for evidence...",
  "Reading video context...",
  "Compressing chunks...",
  "Thinking..."
];

export default function LoadingIndicator({ size = 'md', text = true }: LoadingIndicatorProps) {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    if (!text) return;
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 2500); // Change text every 2.5s
    
    return () => clearInterval(interval);
  }, [text]);

  const dotSize = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-2.5 h-2.5',
  }[size];

  const gap = {
    sm: 'gap-1',
    md: 'gap-1.5',
    lg: 'gap-2',
  }[size];

  return (
    <div className="flex flex-col gap-2">
      <div className={`flex items-center ${gap}`} role="status" aria-label="Loading">
        <div className={`typing-dot ${dotSize}`} />
        <div className={`typing-dot ${dotSize}`} />
        <div className={`typing-dot ${dotSize}`} />
      </div>
      {text && (
        <div className="text-xs text-brand-300/80 animate-pulse transition-opacity duration-300">
          {LOADING_MESSAGES[messageIndex]}
        </div>
      )}
    </div>
  );
}
