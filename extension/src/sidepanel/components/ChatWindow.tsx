/**
 * ChatWindow — scrollable message list with auto-scroll.
 */

import { useEffect, useRef, type ReactNode } from 'react';
import type { ChatMessage } from '../types';
import MessageBubble from './MessageBubble';
import LoadingIndicator from './LoadingIndicator';

interface ChatWindowProps {
  messages: ChatMessage[];
  isLoading: boolean;
  children?: ReactNode;
}

export default function ChatWindow({ messages, isLoading, children }: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (messages.length === 0) {
    return (
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {children}
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-4 space-y-4">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}

      {isLoading && (
        <div className="flex items-start gap-2 animate-fade-in">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-brand-500/20 to-accent-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
            <div className="w-3.5 h-3.5 rounded-full border-2 border-brand-400 border-t-transparent animate-spin" />
          </div>
          <div className="glass-card px-4 py-3">
            <LoadingIndicator size="sm" />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
