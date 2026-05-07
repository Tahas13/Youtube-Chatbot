/**
 * MessageBubble — renders a single chat message (user or bot).
 * Bot messages are rendered as markdown with clickable timestamps.
 */

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '../types';
import TimestampLink from './TimestampLink';
import { User, Sparkles } from 'lucide-react';

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex items-start gap-2 ${
        isUser ? 'flex-row-reverse animate-slide-in-right' : 'animate-slide-in-left'
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 ${
          isUser
            ? 'bg-gradient-to-br from-brand-500 to-brand-700'
            : 'bg-gradient-to-br from-brand-500/20 to-accent-500/20'
        }`}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5 text-white" />
        ) : (
          <Sparkles className="w-3.5 h-3.5 text-brand-400" />
        )}
      </div>

      {/* Message Content */}
      <div
        className={`max-w-[85%] px-4 py-3 text-sm leading-relaxed ${
          isUser ? 'message-user' : 'message-bot'
        }`}
      >
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="prose-sm">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Replace timestamp patterns with clickable links
                p: ({ children }) => (
                  <p className="mb-2 last:mb-0">
                    {processTimestamps(children)}
                  </p>
                ),
                li: ({ children }) => (
                  <li className="mb-1">
                    {processTimestamps(children)}
                  </li>
                ),
                a: ({ href, children }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-brand-400 hover:text-brand-300 underline"
                  >
                    {children}
                  </a>
                ),
                table: ({ children }) => (
                  <div className="overflow-x-auto mb-2">
                    <table className="w-full text-left border-collapse border border-white/10 text-xs">
                      {children}
                    </table>
                  </div>
                ),
                th: ({ children }) => (
                  <th className="border border-white/10 px-2 py-1 bg-white/5 font-semibold">
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td className="border border-white/10 px-2 py-1">
                    {processTimestamps(children)}
                  </td>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Citation chips (shown below bot messages) */}
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-white/5">
            {message.citations.slice(0, 5).map((citation, idx) => (
              <TimestampLink
                key={idx}
                seconds={citation.timestamp_seconds}
                display={citation.timestamp_display}
                tooltip={citation.text}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Process text children to find and replace timestamp patterns
 * like [05:23] or [1:02:30] with clickable TimestampLink components.
 */
function processTimestamps(children: React.ReactNode): React.ReactNode {
  if (!children) return children;

  const processNode = (node: React.ReactNode): React.ReactNode => {
    if (typeof node === 'string') {
      // Match [HH:MM:SS] or [MM:SS] patterns
      const regex = /\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g;
      const parts: React.ReactNode[] = [];
      let lastIndex = 0;
      let match;

      while ((match = regex.exec(node)) !== null) {
        // Add text before the match
        if (match.index > lastIndex) {
          parts.push(node.slice(lastIndex, match.index));
        }

        // Convert timestamp to seconds
        const display = match[1];
        const timeParts = display.split(':').map(Number);
        let seconds = 0;
        if (timeParts.length === 3) {
          seconds = timeParts[0] * 3600 + timeParts[1] * 60 + timeParts[2];
        } else if (timeParts.length === 2) {
          seconds = timeParts[0] * 60 + timeParts[1];
        }

        parts.push(
          <TimestampLink
            key={`ts-${match.index}`}
            seconds={seconds}
            display={display}
          />
        );

        lastIndex = match.index + match[0].length;
      }

      // Add remaining text
      if (lastIndex < node.length) {
        parts.push(node.slice(lastIndex));
      }

      return parts.length > 0 ? <>{parts}</> : node;
    }

    if (Array.isArray(node)) {
      return node.map((child, i) => (
        <span key={i}>{processNode(child)}</span>
      ));
    }

    return node;
  };

  if (Array.isArray(children)) {
    return children.map((child, i) => (
      <span key={i}>{processNode(child)}</span>
    ));
  }

  return processNode(children);
}
