/**
 * ChatInput — text input with send button.
 */

import { useState, useRef, useCallback, type KeyboardEvent } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export default function ChatInput({ onSend, isLoading, disabled }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (trimmed && !isLoading && !disabled) {
      onSend(trimmed);
      setValue('');

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [value, onSend, isLoading, disabled]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  };

  return (
    <div className="p-3 border-t border-white/5">
      <div className="glass-input flex items-end gap-2 p-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            handleInput();
          }}
          onKeyDown={handleKeyDown}
          placeholder={
            disabled
              ? 'Waiting for video...'
              : isLoading
              ? 'Thinking...'
              : 'Ask about this video...'
          }
          disabled={disabled || isLoading}
          rows={1}
          className="flex-1 bg-transparent text-sm text-surface-200 placeholder-surface-500 resize-none outline-none min-h-[36px] max-h-[120px] py-1.5 px-2 leading-snug"
          id="chat-input"
        />
        <button
          onClick={handleSend}
          disabled={!value.trim() || isLoading || disabled}
          className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-200 ${
            value.trim() && !isLoading && !disabled
              ? 'bg-gradient-to-r from-brand-500 to-brand-600 text-white hover:from-brand-600 hover:to-brand-700 shadow-lg shadow-brand-500/20'
              : 'bg-surface-700/50 text-surface-500 cursor-not-allowed'
          }`}
          aria-label="Send message"
          id="send-button"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
      <p className="text-[10px] text-surface-600 text-center mt-1.5">
        Shift + Enter for new line
      </p>
    </div>
  );
}
