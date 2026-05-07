import { describe, expect, it, vi } from 'vitest';
import { processSseBuffer, processSseEvent } from './api';
import type { ChatStreamHandlers } from './api';

function createHandlers() {
  const handlers: ChatStreamHandlers = {
    onMeta: vi.fn(),
    onChunk: vi.fn(),
    onFinal: vi.fn(),
  };
  return handlers;
}

describe('SSE parser helpers', () => {
  it('handles partial chunk boundaries across frames', () => {
    const handlers = createHandlers();
    let buffer = '';

    buffer = processSseBuffer(
      buffer,
      'event: chunk\ndata: {"text":"Hel',
      handlers
    );
    expect((handlers.onChunk as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);

    buffer = processSseBuffer(buffer, 'lo"}\n\n', handlers);
    expect((handlers.onChunk as ReturnType<typeof vi.fn>).mock.calls).toEqual([['Hello']]);
    expect(buffer).toBe('');
  });

  it('handles multiple events in one frame', () => {
    const handlers = createHandlers();
    const frame =
      'event: chunk\ndata: {"text":"A"}\n\n' +
      'event: chunk\ndata: {"text":"B"}\n\n' +
      'event: meta\ndata: {"session_id":"s1"}\n\n';

    const buffer = processSseBuffer('', frame, handlers);

    expect((handlers.onChunk as ReturnType<typeof vi.fn>).mock.calls).toEqual([['A'], ['B']]);
    expect((handlers.onMeta as ReturnType<typeof vi.fn>).mock.calls).toEqual([[{ session_id: 's1' }]]);
    expect(buffer).toBe('');
  });

  it('ignores unknown events', () => {
    const handlers = createHandlers();

    expect(() =>
      processSseEvent('event: ping\ndata: {"ts":123}\n\n', handlers)
    ).not.toThrow();

    expect((handlers.onMeta as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
    expect((handlers.onChunk as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
    expect((handlers.onFinal as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
  });

  it('ignores malformed JSON payloads', () => {
    const handlers = createHandlers();

    expect(() =>
      processSseEvent('event: chunk\ndata: {"text":"bad"\n\n', handlers)
    ).not.toThrow();

    expect((handlers.onChunk as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
  });
});
