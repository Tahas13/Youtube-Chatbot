# Extension Notes

## Chat Streaming Mode

- Default behavior comes from `VITE_CHAT_STREAMING` at build time:
  - `true` or `1` enables streaming by default.
  - Any other value keeps non-streaming mode by default.
- Users can override this at runtime from the sidepanel header using the `Streaming mode` toggle.
- The runtime preference is persisted in `chrome.storage.local` under `chat_streaming_enabled_override`, so it survives extension reloads.
- If no override is stored, the extension uses the existing env-flag behavior.
