/**
 * Content script — runs on YouTube pages.
 * Detects video changes and handles video seeking from the side panel.
 */

(() => {
  let currentVideoId: string | null = null;

  /**
   * Extract video ID from current URL.
   */
  function getVideoId(): string | null {
    const params = new URLSearchParams(window.location.search);
    return params.get('v');
  }

  /**
   * Notify background worker about the current video.
   */
  function notifyVideoDetected(videoId: string) {
    chrome.runtime.sendMessage({
      type: 'VIDEO_DETECTED',
      payload: {
        videoId,
        url: window.location.href,
      },
    }).catch(() => {
      // Extension context may be invalidated, ignore
    });
  }

  /**
   * Check for video changes (YouTube is a SPA).
   */
  function checkForVideoChange() {
    const videoId = getVideoId();
    if (videoId && videoId !== currentVideoId) {
      currentVideoId = videoId;
      notifyVideoDetected(videoId);
    }
  }

  /**
   * Seek the YouTube video player to a specific time.
   */
  function seekVideo(seconds: number) {
    const video = document.querySelector('video') as HTMLVideoElement | null;
    if (video) {
      video.currentTime = seconds;
      video.play().catch(() => {
        // Autoplay might be blocked, that's ok
      });

      // Scroll video into view
      video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  // ── Listen for messages from background/sidepanel ──
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === 'SEEK_VIDEO') {
      const seconds = message.payload?.seconds;
      if (typeof seconds === 'number') {
        seekVideo(seconds);
        sendResponse({ status: 'ok' });
      }
      return true;
    }

    if (message.type === 'GET_VIDEO_ID') {
      sendResponse({ videoId: getVideoId() });
      return true;
    }
  });

  // ── Initial detection ──
  checkForVideoChange();

  // ── Watch for SPA navigation ──
  // YouTube uses History API for navigation, so we observe URL changes
  let lastUrl = window.location.href;

  const observer = new MutationObserver(() => {
    if (window.location.href !== lastUrl) {
      lastUrl = window.location.href;
      checkForVideoChange();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Also listen for popstate
  window.addEventListener('popstate', checkForVideoChange);

  // Periodic check as a fallback
  setInterval(checkForVideoChange, 2000);
})();
