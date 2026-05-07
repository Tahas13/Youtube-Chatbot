/**
 * Chrome Extension Background Service Worker.
 * Manages side panel lifecycle and tab-level video detection.
 */

// Open side panel when extension icon is clicked
chrome.action.onClicked.addListener(async (tab) => {
  if (tab.id) {
    try {
      await chrome.sidePanel.open({ tabId: tab.id });
    } catch (err) {
      console.error('Failed to open side panel:', err);
    }
  }
});

// Set side panel behavior: open on action click
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(console.error);

// Listen for tab updates to detect YouTube navigation
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.url && tab.url) {
    const isYouTube = tab.url.includes('youtube.com/watch');

    // Enable/disable side panel based on whether we're on a YouTube video page
    chrome.sidePanel.setOptions({
      tabId,
      path: 'sidepanel.html',
      enabled: isYouTube,
    }).catch(console.error);

    // If we're on a YouTube video page, extract video ID and notify
    if (isYouTube) {
      const videoId = extractVideoId(tab.url);
      if (videoId) {
        // Store the current video ID
        chrome.storage.session.set({
          [`tab_${tabId}_videoId`]: videoId,
          [`tab_${tabId}_url`]: tab.url,
        }).catch(console.error);
      }
    }
  }
});

// Listen for messages from content script or side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'VIDEO_DETECTED') {
    const videoId = message.payload?.videoId;
    const tabId = sender.tab?.id;

    if (videoId && tabId) {
      chrome.storage.session.set({
        [`tab_${tabId}_videoId`]: videoId,
        [`tab_${tabId}_url`]: message.payload?.url || '',
      }).catch(console.error);
    }

    sendResponse({ status: 'ok' });
    return true;
  }

  if (message.type === 'GET_CURRENT_VIDEO') {
    // Get the active tab's video ID
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const tab = tabs[0];
      if (tab?.id) {
        const data = await chrome.storage.session.get(`tab_${tab.id}_videoId`);
        const videoId = data[`tab_${tab.id}_videoId`] || null;
        sendResponse({ videoId, url: tab.url });
      } else {
        sendResponse({ videoId: null, url: null });
      }
    });
    return true; // async response
  }

  if (message.type === 'SEEK_VIDEO') {
    // Forward seek command to the content script on the active tab
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      if (tab?.id) {
        chrome.tabs.sendMessage(tab.id, {
          type: 'SEEK_VIDEO',
          payload: message.payload,
        }).catch(console.error);
      }
    });
    sendResponse({ status: 'ok' });
    return true;
  }
});

// Clean up when tabs are closed
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove([
    `tab_${tabId}_videoId`,
    `tab_${tabId}_url`,
  ]).catch(console.error);
});

/**
 * Extract video ID from a YouTube URL.
 */
function extractVideoId(url: string): string | null {
  try {
    const urlObj = new URL(url);
    return urlObj.searchParams.get('v');
  } catch {
    return null;
  }
}
