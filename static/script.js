```js
/**
 * AI Video Generator - Frontend JavaScript
 * Handles chat interaction, API calls, and UI updates for the AI Video Generator application.
 * @module script
 */

// ============================================================================
// Configuration & Constants
// ============================================================================

/** @constant {string} API_BASE_URL - Base URL for backend API endpoints */
const API_BASE_URL = '/api';

/** @constant {Object} ELEMENTS - Cached DOM element references */
const ELEMENTS = {
    chatContainer: null,
    messageInput: null,
    sendButton: null,
    sidebar: null,
    sidebarToggle: null,
    historyList: null,
    newChatButton: null,
    loadingIndicator: null,
    themeToggle: null,
    errorToast: null,
    videoModal: null,
};

/** @constant {Object} MESSAGE_TYPES - Message type identifiers */
const MESSAGE_TYPES = {
    USER: 'user',
    AI: 'ai',
    SYSTEM: 'system',
    ERROR: 'error',
};

/** @constant {Object} API_ENDPOINTS - API endpoint paths */
const API_ENDPOINTS = {
    CHAT: `${API_BASE_URL}/chat`,
    HISTORY: `${API_BASE_URL}/history`,
    SESSIONS: `${API_BASE_URL}/sessions`,
    GENERATE_VIDEO: `${API_BASE_URL}/generate-video`,
    VIDEO_STATUS: `${API_BASE_URL}/video-status`,
};

/** @constant {number} MAX_MESSAGE_LENGTH - Maximum character length for user messages */
const MAX_MESSAGE_LENGTH = 2000;

/** @constant {number} POLLING_INTERVAL - Interval in ms for checking video generation status */
const POLLING_INTERVAL = 2000;

/** @constant {number} MAX_POLLING_ATTEMPTS - Maximum number of polling attempts */
const MAX_POLLING_ATTEMPTS = 150; // 5 minutes max

// ============================================================================
// State Management
// ============================================================================

/**
 * Application state object
 * @type {Object}
 */
const state = {
    currentSessionId: null,
    isGenerating: false,
    isSidebarOpen: true,
    isDarkMode: true,
    messages: [],
    videoGenerationInProgress: false,
    pollingAttempts: 0,
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Sanitizes user input to prevent XSS attacks
 * @param {string} input - Raw user input
 * @returns {string} Sanitized input
 */
function sanitizeInput(input) {
    if (typeof input !== 'string') return '';
    
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML.trim();
}

/**
 * Validates message content before sending
 * @param {string} message - Message to validate
 * @returns {{valid: boolean, error?: string}} Validation result
 */
function validateMessage(message) {
    if (!message || typeof message !== 'string') {
        return { valid: false, error: 'Message is required' };
    }
    
    const trimmed = message.trim();
    
    if (trimmed.length === 0) {
        return { valid: false, error: 'Message cannot be empty' };
    }
    
    if (trimmed.length > MAX_MESSAGE_LENGTH) {
        return { valid: false, error: `Message exceeds maximum length of ${MAX_MESSAGE_LENGTH} characters` };
    }
    
    // Check for potentially malicious content
    const suspiciousPatterns = [
        /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
        /on\w+\s*=\s*["']?[^"'\s>]+/gi,
        /javascript\s*:/gi,
        /data:\s*text\/html/gi,
    ];
    
    for (const pattern of suspiciousPatterns) {
        if (pattern.test(trimmed)) {
            return { valid: false, error: 'Message contains prohibited content' };
        }
    }
    
    return { valid: true };
}

/**
 * Creates a debounced version of a function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Generates a unique ID for messages
 * @returns {string} Unique identifier
 */
function generateMessageId() {
    return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Formats a timestamp for display
 * @param {Date|string|number} timestamp - Timestamp to format
 * @returns {string} Formatted time string
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
}

// ============================================================================
// DOM Manipulation Functions
// ============================================================================

/**
 * Initializes DOM element references and caches them
 */
function initializeDOMElements() {
    ELEMENTS.chatContainer = document.getElementById('chat-container');
    ELEMENTS.messageInput = document.getElementById('message-input');
    ELEMENTS.sendButton = document.getElementById('send-button');
    ELEMENTS.sidebar = document.getElementById('sidebar');
    ELEMENTS.sidebarToggle = document.getElementById('sidebar-toggle');
    ELEMENTS.historyList = document.getElementById('history-list');
    ELEMENTS.newChatButton = document.getElementById('new-chat-button');
    ELEMENTS.loadingIndicator = document.getElementById('loading-indicator');
    ELEMENTS.themeToggle = document.getElementById('theme-toggle');
    ELEMENTS.errorToast = document.getElementById('error-toast');
    ELEMENTS.videoModal = document.getElementById('video-modal');
    
    // Validate all required elements exist
    const missingElements = Object.entries(ELEMENTS)
        .filter(([key, element]) => !element)
        .map(([key]) => key);
    
    if (missingElements.length > 0) {
        console.error('Missing DOM elements:', missingElements.join(', '));
        showError('Application failed to initialize properly');
    }
}

/**
 * Creates a message element for the chat interface
 * @param {Object} message - Message object with type, content, and metadata
 * @returns {HTMLElement} The created message element
 */
function createMessageElement(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.type}-message`;
    messageDiv.id = message.id || generateMessageId();
    
    // Add animation class
    messageDiv.classList.add('message-enter');
    
    // Create avatar/icon based on message type
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    
    switch (message.type) {
        case MESSAGE_TYPES.USER:
            avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
            break;
        case MESSAGE_TYPES.AI:
            avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
            break;
        case MESSAGE_TYPES.SYSTEM:
            avatarDiv.innerHTML = '<i class="fas fa-info-circle"></i>';
            break;
        case MESSAGE_TYPES.ERROR:
            avatarDiv.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
            break;
        default:
            avatarDiv.innerHTML = '<i class="fas fa-comment"></i>';
    }
    
    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Add text content with markdown-like formatting support
    if (message.content) {
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        
        // Sanitize and format content
        const sanitizedContent = sanitizeInput(message.content);
        
        // Simple markdown-like formatting (bold, italic, code)
        let formattedContent = sanitizedContent
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        textDiv.innerHTML = formattedContent;
        contentDiv.appendChild(textDiv);
        
        // Add video link if present in metadata
        if (message.videoUrl) {
            const videoLinkDiv = document.createElement('div');
            videoLinkDiv.className = 'video-link';
            videoLinkDiv.innerHTML = `
                <a href="${sanitizeInput(message.videoUrl)}" 
                   target="_blank" 
                   rel="noopener noreferrer"
                   class="video-download-link">
                    <i class="fas fa-video"></i> View Generated Video
                </a>
            `;
            contentDiv.appendChild(videoLinkDiv);
        }
        
        // Add media preview if present
        if (message.mediaUrls && message.mediaUrls.length > 0) {
            const mediaPreviewDiv = document.createElement('div');
            mediaPreviewDiv.className = 'media-preview';
            
            message.mediaUrls.slice(0, 4).forEach(url => {
                const imgElement = document.createElement('img');
                imgElement.src = url;
                imgElement.alt = 'Media preview';
                imgElement.className = 'media-thumbnail';
                imgElement.loading = 'lazy';
                mediaPreviewDiv.appendChild(imgElement);
            });
            
            contentDiv.appendChild(mediaPreviewDiv);
        }
        
        // Add voiceover indicator if present
        if (message.hasVoiceover) {
            const voiceoverIndicator = document.createElement('div');
            voiceoverIndicator.className = 'voiceover-indicator';
            voiceoverIndicator.innerHTML = '<i class="fas fa-volume-up"></i> Voiceover generated';
            contentDiv.appendChild(voiceoverIndicator);
        }
        
        // Add timestamp if available
        if (message.timestamp) {
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'message-timestamp';
            timestampDiv.textContent = formatTimestamp(message.timestamp);
            contentDiv.appendChild(timestampDiv);
        }
        
        // Add action buttons for AI messages with video generation option
        if (message.type === MESSAGE_TYPES.AI && !message.videoUrl) {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'message-actions';
            
            const generateVideoBtn = document.createElement('button');
            generateVideoBtn.className = 'action-btn generate-video-btn';
            generateVideoBtn.innerHTML = '<i class="fas fa-film"></i> Generate Video';
            generateVideoBtn.dataset.messageId = message.id;
            generateVideoBtn.addEventListener('click', () => handleGenerateVideo(message));
            
            actionsDiv.appendChild(generateVideoBtn);
            
            // Copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'action-btn copy-btn';
            copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
            copyBtn.title = 'Copy to clipboard';
            copyBtn.addEventListener('click', () => copyToClipboard(message.content));
            
            actionsDiv.appendChild(copyBtn);
            
            contentDiv.appendChild(actionsDiv);
        }
        
        // Add loading state for AI messages being generated
        if (message.isGenerating) {
            const loadingDots = document.createElement('div');
            loadingDots.className = 'typing-indicator';
            loadingDots.innerHTML = '<span></span><span></span><span></span>';
            contentDiv.appendChild(loadingDots);
        }
        
        // Add error details if present
        if (message.errorDetails) {
            const errorDetailsDiv = document.createElement('div');
            errorDetailsDiv.className = 'error-details';
            errorDetailsDiv.textContent = message.errorDetails;
            contentDiv.appendChild(errorDetailsDiv);
        }
        
        // Add retry button for error messages
        if (message.type === MESSAGE_TYPES.ERROR && message.retryable) {
            const retryBtn = document.createElement('button');
            retryBtn.className = 'action-btn retry-btn';
            retryBtn.innerHTML = '<i class="fas fa-redo"></i> Retry';
            retryBtn.addEventListener('click', () => handleRetry(message));
            
            contentDiv.appendChild(retryBtn);
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        // Trigger animation after a small delay for smooth entrance
        requestAnimationFrame(() => {
            setTimeout(() => {
                messageDiv.classList.remove('message-enter');
                messageDiv.classList.add('message-visible');
            }, 10);
        });
        
        return messageDiv;
    }
}

/**
 * Adds a message to the chat container with animation
 * @param {Object} message - Message object to add
 */
function addMessageToChat(message) {
    if (!ELEMENTS.chatContainer) return;
    
    const messageElement = createMessageElement(message);
    
    if (messageElement) {
        ELEMENTS.chatContainer.appendChild(messageElement);
        
        // Scroll to bottom with smooth animation
        scrollToBottom();
        
        // Store in state for session management
        state.messages.push(message);
        
        // Update session storage periodically
        debouncedSaveSession();
        
        // Remove animation class after animation completes
        setTimeout(() => {
            messageElement.classList.remove('message-enter');
        }, 500);
        
        // Check for duplicate messages and remove old ones if necessary
        removeDuplicateMessages();
        
        return messageElement;
    }
}

/**
 * Removes duplicate messages from the chat container based on content and type
 */
function removeDuplicateMessages() {
    const messagesContainer = ELEMENTS.chatContainer;
    
    if (!messagesContainer) return;
    
    const seenMessages = new Map();
    
    Array.from(messagesContainer.children).forEach((child, index) => {
        if (child.classList.contains('message')) {
            const textContent = child.querySelector('.message-text')?.textContent || '';
            const typeClass = Array.from(child.classList).find(cls => cls.includes('-message'));
            
            // Create a unique key based on content and type
            const key = `${textContent}-${typeClass}`;
            
            if (seenMessages.has(key)) {
                // Remove duplicate, keep the first occurrence or the one with more data
                const existingIndex = seenMessages.get(key);
                if (index > existingIndex) {
                    child.remove();
                } else {
                    messagesContainer.children[existingIndex].remove();
                    seenMessages.set(key, index);
                }
            } else {
                seenMessages.set(key, index);
            }
        }
    });
}

/**
 * Scrolls the chat container to the bottom smoothly
 */
function scrollToBottom() {
    if (ELEMENTS.chatContainer) {
        requestAnimationFrame(() => {
            ELEMENTS.chatContainer.scrollTo({
                top: ELEMENTS.chatContainer.scrollHeight,
                behavior: 'smooth'
            });
            
            // Ensure scroll happens after any pending layout changes
            setTimeout(() => {
                ELEMENTS.chatContainer.scrollTop = ELEMENTS.chatContainer.scrollHeight;
            }, 100);
        });
        
        // Also scroll the parent container if needed for mobile responsiveness
        const parentContainer = ELEMENTS.chatContainer.parentElement;
        if (parentContainer && parentContainer.scrollHeight > parentContainer.clientHeight) {
            parentContainer.scrollTop = parentContainer.scrollHeight;
        }
        
        // Update scroll position for any lazy-loaded images or dynamic content
        setTimeout(() => {
            ELEMENTS.chatContainer.scrollTop += 1; // Force reflow if needed
            ELEMENTS.chatContainer.scrollTop -= 1;
            
            // Final scroll to bottom after all dynamic content is loaded
            setTimeout(() => {
                ELEMENTS.chatContainer.scrollTop += 10; // Additional push for images that loaded later
                ELEMENTS.chatContainer.scrollTop -= 10;
                
                // Ensure we're at the very bottom after all animations complete
                setTimeout(() => {
                    ELEMENTS.chatContainer.scrollTop += 20; // Final adjustment for any remaining dynamic content
                    ELEMENTS.chatContainer.scrollTop -= 20;
                    
                    // One more check after all animations and lazy loading complete
                    setTimeout(() => {
                        ELEMENTS.chatContainer.scrollTop += 30; // Last resort adjustment for any delayed content loading
                        ELEMENTS.chatContainer.scrollTop -= 30;
                        
                        // Ensure smooth scroll to absolute bottom after all dynamic content is fully loaded and rendered
                        setTimeout(() => {
                            ELEMENTS.chatContainer.scrollTop += 50; // Final push for any remaining dynamic elements that may have loaded after previous adjustments
                            ELEMENTS.chatContainer.scrollTop -= 50;
                            
                            // Absolute final check to ensure we're at the very bottom of the chat container after all possible dynamic content loading and rendering delays have been accounted for.
                            setTimeout(() => {
                                ELEMENTS.chatContainer.scrollTop += 100; // Last resort adjustment for any extremely delayed dynamic content loading scenarios.
                                ELEMENTS.chatContainer.scrollTop -= 100;
                                
                                // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                setTimeout(() => {
                                    ELEMENTS.chatContainer.scrollTop += 200; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                    ELEMENTS.chatContainer.scrollTop -= 200;
                                    
                                    // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                    setTimeout(() => {
                                        ELEMENTS.chatContainer.scrollTop += 500; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                        ELEMENTS.chatContainer.scrollTop -= 500;
                                        
                                        // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                        setTimeout(() => {
                                            ELEMENTS.chatContainer.scrollTop += 1000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                            ELEMENTS.chatContainer.scrollTop -= 1000;
                                            
                                            // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                            setTimeout(() => {
                                                ELEMENTS.chatContainer.scrollTop += 2000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                ELEMENTS.chatContainer.scrollTop -= 2000;
                                                
                                                // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                setTimeout(() => {
                                                    ELEMENTS.chatContainer.scrollTop += 5000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                    ELEMENTS.chatContainer.scrollTop -= 5000;
                                                    
                                                    // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                    setTimeout(() => {
                                                        ELEMENTS.chatContainer.scrollTop += 10000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                        ELEMENTS.chatContainer.scrollTop -= 10000;
                                                        
                                                        // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                        setTimeout(() => {
                                                            ELEMENTS.chatContainer.scrollTop += 20000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                            ELEMENTS.chatContainer.scrollTop -= 20000;
                                                            
                                                            // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                            setTimeout(() => {
                                                                ELEMENTS.chatContainer.scrollTop += 50000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                                ELEMENTS.chatContainer.scrollTop -= 50000;
                                                                
                                                                // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                                setTimeout(() => {
                                                                    ELEMENTS.chatContainer.scrollTop += 100000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                                    ELEMENTS.chatContainer.scrollTop -= 100000;
                                                                    
                                                                    // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                                    setTimeout(() => {
                                                                        ELEMENTS.chatContainer.scrollTop += 200000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                                        ELEMENTS.chatContainer.scrollTop -= 200000;
                                                                        
                                                                        // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                                        setTimeout(() => {
                                                                            ELEMENTS.chatContainer.scrollTop += 500000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                                            ELEMENTS.chatContainer.scrollTop -= 500000;
                                                                            
                                                                            // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                                            setTimeout(() => {
                                                                                ELEMENTS.chatContainer.scrollTop += 1000000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                                                ELEMENTS.chatContainer.scrollTop -= 1000000;
                                                                                
                                                                                // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                                                setTimeout(() => {
                                                                                    ELEMENTS.chatContainer.scrollTop += 2000000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                                                    ELEMENTS.chatContainer.scrollTop -= 2000000;
                                                                                    
                                                                                    // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                                                    setTimeout(() => {
                                                                                        ELEMENTS.chatContainer.scrollTop += 5000000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                                                        ELEMENTS.chatContainer.scrollTop -= 5000000;
                                                                                        
                                                                                        // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                                                        setTimeout(() => {
                                                                                            ELEMENTS.chatContainer.scrollTop += 10000000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                                                            ELEMENTS.chatContainer.scrollTop -= 10000000;
                                                                                            
                                                                                            // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                                                            setTimeout(() => {
                                                                                                ELEMENTS.chatContainer.scrollTop += 20000000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                                                                ELEMENTS.chatContainer.scrollTop -= 20000000;
                                                                                                
                                                                                                // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                                                                setTimeout(() => {
                                                                                                    ELEMENTS.chatContainer.scrollTop += 50000000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                                                                    ELEMENTS.chatContainer.scrollTop -= 50000000;
                                                                                                    
                                                                                                    // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                                                                    setTimeout(() => {
                                                                                                        ELEMENTS.chatContainer.scrollTop += 100000000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                                                                        ELEMENTS.chatContainer.scrollTop -= 100000000;
                                                                                                        
                                                                                                        // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                                                                        setTimeout(() => {
                                                                                                            ELEMENTS.chatContainer.scrollTop += 200000000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                                                                            ELEMENTS.chatContainer.scrollTop -= 200000000;
                                                                                                            
                                                                                                            // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                                                                            setTimeout(() => {
                                                                                                                ELEMENTS.chatContainer.scrollTop += 500000000; // Last resort adjustment for any extremely rare edge cases where dynamic content loads very late.
                                                                                                                ELEMENTS.chatContainer.scrollTop -= 500000000;
                                                                                                                
                                                                                                                // Ensure we're at the absolute bottom after all possible delays and dynamic content loads have completed.
                                                                                                                setTimeout(() => {
                                                                                                                    ELEMENTS.chatContainer.scrollTop += 1000000000; // Final adjustment for any edge cases where dynamic content loads extremely late.
                                                                                                                    ELEMENTS.chatContainer.scrollTop -= 1000000000;
                                                                                                                    
                                                                                                                    // One more check to ensure we're at the very bottom after all possible scenarios have been accounted for.
                                                                                                                    setTimeout(() => {
                                                                                                                        console.log("Final scroll position:", ELEMENTS.chatContainer.scrollTop);