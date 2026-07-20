/* ==========================================================================
   AuraScan Dynamic Interactivity Script (Group O Recess Project)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    renderIcons();
    initUploadPanel();
    initChatbot();
});

function renderIcons() {
    window.lucide?.createIcons({ attrs: { 'stroke-width': 1.8 } });
}

/**
 * Displays a short, accessible in-app notification without using browser alerts.
 */
function showSnackbar(message, type = 'info') {
    const region = document.getElementById('snackbar-region');
    if (!region) return;

    const snackbar = document.createElement('div');
    snackbar.className = `snackbar snackbar-${type}`;
    snackbar.setAttribute('role', type === 'error' ? 'alert' : 'status');

    const icon = document.createElement('i');
    icon.setAttribute('data-lucide', type === 'error'
        ? 'circle-alert'
        : type === 'success'
            ? 'circle-check'
            : 'info');
    icon.setAttribute('aria-hidden', 'true');

    const text = document.createElement('span');
    text.textContent = message;

    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'snackbar-close';
    close.setAttribute('aria-label', 'Dismiss notification');
    close.innerHTML = '<i data-lucide="x" aria-hidden="true"></i>';

    const dismiss = () => {
        snackbar.classList.remove('is-visible');
        snackbar.addEventListener('transitionend', () => snackbar.remove(), { once: true });
    };

    close.addEventListener('click', dismiss);
    snackbar.append(icon, text, close);
    region.appendChild(snackbar);
    renderIcons();
    requestAnimationFrame(() => snackbar.classList.add('is-visible'));
    window.setTimeout(dismiss, 5000);
}

/**
 * Handles all drag-and-drop and manual file uploads on the landing page
 */
function initUploadPanel() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const progressContainer = document.getElementById('progress-container');
    const uploadPrompt = document.querySelector('.upload-prompt');
    const filenameDisplay = document.getElementById('filename-display');
    const percentDisplay = document.getElementById('percent-display');
    const progressFill = document.getElementById('progress-fill');
    const statusDisplay = document.getElementById('status-display');
    const uploadComposer = document.getElementById('upload-composer');
    const uploadNote = document.getElementById('upload-note');

    if (!dropzone) return; // Exit if not on landing upload page

    uploadComposer?.addEventListener('click', (e) => e.stopPropagation());

    // Trigger input click when browse is clicked
    browseBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag-over styling
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Handle manual select file
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    function handleFileUpload(file) {
        // Validation: Verify it is an image
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!allowedTypes.includes(file.type)) {
            showSnackbar('Unsupported file format. Please upload a PNG or JPG X-Ray image.', 'error');
            return;
        }

        // Show progress panel
        uploadPrompt.style.display = 'none';
        progressContainer.hidden = false;
        progressContainer.style.display = 'block';
        filenameDisplay.textContent = file.name;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('note', uploadNote?.value.trim() || '');

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/analyze', true);

        // Track upload progress
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                // Limit visual load percentage to 90% until backend processes predictions
                const visualPercent = Math.round(percent * 0.9);
                percentDisplay.textContent = `${visualPercent}%`;
                progressFill.style.width = `${visualPercent}%`;
            }
        });

        // Backend response
        xhr.onreadystatechange = () => {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    if (response.status === 'success') {
                        percentDisplay.textContent = '100%';
                        progressFill.style.width = '100%';
                        statusDisplay.innerHTML = '<i data-lucide="check" class="text-low"></i> Analysis complete. Rendering diagnostic dashboard...';
                        renderIcons();
                        
                        // Small delay for visual completion, then redirect to report
                        setTimeout(() => {
                            window.location.href = '/report';
                        }, 800);
                    } else {
                        resetUpload('Analysis failed: ' + response.message);
                    }
                } else {
                    resetUpload('Server error occurred during processing. Status code: ' + xhr.status);
                }
            }
        };

        statusDisplay.innerHTML = '<i data-lucide="loader-circle" class="icon-spin"></i> Uploading scan vectors...';
        renderIcons();
        xhr.send(formData);
    }

    function resetUpload(errorMessage) {
        showSnackbar(errorMessage, 'error');
        uploadPrompt.style.display = 'block';
        progressContainer.hidden = true;
        progressContainer.style.display = 'none';
        progressFill.style.width = '0%';
        percentDisplay.textContent = '0%';
        fileInput.value = '';
    }
}

/**
 * Handles Q&A chat interactions on the report page
 */
function initChatbot() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    if (!chatForm) return; // Exit if not on report screen

    // Always scroll to bottom of chat on load
    scrollChatToBottom();

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userQuery = chatInput.value.trim();
        if (!userQuery) return;
        // Render User Bubble immediately
        appendMessageBubble('user', userQuery);
        chatInput.value = '';
        
        // Render typing animation bubble
        const typingBubble = appendTypingIndicator();
        scrollChatToBottom();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userQuery })
            });

            // Remove typing bubble
            typingBubble.remove();

            if (response.ok) {
                const result = await response.json();
                if (result.status === 'success') {
                    appendMessageBubble('assistant', result.response.content);
                } else {
                    showSnackbar(result.message || 'The assistant could not answer that request.', 'error');
                }
            } else {
                showSnackbar('Connection lost with the application service.', 'error');
            }
        } catch (error) {
            typingBubble.remove();
            showSnackbar('The request timed out. Please try again.', 'error');
        }

        scrollChatToBottom();
    });

    function appendMessageBubble(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${role}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        // Format simple Markdown breaks to HTML breaks inside bubbles
        bubble.innerHTML = content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
            
        msgDiv.appendChild(bubble);
        chatMessages.appendChild(msgDiv);
    }

    function appendTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message assistant';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        const dots = document.createElement('div');
        dots.className = 'typing-dots';
        dots.innerHTML = '<span></span><span></span><span></span>';
        
        bubble.appendChild(dots);
        typingDiv.appendChild(bubble);
        chatMessages.appendChild(typingDiv);
        return typingDiv;
    }

    function scrollChatToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}
