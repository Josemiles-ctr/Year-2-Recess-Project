document.addEventListener('DOMContentLoaded', () => {
    renderIcons();
    initUploadPanel();
    initChatbot();
    renderMarkdownMessages();
});

function renderIcons() {
    window.lucide?.createIcons({ attrs: { 'stroke-width': 1.8 } });
}

function safeMarked(text) {
    if (typeof marked?.parse === 'function') {
        try { return marked.parse(text); } catch {}
    }
    return simpleMarkdown(text);
}

function simpleMarkdown(text) {
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    html = html.replace(/^-{3,}\s*$/gm, '<hr>');

    const lines = html.split('\n');
    const out = [];
    let inList = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const liMatch = line.match(/^(\s*)[*-] (.+)$/);
        if (liMatch) {
            if (!inList) { out.push('<ul>'); inList = true; }
            out.push('<li>' + liMatch[2] + '</li>');
        } else {
            if (inList) { out.push('</ul>'); inList = false; }
            out.push(line ? '<p>' + line + '</p>' : '<br>');
        }
    }
    if (inList) out.push('</ul>');
    return out.join('\n');
}

function renderMarkdownMessages() {
    document.querySelectorAll('.chat-message.assistant .message-bubble').forEach(el => {
        if (!el.querySelector('h1, h2, h3, h4, h5, h6, p, ul, ol, hr')) {
            el.innerHTML = safeMarked(el.innerHTML);
        }
    });
}

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

    const text = document.createElement('span');
    text.textContent = message;

    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'snackbar-close';
    close.setAttribute('aria-label', 'Dismiss');
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

function initUploadPanel() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const progressContainer = document.getElementById('progress-container');
    const uploadPrompt = document.querySelector('.upload-prompt');
    const filenameDisplay = document.getElementById('filename-display');
    const percentDisplay = document.getElementById('percent-display');
    const progressFill = document.getElementById('progress-fill');
    const statusDisplay = document.getElementById('status-display');
    const uploadComposer = document.getElementById('upload-composer');
    const uploadNote = document.getElementById('upload-note');

    if (!dropzone) return;

    uploadComposer?.addEventListener('click', (e) => e.stopPropagation());

    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileUpload(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFileUpload(fileInput.files[0]);
    });

    function handleFileUpload(file) {
        const allowed = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!allowed.includes(file.type)) {
            showSnackbar('Unsupported format. Please upload a PNG or JPG.', 'error');
            return;
        }

        uploadPrompt.style.display = 'none';
        progressContainer.hidden = false;
        progressContainer.style.display = 'block';
        filenameDisplay.textContent = file.name;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('note', uploadNote?.value.trim() || '');

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/analyze', true);

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                const vis = Math.round(pct * 0.9);
                percentDisplay.textContent = `${vis}%`;
                progressFill.style.width = `${vis}%`;
            }
        });

        xhr.onreadystatechange = () => {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    const resp = JSON.parse(xhr.responseText);
                    if (resp.status === 'success') {
                        percentDisplay.textContent = '100%';
                        progressFill.style.width = '100%';
                        statusDisplay.innerHTML = '<i data-lucide="check"></i> Complete. Loading report...';
                        renderIcons();
                        setTimeout(() => window.location.href = '/report', 600);
                    } else {
                        resetUpload('Analysis failed: ' + resp.message);
                    }
                } else {
                    resetUpload('Server error. Status: ' + xhr.status);
                }
            }
        };

        statusDisplay.innerHTML = '<i data-lucide="loader-circle" class="icon-spin"></i> Processing...';
        renderIcons();
        xhr.send(formData);
    }

    function resetUpload(msg) {
        showSnackbar(msg, 'error');
        uploadPrompt.style.display = 'block';
        progressContainer.hidden = true;
        progressContainer.style.display = 'none';
        progressFill.style.width = '0%';
        percentDisplay.textContent = '0%';
        fileInput.value = '';
    }
}

function initChatbot() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const clearBtn = document.getElementById('clear-chat');

    if (!chatForm) return;

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.requestSubmit();
        }
    });

    scrollChatToBottom();

    clearBtn?.addEventListener('click', async () => {
        try {
            await fetch('/api/chat/clear', { method: 'POST' });
            chatMessages.innerHTML = `
                <div class="chat-message assistant">
                    <div class="msg-avatar"><i data-lucide="bot"></i></div>
                    <div class="message-bubble">Hello — I've reviewed the predictions for this case. Ask me anything about the findings.</div>
                </div>`;
            renderIcons();
        } catch { }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        appendMessageBubble('user', query);
        chatInput.value = '';

        const typingBubble = appendTypingIndicator();
        scrollChatToBottom();

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: query }),
            });
            typingBubble.remove();

            if (resp.ok) {
                const result = await resp.json();
                if (result.status === 'success') {
                    appendMessageBubble('assistant', result.response.content);
                } else {
                    showSnackbar(result.message || 'Could not get a response.', 'error');
                }
            } else {
                showSnackbar('Connection lost.', 'error');
            }
        } catch {
            typingBubble.remove();
            showSnackbar('Request timed out.', 'error');
        }
        scrollChatToBottom();
    });

    function appendMessageBubble(role, content) {
        const div = document.createElement('div');
        div.className = `chat-message ${role}`;

        if (role === 'assistant') {
            const avatar = document.createElement('div');
            avatar.className = 'msg-avatar';
            avatar.innerHTML = '<i data-lucide="bot"></i>';
            div.appendChild(avatar);
        }

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = safeMarked(content);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        renderIcons();
    }

    function appendTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'chat-message assistant';

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.innerHTML = '<i data-lucide="bot"></i>';
        div.appendChild(avatar);

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        const dots = document.createElement('div');
        dots.className = 'typing-dots';
        dots.innerHTML = '<span></span><span></span><span></span>';
        bubble.appendChild(dots);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        renderIcons();
        return div;
    }

    function scrollChatToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}
