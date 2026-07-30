document.addEventListener('DOMContentLoaded', () => {
    renderIcons();
    initUploadPanel();
    initSidebar();
    initChatbot();
    renderMarkdownMessages();
});

function renderIcons() {
    window.lucide?.createIcons({ attrs: { 'stroke-width': 1.8 } });
}

function simpleMarkdown(text) {
    try {
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        const lines = html.split('\n');
        const out = [];
        let inList = false;

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];

            const h3 = line.match(/^### (.+)$/);
            if (h3) { closeList(out, inList); inList = false; out.push('<h3>' + inline(h3[1]) + '</h3>'); continue; }

            const h2 = line.match(/^## (.+)$/);
            if (h2) { closeList(out, inList); inList = false; out.push('<h2>' + inline(h2[1]) + '</h2>'); continue; }

            const h1 = line.match(/^# (.+)$/);
            if (h1) { closeList(out, inList); inList = false; out.push('<h1>' + inline(h1[1]) + '</h1>'); continue; }

            if (/^-{3,}\s*$/.test(line)) { closeList(out, inList); inList = false; out.push('<hr>'); continue; }

            const li = line.match(/^(\s*)[*-] (.+)$/);
            if (li) {
                if (!inList) { out.push('<ul>'); inList = true; }
                out.push('<li>' + inline(li[2]) + '</li>');
                continue;
            }

            if (inList) { out.push('</ul>'); inList = false; }
            out.push(line ? '<p>' + inline(line) + '</p>' : '<br>');
        }
        if (inList) out.push('</ul>');
        return out.join('\n');
    } catch {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

function inline(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>');
}

function closeList(out, inList) {
    if (inList) out.push('</ul>');
}

function safeMarked(text) {
    return simpleMarkdown(text);
}

function showConfirmDialog(message) {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'confirm-overlay';

        const dialog = document.createElement('div');
        dialog.className = 'confirm-dialog';
        dialog.setAttribute('role', 'alertdialog');
        dialog.setAttribute('aria-modal', 'true');

        const header = document.createElement('div');
        header.className = 'confirm-dialog-header';
        const icon = document.createElement('i');
        icon.setAttribute('data-lucide', 'alert-triangle');
        const title = document.createElement('h3');
        title.className = 'confirm-dialog-title';
        title.textContent = 'Confirm action';
        header.append(icon, title);

        const msg = document.createElement('p');
        msg.className = 'confirm-dialog-message';
        msg.textContent = message;

        const actions = document.createElement('div');
        actions.className = 'confirm-dialog-actions';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'confirm-dialog-btn confirm-dialog-btn-cancel';
        cancelBtn.textContent = 'Cancel';

        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'confirm-dialog-btn confirm-dialog-btn-danger';
        confirmBtn.textContent = 'Delete';

        actions.append(cancelBtn, confirmBtn);
        dialog.append(header, msg, actions);
        overlay.append(dialog);
        document.body.appendChild(overlay);
        renderIcons();
        requestAnimationFrame(() => overlay.classList.add('is-visible'));

        const cleanup = (result) => {
            cancelBtn.removeEventListener('click', onCancel);
            confirmBtn.removeEventListener('click', onConfirm);
            overlay.classList.remove('is-visible');
            overlay.addEventListener('transitionend', () => overlay.remove(), { once: true });
            resolve(result);
        };

        const onCancel = () => cleanup(false);
        const onConfirm = () => cleanup(true);

        cancelBtn.addEventListener('click', onCancel);
        confirmBtn.addEventListener('click', onConfirm);
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) cleanup(false);
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') cleanup(false);
        }, { once: true });
    });
}

function renderMarkdownMessages() {
    document.querySelectorAll('.chat-message.assistant .message-bubble').forEach(el => {
        if (!el.querySelector('h1, h2, h3, h4, h5, h6, p, ul, ol, hr') && !/<[a-z][\s\S]*>/i.test(el.innerHTML)) {
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

function initSidebar() {
    const sidebar = document.getElementById('report-sidebar');
    const overlay = document.getElementById('rs-overlay');
    const toggleBtn = document.getElementById('rs-toggle');
    const closeBtn = document.getElementById('rs-close');

    if (toggleBtn && sidebar && overlay) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.add('open');
            overlay.classList.add('open');
        });
    }

    const closeSidebar = () => {
        sidebar?.classList.remove('open');
        overlay?.classList.remove('open');
    };

    closeBtn?.addEventListener('click', closeSidebar);
    overlay?.addEventListener('click', closeSidebar);

    // Session list item clicks & delete buttons
    const sessionList = document.getElementById('session-list');
    if (sessionList) {
        sessionList.addEventListener('click', async (e) => {
            const delBtn = e.target.closest('.rs-item-del');
            const item = e.target.closest('.rs-item');

            if (delBtn) {
                e.stopPropagation();
                const sid = delBtn.getAttribute('data-sid');
                if (!sid) return;

                if (await showConfirmDialog('Are you sure you want to delete this scan session?')) {
                    try {
                        const resp = await fetch(`/api/sessions/${sid}`, { method: 'DELETE' });
                        if (resp.ok) {
                            if (typeof CURRENT_SESSION_ID !== 'undefined' && String(CURRENT_SESSION_ID) === String(sid)) {
                                window.location.href = '/report';
                            } else {
                                item?.remove();
                                showSnackbar('Session deleted.', 'success');
                            }
                        }
                    } catch {
                        showSnackbar('Failed to delete session.', 'error');
                    }
                }
                return;
            }

            if (item) {
                const sid = item.getAttribute('data-sid');
                if (sid && (typeof CURRENT_SESSION_ID === 'undefined' || String(CURRENT_SESSION_ID) !== String(sid))) {
                    window.location.href = `/report?sid=${sid}`;
                }
            }
        });
    }
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
                        setTimeout(() => window.location.href = `/report?sid=${resp.session_id || ''}`, 600);
                    } else {
                        resetUpload('Analysis failed: ' + resp.message);
                    }
                } else {
                    let errorMsg = 'Upload failed. Status: ' + xhr.status;
                    try {
                        const errResp = JSON.parse(xhr.responseText);
                        if (errResp && errResp.message) {
                            errorMsg = errResp.message;
                        }
                    } catch (e) {}
                    resetUpload(errorMsg);
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

    if (!chatForm || !chatInput || !chatMessages) return;

    // Load Session Chat History from Backend
    const sessionId = typeof CURRENT_SESSION_ID !== 'undefined' ? CURRENT_SESSION_ID : null;
    if (sessionId) {
        fetch(`/api/sessions/${sessionId}`)
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success' && data.session && Array.isArray(data.session.history)) {
                    if (data.session.history.length > 0) {
                        chatMessages.innerHTML = '';
                        data.session.history.forEach(msg => {
                            appendMessageBubble(msg.role, msg.content);
                        });
                        scrollChatToBottom();
                    }
                }
            })
            .catch(() => {});
    }

    function sendMessage() {
        const query = chatInput.value.trim();
        if (!query) return;

        appendMessageBubble('user', query);
        chatInput.value = '';

        const typingBubble = appendTypingIndicator();
        scrollChatToBottom();

        (async () => {
            try {
                const resp = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: query, session_id: sessionId }),
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
        })();
    }

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    scrollChatToBottom();

    clearBtn?.addEventListener('click', async () => {
        if (!await showConfirmDialog('Are you sure you want to clear all chat messages? This cannot be undone.')) return;
        try {
            await fetch('/api/chat/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
            chatMessages.innerHTML = `
                <div class="chat-message assistant">
                    <div class="msg-avatar"><i data-lucide="bot"></i></div>
                    <div class="message-bubble">Hello - I've reviewed the predictions for this case. Ask me anything about the findings.</div>
                </div>`;
            renderIcons();
        } catch { }
    });

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage();
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
        if (/<[a-z][\s\S]*>/i.test(content)) {
            bubble.innerHTML = content;
        } else {
            try { bubble.innerHTML = safeMarked(content); } catch { bubble.textContent = content; }
        }
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
