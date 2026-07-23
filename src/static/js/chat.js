let currentSessionId = null;
let sessions = [];

document.addEventListener('DOMContentLoaded', async () => {
    initTheme();
    initSidebar();
    initChat();
    await initNameModal();
});

function initTheme() {
    const html = document.documentElement;
    const sun = document.getElementById('theme-icon-sun');
    const moon = document.getElementById('theme-icon-moon');
    const btn = document.getElementById('theme-toggle');

    const stored = localStorage.getItem('theme');
    if (stored === 'dark' || stored === 'light') {
        html.setAttribute('data-theme', stored);
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        html.setAttribute('data-theme', 'dark');
    } else {
        html.setAttribute('data-theme', 'light');
    }
    updateThemeIcons();

    if (btn) {
        btn.addEventListener('click', () => {
            const current = html.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            updateThemeIcons();
        });
    }

    function updateThemeIcons() {
        const isDark = html.getAttribute('data-theme') === 'dark';
        if (sun) sun.style.display = isDark ? 'none' : 'block';
        if (moon) moon.style.display = isDark ? 'block' : 'none';
    }
}

async function initNameModal() {
    const modal = document.getElementById('name-modal');
    const form = document.getElementById('name-form');
    const input = document.getElementById('name-input');

    if (!modal || modal.style.display === 'none') {
        await initApp();
        return;
    }

    return new Promise((resolve) => {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const name = input.value.trim() || 'Guest';
            try {
                const res = await fetch('/api/set-name', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name }),
                });
                const data = await res.json();
                updateNameUI(data.status === 'success' ? data.name : name);
            } catch {
                updateNameUI(name);
            }
            modal.style.display = 'none';
            await initApp();
            resolve();
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') form.dispatchEvent(new Event('submit'));
        });

        setTimeout(() => input.focus(), 300);
    });
}

function updateNameUI(name) {
    const els = document.querySelectorAll('#user-name-display, #sidebar-name');
    els.forEach(el => el.textContent = name);
    const avatar = document.getElementById('sidebar-avatar');
    if (avatar) avatar.textContent = name.charAt(0).toUpperCase();
}

async function initApp() {
    await loadSessions();
    if (sessions.length > 0) {
        await switchSession(sessions[0].id);
    } else {
        await createNewSession();
    }
    initSidebarToggle();
}

function initSidebar() {
    const avatar = document.getElementById('sidebar-avatar');
    const name = document.getElementById('sidebar-name');
    if (avatar && name) {
        const n = name.textContent || 'Guest';
        avatar.textContent = n.charAt(0).toUpperCase();
    }
}

function initSidebarToggle() {
    const toggle = document.getElementById('toggle-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const sidebar = document.getElementById('sidebar');
    const layout = document.querySelector('.app-layout');

    if (toggle) {
        toggle.addEventListener('click', () => {
            const isMobile = window.innerWidth <= 768;
            if (isMobile) {
                sidebar.classList.toggle('open');
                overlay.classList.toggle('open');
            } else {
                sidebar.classList.toggle('collapsed');
                layout.classList.toggle('sidebar-collapsed');
            }
        });
    }
    if (overlay) {
        overlay.addEventListener('click', () => {
            sidebar.classList.remove('open');
            overlay.classList.remove('open');
        });
    }
}

function closeSidebarMobile() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('open');
        overlay.classList.remove('open');
    }
}

async function loadSessions() {
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        sessions = data.sessions || [];
        renderSessionList();
    } catch {}
}

function renderSessionList() {
    const list = document.getElementById('session-list');
    if (!list) return;

    if (sessions.length === 0) {
        list.innerHTML = '<div style="padding:1rem;text-align:center;color:var(--text-muted);font-size:0.82rem;">No conversations yet</div>';
        return;
    }

    list.innerHTML = sessions.map(s => {
        const active = s.id === currentSessionId ? 'active' : '';
        const title = s.title || 'New chat';
        return `
            <div class="session-item ${active}" data-id="${s.id}">
                <svg class="session-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                <span class="session-title">${escapeHtml(title)}</span>
                <button class="session-delete" data-id="${s.id}" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            </div>
        `;
    }).join('');

    list.querySelectorAll('.session-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (e.target.closest('.session-delete')) return;
            switchSession(item.dataset.id);
        });
    });

    list.querySelectorAll('.session-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const id = btn.dataset.id;
            try {
                await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
                if (currentSessionId === id) {
                    sessions = sessions.filter(s => s.id !== id);
                    if (sessions.length > 0) {
                        await switchSession(sessions[0].id);
                    } else {
                        await createNewSession();
                    }
                } else {
                    sessions = sessions.filter(s => s.id !== id);
                    renderSessionList();
                }
            } catch {}
        });
    });
}

async function createNewSession() {
    try {
        const res = await fetch('/api/sessions', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            currentSessionId = data.session_id;
            await loadSessions();
            document.getElementById('chat-messages').innerHTML = '';
            showWelcome();
        }
    } catch {}
}

async function switchSession(sid) {
    if (sid === currentSessionId) return;
    currentSessionId = sid;
    renderSessionList();
    try {
        const res = await fetch(`/api/sessions/${sid}`);
        const data = await res.json();
        if (data.status === 'success') {
            renderHistory(data.session.history || []);
        }
    } catch {}
    closeSidebarMobile();
}

function renderHistory(history) {
    const container = document.getElementById('chat-messages');
    container.innerHTML = '';
    if (history.length === 0) {
        showWelcome();
    } else {
        hideWelcome();
        history.forEach(msg => {
            appendMessageBubble(msg.role, msg.content);
        });
    }
    scrollToBottom();
}

async function sendQuery(query) {
    if (!query || !currentSessionId) return;

    appendMessageBubble('user', escapeHtml(query));

    const input = document.getElementById('chat-input');
    input.value = '';
    input.disabled = true;
    document.getElementById('send-btn').disabled = true;

    const typingEl = appendTypingIndicator();
    scrollToBottom();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: query, session_id: currentSessionId }),
        });

        typingEl.remove();

        try {
            const data = await res.json();
            if (data.status === 'success') {
                appendMessageBubble('assistant', data.response.content);
                await loadSessions();
            } else {
                showErrorMessage(data.message || 'Could not get an answer.');
            }
        } catch {
            showErrorMessage('Connection lost with the server.');
        }
    } catch {
        typingEl.remove();
        showErrorMessage('Request timed out. Please try again.');
    }

    hideWelcome();
    input.disabled = false;
    document.getElementById('send-btn').disabled = false;
    input.focus();
    scrollToBottom();
}

function hideWelcome() {
    const el = document.getElementById('chat-welcome');
    if (el) el.style.display = 'none';
}

function showWelcome() {
    const el = document.getElementById('chat-welcome');
    if (el) el.style.display = '';
}

function initChat() {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const chips = document.getElementById('suggestion-chips');
    const newChatBtn = document.getElementById('new-chat-btn');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await sendQuery(input.value.trim());
    });

    if (chips) {
        chips.addEventListener('click', (e) => {
            const chip = e.target.closest('.chip');
            if (chip) {
                sendQuery(chip.dataset.prompt);
            }
        });
    }

    if (newChatBtn) {
        newChatBtn.addEventListener('click', createNewSession);
    }
}

function appendMessageBubble(role, content) {
    const container = document.getElementById('chat-messages');
    const wrap = document.createElement('div');
    wrap.className = `chat-message-wrap ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (role === 'assistant') {
        bubble.innerHTML = renderMarkdown(content);
        bubble.querySelectorAll('pre code').forEach(addCopyButton);
    } else {
        bubble.textContent = content;
    }

    const actions = document.createElement('div');
    actions.className = 'message-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'msg-copy-btn';
    copyBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    copyBtn.title = 'Copy message';
    copyBtn.addEventListener('click', async () => {
        const text = role === 'assistant' ? bubble.textContent : content;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
        copyBtn.classList.add('copied');
        setTimeout(() => copyBtn.classList.remove('copied'), 1500);
    });

    actions.appendChild(copyBtn);
    wrap.appendChild(bubble);
    wrap.appendChild(actions);
    container.appendChild(wrap);
}

function appendTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const wrap = document.createElement('div');
    wrap.className = 'chat-message-wrap assistant';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const dots = document.createElement('div');
    dots.className = 'typing-indicator';
    dots.innerHTML = '<span></span><span></span><span></span>';

    bubble.appendChild(dots);
    wrap.appendChild(bubble);
    container.appendChild(wrap);
    return wrap;
}

function showErrorMessage(msg) {
    const container = document.getElementById('chat-messages');
    const wrap = document.createElement('div');
    wrap.className = 'chat-message-wrap assistant';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.style.color = '#f85149';
    bubble.textContent = msg;
    wrap.appendChild(bubble);
    container.appendChild(wrap);
    scrollToBottom();
}

function scrollToBottom() {
    const container = document.getElementById('chat-messages');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

function closeSidebarMobile() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    sidebar.classList.remove('open');
    overlay.classList.remove('open');
}

function renderMarkdown(text) {
    const esc = (s) => {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    };

    text = text
        .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const cls = lang ? ` class="language-${esc(lang)}"` : '';
            return `<pre><code${cls}>${esc(code.trim())}</code></pre>`;
        })
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    text = text.replace(/(<li>[\s\S]*?)(<\/li>)/g, '$1$2');
    text = text.replace(/((?:<li>.*?<\/li>\s*)+)/g, (match) => {
        const items = match.match(/<li>.*?<\/li>/g) || [];
        const isOrdered = /^\d+\./.test(match);
        const tag = isOrdered ? 'ol' : 'ul';
        return `<${tag}>${items.join('')}</${tag}>`;
    });

    text = text.replace(/\n{2,}/g, '</p><p>');
    text = text.replace(/\n/g, '<br>');
    text = text.replace(/<br><\/(ul|ol)>/g, '</$1>');
    text = text.replace(/<\/(ul|ol)><br>/g, '</$1>');

    return `<p>${text}</p>`;
}

function addCopyButton(codeBlock) {
    const pre = codeBlock.parentElement;
    if (pre.querySelector('.copy-btn')) return;

    const header = document.createElement('div');
    header.className = 'code-header';

    const lang = codeBlock.className.replace('language-', '') || 'code';
    const label = document.createElement('span');
    label.className = 'code-lang';
    label.textContent = lang;

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';

    btn.addEventListener('click', async () => {
        const text = codeBlock.textContent;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
        }, 2000);
    });

    header.appendChild(label);
    header.appendChild(btn);
    pre.insertBefore(header, codeBlock);
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
