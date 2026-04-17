document.addEventListener('DOMContentLoaded', () => {
  const queryInput = document.getElementById('queryInput');
  const askButton = document.getElementById('askButton');
  const responseContainer = document.getElementById('responseContainer');
  const themeToggle = document.getElementById('themeToggle');
  const newChatBtn = document.getElementById('newChatBtn');
  const chipsContainer = document.getElementById('exampleChips');

  /* ── Theme ── */
  function initTheme() {
    const saved = localStorage.getItem('theme') || 'dark';
    setTheme(saved);
  }

  function setTheme(t) {
    document.body.classList.toggle('light', t === 'light');
    document.body.classList.toggle('dark', t === 'dark');
    if (themeToggle) {
      themeToggle.innerHTML = t === 'dark'
        ? '<i class="fas fa-sun"></i>'
        : '<i class="fas fa-moon"></i>';
    }
    localStorage.setItem('theme', t);
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const next = document.body.classList.contains('dark') ? 'light' : 'dark';
      setTheme(next);
    });
  }

  /* ── New chat ── */
  if (newChatBtn) {
    newChatBtn.addEventListener('click', () => location.reload());
  }

  /* ── Example chips ── */
  const examples = [
    "How does a quantum computer differ from a classical one?",
    "What is Retrieval-Augmented Generation (RAG)?",
    "How do self-driving cars detect obstacles?",
    "What is the impact of AI on job markets?",
    "What causes auroras near the poles?",
    "How do vaccines trigger immunity?",
    "What triggered World War I?",
    "Why is the Amazon rainforest important?",
    "How can I improve my time management?",
    "What is Maslow's hierarchy of needs?",
    "What is inflation and how is it measured?",
    "What is a healthy sleep cycle?",
    "What is the Paris Climate Agreement?",
    "Who painted the Mona Lisa?",
    "What is blockchain and how does it ensure security?",
    "What is intermittent fasting?",
    "How are black holes formed?",
    "What is the stock market and how does it work?",
    "What are the benefits of yoga?",
    "What is cognitive dissonance?",
    "Explain the theory of relativity in simple terms.",
    "What is emotional intelligence?",
    "What are cryptocurrencies?",
    "How does natural language processing power chatbots?",
    "What is the meaning of life?",
    "What is 5G technology?",
    "Who was Nelson Mandela?",
    "What is mindfulness meditation?",
    "How do habits form in the brain?",
    "What is the metaverse?"
  ];

  function renderChips() {
    const picked = [...examples].sort(() => 0.5 - Math.random()).slice(0, 3);
    chipsContainer.innerHTML = picked
      .map(q => `<span class="chip">${q}</span>`)
      .join('');
    chipsContainer.querySelectorAll('.chip').forEach(c => {
      c.addEventListener('click', () => {
        queryInput.value = c.textContent;
        queryInput.focus();
      });
    });
  }

  renderChips();
  setInterval(renderChips, 10000);

  /* ── Snowflakes ── */
  function spawnFlakes() {
    const container = document.querySelector('.snowfall');
    for (let i = 0; i < 40; i++) {
      setTimeout(() => {
        const f = document.createElement('div');
        f.className = 'snowflake';
        const sz = Math.random() * 3 + 2;
        const dur = Math.random() * 10 + 6;
        f.style.cssText = `
          width:${sz}px; height:${sz}px;
          left:${Math.random() * 100}%;
          animation-duration:${dur}s;
          animation-delay:${Math.random() * 4}s;
          opacity:${Math.random() * 0.5 + 0.2};
        `;
        container.appendChild(f);
        setTimeout(() => f.remove(), (dur + 4) * 1000);
      }, i * 200);
    }
  }
  spawnFlakes();
  setInterval(spawnFlakes, 14000);

  /* ── Session ── */
  const sessionId = localStorage.getItem('chat_session_id')
    || 'session_' + Math.random().toString(36).substr(2, 9);
  localStorage.setItem('chat_session_id', sessionId);

  /* ── Helpers ── */
  function removeWelcome() {
    const w = document.querySelector('.welcome-center');
    if (w) w.remove();
  }

  function scrollBottom() {
    setTimeout(() => responseContainer.scrollTop = responseContainer.scrollHeight, 60);
  }

  function addUserMsg(text) {
    const row = document.createElement('div');
    row.className = 'msg-row user';
    row.innerHTML = `
      <div class="msg-bubble">${escHtml(text)}</div>
    `;
    responseContainer.appendChild(row);
    scrollBottom();
  }

  function addBotMsg(html) {
    const row = document.createElement('div');
    row.className = 'msg-row bot';
    row.innerHTML = `
      <div class="msg-bubble">${html}</div>
    `;
    responseContainer.appendChild(row);
    scrollBottom();
    return row;
  }

  function addTyping() {
    const row = document.createElement('div');
    row.className = 'msg-row bot';
    row.id = 'typing';
    row.innerHTML = `
      <div class="msg-bubble typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;
    responseContainer.appendChild(row);
    scrollBottom();
  }

  function removeTyping() {
    const t = document.getElementById('typing');
    if (t) t.remove();
  }

  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function formatAnswer(text) {
    if (typeof text !== 'string') text = String(text ?? 'No response received.');
    return text.split('\n').filter(l => l.trim()).map(l => `<p>${l}</p>`).join('');
  }

  /* ── Query handler ── */
  async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    queryInput.value = '';
    removeWelcome();
    addUserMsg(query);
    addTyping();

    try {
      const res = await fetch(`/query/?query=${encodeURIComponent(query)}&session_id=${sessionId}`);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      removeTyping();

      // Parse Source Tag (e.g. "[Wikipedia]\nAnswer text")
      let fullText = data.response;
      let sourceLabel = "";
      if (fullText.startsWith('[')) {
        const lines = fullText.split('\n');
        sourceLabel = lines[0].replace('[', '').replace(']', '');
        fullText = lines.slice(1).join('\n');
      }

      const formatted = formatAnswer(fullText);
      const finalHtml = sourceLabel 
        ? `<div class="source-badge">${sourceLabel}</div>${formatted}`
        : formatted;

      addBotMsg(finalHtml);
    } catch (err) {
      removeTyping();
      const row = addBotMsg(`<span style="color:#f87171">${err.message}</span>`);
      row.querySelector('.msg-bubble').classList.add('error');
    }
  }

  askButton.addEventListener('click', handleQuery);
  queryInput.addEventListener('keypress', e => { if (e.key === 'Enter') handleQuery(); });

  /* ── Init ── */
  initTheme();
});