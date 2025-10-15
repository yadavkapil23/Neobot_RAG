document.addEventListener('DOMContentLoaded', () => {
  const queryInput = document.getElementById('queryInput');
  const askButton = document.getElementById('askButton');
  const responseContainer = document.getElementById('responseContainer');
  const themeToggle = document.getElementById('themeToggle');
  const exampleChipsContainer = document.querySelector('.example-chips');

  // 🎨 Apply a random light theme
  function applyRandomTheme() {
    const themes = ['theme-blue', 'theme-green', 'theme-orange', 'theme-purple'];
    const saved = localStorage.getItem('color-theme');
    const theme = saved || themes[Math.floor(Math.random() * themes.length)];
    document.body.classList.add(theme);
    localStorage.setItem('color-theme', theme);
  }

  // 🌙 Theme (dark/light) toggle
  function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.classList.toggle('dark-mode', savedTheme === 'dark');
    themeToggle.innerHTML = savedTheme === 'dark'
      ? '<i class="fas fa-sun"></i>'
      : '<i class="fas fa-moon"></i>';
  }

  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    themeToggle.innerHTML = isDark
      ? '<i class="fas fa-sun"></i>'
      : '<i class="fas fa-moon"></i>';
  });

  // 🧠 Example questions
const examples = [
  // 🌐 Tech & AI
  "How does a quantum computer differ from a classical one?",
  "What is Retrieval-Augmented Generation (RAG)?",
  "How do self-driving cars detect obstacles?",
  "What is the impact of AI on job markets?",
  "How do recommender systems work?",
  "What is Web3 and how is it different from Web2?",
  "How does natural language processing power chatbots?",
  "What is blockchain and how does it ensure security?",
  "Explain edge computing with an example.",
  "What is federated learning in AI?",

  // 🧠 Science & Space
  "What causes auroras near the poles?",
  "How do vaccines trigger immunity?",
  "Why is Pluto no longer a planet?",
  "What is antimatter?",
  "Explain the theory of relativity in simple terms.",
  "What causes lightning during a storm?",
  "What is DNA and how does it function?",
  "How are black holes formed?",
  "What is the greenhouse effect?",
  "Why do some materials conduct electricity?",

  // 🏛️ History & Politics
  "Who was the first emperor of China?",
  "What triggered World War I?",
  "Explain the Cold War in brief.",
  "What was the Silk Road and why was it important?",
  "Tell me about the Mughal Empire.",
  "What is the United Nations and what does it do?",
  "How did Gandhi influence India's independence?",
  "What is the Magna Carta?",
  "Who was Nelson Mandela?",
  "Explain the French Revolution.",

  // 🌍 Geography & Culture
  "Why is the Amazon rainforest important?",
  "What are the Seven Wonders of the World?",
  "Tell me about the culture of Japan.",
  "What are the major rivers of Africa?",
  "Why is Mount Everest so difficult to climb?",
  "What are the key differences between North and South Korea?",
  "What are traditional foods of Italy?",
  "What is Diwali and how is it celebrated?",
  "What is the origin of the Olympics?",
  "What languages are spoken in Switzerland?",

  // 📚 Education & Career
  "How can I improve my time management?",
  "What are the top universities in the world?",
  "How to write an impressive resume?",
  "Tips to crack a coding interview.",
  "What are essential skills for data analysts?",
  "How to become a product manager?",
  "What are MOOCs and how do they help?",
  "What are some popular scholarships abroad?",
  "Best ways to learn a new language?",
  "What is the difference between MBA and MTech?",

  // 💡 Philosophy & Psychology
  "What is the meaning of life?",
  "What is Maslow’s hierarchy of needs?",
  "Why do humans dream?",
  "What is existentialism?",
  "What is the placebo effect?",
  "How does the brain store memories?",
  "What causes anxiety and how can it be managed?",
  "What is cognitive dissonance?",
  "What is emotional intelligence?",
  "How do habits form in the brain?",

  // 💰 Business & Economics
  "What is inflation and how is it measured?",
  "What are cryptocurrencies?",
  "What is the stock market and how does it work?",
  "What caused the 2008 financial crisis?",
  "What is supply chain management?",
  "How does international trade work?",
  "What is GDP and why is it important?",
  "What are NFTs and why are they controversial?",
  "How do banks create money?",
  "What is microfinance?",

  // 🧘 Lifestyle & Health
  "What is a healthy sleep cycle?",
  "What are the benefits of yoga?",
  "How can I eat healthier on a budget?",
  "What is intermittent fasting?",
  "How much water should I drink daily?",
  "How does regular exercise affect mental health?",
  "What is mindfulness meditation?",
  "How to improve focus while studying?",
  "How can one build better habits?",
  "What are superfoods?",

  // 🌎 Current Affairs & Trends
  "What is the Paris Climate Agreement?",
  "What are the key outcomes of COP28?",
  "What is 5G technology?",
  "Why are electric vehicles important?",
  "What is the metaverse?",
  "How is climate change affecting the Arctic?",
  "What are the major goals of the UN's SDGs?",
  "What are smart cities?",
  "What is TikTok's global impact?",
  "What is the controversy around AI regulation?",

  // 🎨 Arts, Music & Literature
  "Who painted the Mona Lisa?",
  "What is minimalism in art?",
  "Tell me about Shakespeare’s major works.",
  "What is the difference between classical and jazz music?",
  "Who are some famous Indian authors?",
  "What is the story behind Van Gogh’s Starry Night?",
  "What are the Nobel Prizes in Literature?",
  "What is haiku poetry?",
  "Explain surrealism in art.",
  "What is calligraphy and where is it practiced?"
];


  function updateExampleChips() {
    const shuffled = [...examples].sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, 3);
    exampleChipsContainer.innerHTML = selected.map(q => `<span class="chip">${q}</span>`).join('');

    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        queryInput.value = chip.textContent;
        queryInput.focus();
      });
    });
  }

  // ❄️ Snowflake animation
  function createSnowflakes() {
    const snowfall = document.querySelector('.snowfall');
    const count = 50;
    for (let i = 0; i < count; i++) {
      setTimeout(() => {
        const flake = document.createElement('div');
        flake.className = 'snowflake';
        const size = Math.random() * 4 + 2;
        const posX = Math.random() * window.innerWidth;
        const duration = Math.random() * 10 + 5;
        const delay = Math.random() * 5;
        const opacity = Math.random() * 0.5 + 0.3;

        flake.style.width = `${size}px`;
        flake.style.height = `${size}px`;
        flake.style.left = `${posX}px`;
        flake.style.animationDuration = `${duration}s`;
        flake.style.animationDelay = `${delay}s`;
        flake.style.opacity = opacity;

        snowfall.appendChild(flake);
        setTimeout(() => flake.remove(), duration * 1000);
      }, i * 300);
    }
  }

  // 🔍 Query handler
  async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    responseContainer.innerHTML = `
      <div class="loading-state">
        <div class="loader"></div>
        <p class="loading-text">Searching knowledge base...</p>
      </div>
    `;

    try {
      const res = await fetch(`/query/?query=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();

      responseContainer.innerHTML = `
        <div class="answer-container">
          <div class="question-text">
            <i class="fas fa-question-circle"></i> ${data.query}
          </div>
          <div class="answer-text">${formatAnswer(data.response)}</div>
        </div>
      `;
    } catch (err) {
      responseContainer.innerHTML = `
        <div class="answer-container error">
          <div class="question-text">
            <i class="fas fa-exclamation-triangle"></i> Error
          </div>
          <div class="answer-text">
            Failed to get response: ${err.message}
          </div>
        </div>
      `;
    }
  }

function formatAnswer(text) {
  if (typeof text !== "string") {
    text = String(text ?? "No response received.");
  }
  return text
    .split('\n')
    .filter(line => line.trim())
    .map(line => `<p>${line}</p>`)
    .join('');
}

  // 🔗 Event listeners
  askButton.addEventListener('click', handleQuery);
  queryInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') handleQuery();
  });

function applyRandomTheme() {
  const themes = ['theme-blue', 'theme-green', 'theme-orange', 'theme-purple', 'theme-pink', 'theme-teal'];
  const currentTheme = localStorage.getItem('color-theme');
  let selectedTheme = currentTheme;

  if (!selectedTheme) {
    selectedTheme = themes[Math.floor(Math.random() * themes.length)];
    localStorage.setItem('color-theme', selectedTheme);
  }

  document.body.classList.add(selectedTheme);
}


  // 🔄 Init everything
  applyRandomTheme();
  initTheme();
  updateExampleChips();
  setInterval(updateExampleChips, 10000); // refresh chips every 10s
  createSnowflakes();
  setInterval(createSnowflakes, 15000); // new flakes every 15s
});
