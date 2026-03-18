import { useState, useRef, useEffect } from "react";

const API = "http://localhost:5000/api";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #f7f8fc;
    --surface: #ffffff;
    --surface2: #f0f2f8;
    --border: #e4e7f0;
    --accent: #2563eb;
    --accent-light: #dbeafe;
    --accent2: #7c3aed;
    --text: #0f172a;
    --text2: #475569;
    --text3: #94a3b8;
    --success: #059669;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.08);
    --radius: 14px;
    --radius-sm: 8px;
  }

  body {
    font-family: 'Sora', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow: hidden;
  }

  .app {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  .sidebar {
    width: 280px;
    min-width: 280px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .sidebar-header {
    padding: 24px 20px 16px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
  }

  .logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
  }

  .logo-text {
    font-size: 16px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.3px;
  }

  .logo-sub {
    font-size: 11px;
    color: var(--text3);
    font-weight: 400;
    margin-left: 46px;
    margin-top: -2px;
  }

  .new-chat-btn {
    margin: 16px 20px 0;
    padding: 10px 16px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: var(--radius-sm);
    font-family: 'Sora', sans-serif;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.2s;
    width: calc(100% - 40px);
    justify-content: center;
  }

  .new-chat-btn:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.3);
  }

  .sidebar-section {
    padding: 16px 20px 8px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  .history-list {
    flex: 1;
    overflow-y: auto;
    padding: 0 12px;
  }

  .history-list::-webkit-scrollbar { width: 4px; }
  .history-list::-webkit-scrollbar-track { background: transparent; }
  .history-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .history-item {
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background 0.15s;
    margin-bottom: 2px;
  }

  .history-item:hover { background: var(--surface2); }
  .history-item.active { background: var(--accent-light); }

  .history-q {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .history-time {
    font-size: 11px;
    color: var(--text3);
    margin-top: 2px;
  }

  .sidebar-footer {
    padding: 16px 20px;
    border-top: 1px solid var(--border);
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--success);
    font-weight: 500;
  }

  .status-dot {
    width: 7px;
    height: 7px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg);
  }

  .topbar {
    padding: 16px 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .topbar-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
  }

  .topbar-pills {
    display: flex;
    gap: 8px;
  }

  .pill {
    padding: 4px 12px;
    background: var(--surface2);
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text2);
  }

  .pill.blue {
    background: var(--accent-light);
    color: var(--accent);
  }

  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 28px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .chat-area::-webkit-scrollbar { width: 5px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .welcome {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px;
    gap: 16px;
  }

  .welcome-icon {
    width: 72px;
    height: 72px;
    background: linear-gradient(135deg, var(--accent-light), #ede9fe);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 36px;
    margin-bottom: 8px;
  }

  .welcome h1 {
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
  }

  .welcome p {
    font-size: 14px;
    color: var(--text2);
    max-width: 400px;
    line-height: 1.6;
  }

  .suggestions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
    max-width: 560px;
    width: 100%;
  }

  .suggestion {
    padding: 14px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 13px;
    font-weight: 500;
    color: var(--text2);
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
    line-height: 1.4;
  }

  .suggestion:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow);
  }

  .message {
    display: flex;
    gap: 12px;
    animation: fadeUp 0.3s ease;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .message.user { flex-direction: row-reverse; }

  .avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }

  .bubble {
    max-width: 68%;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .bubble-text {
    padding: 14px 18px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.7;
  }

  .message.ai .bubble-text {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    box-shadow: var(--shadow);
    border-top-left-radius: 4px;
  }

  .message.user .bubble-text {
    background: var(--accent);
    color: white;
    border-top-right-radius: 4px;
    align-self: flex-end;
  }

  .sources {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .sources-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }

  .source-card {
    padding: 10px 14px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text2);
    line-height: 1.5;
  }

  .source-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }

  .source-score {
    padding: 2px 8px;
    background: var(--accent-light);
    color: var(--accent);
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .source-file {
    font-size: 11px;
    color: var(--text3);
    font-family: 'JetBrains Mono', monospace;
  }

  .score-bar {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }

  .score-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    transition: width 0.6s ease;
  }

  .typing {
    display: flex;
    gap: 5px;
    padding: 14px 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    border-top-left-radius: 4px;
    width: fit-content;
    box-shadow: var(--shadow);
  }

  .dot {
    width: 7px;
    height: 7px;
    background: var(--text3);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }

  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }

  .input-area {
    padding: 20px 28px 24px;
    background: var(--surface);
    border-top: 1px solid var(--border);
  }

  .input-box {
    display: flex;
    align-items: flex-end;
    gap: 12px;
    background: var(--bg);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .input-box:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
  }

  .input-box textarea {
    flex: 1;
    border: none;
    background: transparent;
    font-family: 'Sora', sans-serif;
    font-size: 14px;
    color: var(--text);
    resize: none;
    outline: none;
    max-height: 120px;
    line-height: 1.5;
  }

  .input-box textarea::placeholder { color: var(--text3); }

  .send-btn {
    width: 38px;
    height: 38px;
    background: var(--accent);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    flex-shrink: 0;
  }

  .send-btn:hover:not(:disabled) {
    background: #1d4ed8;
    transform: scale(1.05);
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .input-hint {
    font-size: 11px;
    color: var(--text3);
    margin-top: 8px;
    text-align: center;
  }

  .pipeline-steps {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
    margin-top: 8px;
  }

  .step {
    padding: 3px 10px;
    background: var(--surface2);
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text2);
  }

  .step.done {
    background: #f0fdf4;
    color: var(--success);
  }

  .step-arrow {
    font-size: 10px;
    color: var(--text3);
  }
    .watermark {
    position: fixed;
    bottom: 16px;
    right: 20px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text3);
    font-family: 'Sora', sans-serif;
    letter-spacing: 0.3px;
    pointer-events: none;
    z-index: 999;
  }

  .watermark span {
    color: var(--accent);
  }
`;

const SUGGESTIONS = [
  "What is Retrieval-Augmented Generation?",
  "How do neural networks work?",
  "Explain machine learning simply",
  "What are vector embeddings?",
];

function timeNow() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeIdx, setActiveIdx] = useState(null);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const ask = async (question) => {
    if (!question.trim() || loading) return;
    setInput("");

    const userMsg = { role: "user", text: question, time: timeNow() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();

      const aiMsg = {
        role: "ai",
        text: data.answer,
        sources: data.sources || [],
        time: timeNow(),
      };
      setMessages((prev) => [...prev, aiMsg]);
      setHistory((prev) => [{ q: question, time: timeNow() }, ...prev]);
      setActiveIdx(0);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: "Could not connect to the API. Make sure api.py is running on port 5000.",
          sources: [],
          time: timeNow(),
        },
      ]);
    }
    setLoading(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  };

  const newChat = () => {
    setMessages([]);
    setActiveIdx(null);
    setInput("");
  };

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="logo">
              <div className="logo-icon">🔍</div>
              <span className="logo-text">SemanticRAG</span>
            </div>
            <div className="logo-sub">Powered by Endee + Groq</div>
          </div>

          <button className="new-chat-btn" onClick={newChat}>
            ✦ New Chat
          </button>

          <div className="sidebar-section">History</div>

          <div className="history-list">
            {history.length === 0 && (
              <div
                style={{
                  padding: "12px",
                  fontSize: "13px",
                  color: "var(--text3)",
                  textAlign: "center",
                }}
              >
                No conversations yet
              </div>
            )}
            {history.map((item, i) => (
              <div
                key={i}
                className={`history-item ${activeIdx === i ? "active" : ""}`}
                onClick={() => setActiveIdx(i)}
              >
                <div className="history-q">{item.q}</div>
                <div className="history-time">{item.time}</div>
              </div>
            ))}
          </div>

          <div className="sidebar-footer">
            <div className="status-badge">
              <div className="status-dot" />
              Endee + Groq Active
            </div>
          </div>
        </aside>

        <main className="main">
          <div className="topbar">
            <span className="topbar-title">AI Chat Assistant</span>
            <div className="topbar-pills">
              <span className="pill blue">Llama 3.1</span>
              <span className="pill">Endee Vector DB</span>
              <span className="pill">all-MiniLM-L6-v2</span>
            </div>
          </div>

          <div className="chat-area">
            {messages.length === 0 ? (
              <div className="welcome">
                <div className="welcome-icon">🤖</div>
                <h1>Ask me anything</h1>
                <p>
                  Powered by Endee vector search and Groq AI. Ask questions
                  about AI, programming, or any topic.
                </p>
                <div className="suggestions">
                  {SUGGESTIONS.map((s, i) => (
                    <button
                      key={i}
                      className="suggestion"
                      onClick={() => ask(s)}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div
                    className="avatar"
                    style={
                      msg.role === "ai"
                        ? {
                            background:
                              "linear-gradient(135deg, #2563eb, #7c3aed)",
                          }
                        : {
                            background: "var(--surface2)",
                            border: "1px solid var(--border)",
                          }
                    }
                  >
                    {msg.role === "ai" ? "🤖" : "👤"}
                  </div>
                  <div className="bubble">
                    <div className="bubble-text">{msg.text}</div>

                    {msg.role === "ai" &&
                      msg.sources &&
                      msg.sources.length > 0 && (
                        <div className="sources">
                          <div className="sources-label">
                            📄 Sources Retrieved ({msg.sources.length})
                          </div>
                          {msg.sources.map((src, j) => (
                            <div key={j} className="source-card">
                              <div className="source-meta">
                                <span className="source-score">
                                  {src.score}
                                </span>
                                <span className="source-file">
                                  {src.source} · chunk {src.chunk_index}
                                </span>
                              </div>
                              <div>{src.text}</div>
                              <div className="score-bar">
                                <div
                                  className="score-fill"
                                  style={{ width: `${src.score * 100}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                    {msg.role === "ai" && (
                      <div className="pipeline-steps">
                        {[
                          "Query",
                          "→",
                          "Embed",
                          "→",
                          "Endee Search",
                          "→",
                          "Groq LLM",
                          "→",
                          "Answer",
                        ].map((s, j) =>
                          s === "→" ? (
                            <span key={j} className="step-arrow">
                              {s}
                            </span>
                          ) : (
                            <span key={j} className="step done">
                              {s}
                            </span>
                          ),
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {loading && (
              <div className="message ai">
                <div
                  className="avatar"
                  style={{
                    background: "linear-gradient(135deg, #2563eb, #7c3aed)",
                  }}
                >
                  🤖
                </div>
                <div className="bubble">
                  <div className="typing">
                    <div className="dot" />
                    <div className="dot" />
                    <div className="dot" />
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="input-area">
            <div className="input-box">
              <textarea
                ref={textareaRef}
                rows={1}
                placeholder="Ask anything — AI, programming, general knowledge..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
              />
              <button
                className="send-btn"
                onClick={() => ask(input)}
                disabled={!input.trim() || loading}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="white"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <div className="input-hint">
              Press Enter to send · Shift+Enter for new line
            </div>
          </div>
        </main>
      </div>
      <div className="watermark">
        Built by <span>Kategaru Shivakumar</span>
      </div>
    </>
  );
}
