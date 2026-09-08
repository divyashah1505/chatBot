import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const generateUserId = () => {
  return "user_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
};

// SVG Icon Library
const Icons = {
  Sparkle: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
    </svg>
  ),
  User: () => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  ),
  Plus: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14M12 5v14" />
    </svg>
  ),
  Paperclip: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  ),
  Mic: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" x2="12" y1="19" y2="22" />
    </svg>
  ),
  Send: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="m5 12 7-7 7 7" />
      <path d="M12 19V5" />
    </svg>
  ),
  Copy: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
    </svg>
  ),
  Check: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  Trash: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  ),
  Chat: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  ),
  Sidebar: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="18" height="18" x="3" y="3" rx="2" />
      <path d="M9 3v18" />
    </svg>
  ),
  Briefcase: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="20" height="14" x="2" y="7" rx="2" ry="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  ),
  GraduationCap: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 10v6M2 10l10-5 10 5-10 5z" />
      <path d="M6 12v5c3 3 9 3 12 0v-5" />
    </svg>
  ),
  FileDoc: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  ),
  FileText: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
      <path d="M10 9H8" />
      <path d="M16 13H8" />
      <path d="M16 17H8" />
    </svg>
  ),
  ArrowUpRight: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M7 17 17 7M7 7h10v10" />
    </svg>
  ),
  Shield: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  ),
  Stethoscope: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3" />
      <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4" />
      <circle cx="20" cy="10" r="2" />
    </svg>
  ),
  Palette: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="13.5" cy="6.5" r=".5" fill="currentColor" />
      <circle cx="17.5" cy="10.5" r=".5" fill="currentColor" />
      <circle cx="8.5" cy="7.5" r=".5" fill="currentColor" />
      <circle cx="6.5" cy="12.5" r=".5" fill="currentColor" />
      <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z" />
    </svg>
  ),
  Cpu: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
    </svg>
  ),
  Server: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="20" height="8" x="2" y="2" rx="2" ry="2" />
      <rect width="20" height="8" x="2" y="14" rx="2" ry="2" />
      <line x1="6" x2="6.01" y1="6" y2="6" />
      <line x1="6" x2="6.01" y1="18" y2="18" />
    </svg>
  ),
  Refresh: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 21h5v-5" />
    </svg>
  ),
  ChevronDown: () => (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
  Cross: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  ),
  Zap: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  ),
  Sliders: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="4" x2="4" y1="21" y2="14" />
      <line x1="4" x2="4" y1="10" y2="3" />
      <line x1="12" x2="12" y1="21" y2="12" />
      <line x1="12" x2="12" y1="8" y2="3" />
      <line x1="20" x2="20" y1="21" y2="16" />
      <line x1="20" x2="20" y1="12" y2="3" />
      <line x1="1" x2="7" y1="14" y2="14" />
      <line x1="9" x2="15" y1="8" y2="8" />
      <line x1="17" x2="23" y1="16" y2="16" />
    </svg>
  ),
  Volume: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    </svg>
  ),
  VolumeX: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <line x1="22" x2="16" y1="9" y2="15" />
      <line x1="16" x2="22" y1="9" y2="15" />
    </svg>
  ),
  RotateCcw: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  ),
  ThumbsUp: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
    </svg>
  ),
  ThumbsDown: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 14V2" />
      <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
    </svg>
  )
};

// Curated Designer Themes
const THEMES = [
  { id: "mocha", name: "Mocha & Cream", tag: "Brown & White", desc: "Warm porcelain white with rich chocolate espresso", icon: "☕", color: "#6d422a" },
  { id: "espresso", name: "Velvet Espresso", tag: "Deep Brown", desc: "Roasted dark walnut with crisp white contrast", icon: "🤎", color: "#b06e40" },
  { id: "daylight", name: "Pristine Daylight", tag: "Light Studio", desc: "Clean porcelain studio with crisp contrast", icon: "☀️", color: "#0284c7" },
  { id: "obsidian", name: "Obsidian Aurora", tag: "Classic Dark", desc: "Velvet onyx with cyan-indigo aurora glow", icon: "✨", color: "#3b82f6" },
  { id: "sapphire", name: "Midnight Sapphire", tag: "Deep Navy", desc: "Oceanic navy with royal sapphire accents", icon: "🌌", color: "#2563eb" },
  { id: "emerald", name: "Cyber Emerald", tag: "High-Tech", desc: "Deep carbon with luminous cyber emerald", icon: "⚡", color: "#10b981" },
  { id: "amethyst", name: "Nebula Amethyst", tag: "Cosmic", desc: "Royal twilight with vibrant violet & lavender", icon: "🔮", color: "#8b5cf6" }
];

// Interactive Code Block with Language Tag and Copy Code Action
const CodeBlock = ({ className, children }) => {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || "");
  const language = match ? match[1] : "code";
  const rawCode = String(children || "").replace(/\n$/, "");

  const handleCopyCode = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(rawCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="ai-code-card">
      <div className="ai-code-header">
        <span className="ai-code-lang">
          <span className="code-lang-dot"></span>
          {language}
        </span>
        <button
          type="button"
          className={`ai-code-copy-btn ${copied ? "copied" : ""}`}
          onClick={handleCopyCode}
          title="Copy code to clipboard"
        >
          {copied ? (
            <>
              <Icons.Check />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Icons.Copy />
              <span>Copy code</span>
            </>
          )}
        </button>
      </div>
      <div className="ai-code-body">
        <pre className="ai-code-pre">
          <code className={className}>{rawCode}</code>
        </pre>
      </div>
    </div>
  );
};

// Full State-of-the-Art Markdown Renderer for AI Responses
const FormattedText = ({ text }) => {
  if (!text) return null;

  return (
    <div className="ai-markdown-output">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const hasNewline = String(children || "").includes("\n");
            const isCodeBlock = Boolean(className) || hasNewline;
            if (!isCodeBlock) {
              return (
                <code className="ai-inline-badge" {...props}>
                  {children}
                </code>
              );
            }
            return <CodeBlock className={className}>{children}</CodeBlock>;
          },
          table({ children }) {
            return (
              <div className="ai-table-wrapper">
                <table className="ai-markdown-table">{children}</table>
              </div>
            );
          },
          blockquote({ children }) {
            return (
              <blockquote className="ai-callout-quote">
                <div className="ai-quote-accent"></div>
                <div className="ai-quote-content">{children}</div>
              </blockquote>
            );
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="ai-markdown-link"
              >
                {children}
                <span className="ai-link-icon"><Icons.ArrowUpRight /></span>
              </a>
            );
          },
          h1: ({ children }) => <h2 className="ai-heading-1">{children}</h2>,
          h2: ({ children }) => <h3 className="ai-heading-2">{children}</h3>,
          h3: ({ children }) => <h4 className="ai-heading-3">{children}</h4>,
          h4: ({ children }) => <h5 className="ai-heading-4">{children}</h5>,
          hr: () => <hr className="ai-markdown-divider" />,
          ul: ({ children }) => <ul className="ai-markdown-ul">{children}</ul>,
          ol: ({ children }) => <ol className="ai-markdown-ol">{children}</ol>,
          li: ({ children }) => <li className="ai-markdown-li">{children}</li>,
          p: ({ children }) => <p className="ai-markdown-p">{children}</p>
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
};

const getTimeGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
};

function App() {
  const [userId, setUserId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Theme & Model Studio state (Default: Mocha & Cream - Brown & White)
  const [currentTheme, setCurrentTheme] = useState(() => {
    const saved = localStorage.getItem("divya_theme");
    if (!saved || saved === "obsidian") {
      localStorage.setItem("divya_theme", "mocha");
      return "mocha";
    }
    return saved;
  });
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false);
  const [pingLatency, setPingLatency] = useState(null);
  const [isPinging, setIsPinging] = useState(false);

  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [feedbackState, setFeedbackState] = useState({});
  const [modelStatus, setModelStatus] = useState(null);
  const [docFaqs, setDocFaqs] = useState([]);
  const [activeDocName, setActiveDocName] = useState(null);

  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleSelectTheme = (themeId) => {
    setCurrentTheme(themeId);
    localStorage.setItem("divya_theme", themeId);
    document.documentElement.setAttribute("data-theme", themeId);
    setIsThemeMenuOpen(false);
  };

  const handlePingModel = async () => {
    setIsPinging(true);
    const start = performance.now();
    try {
      await fetchModelStatus();
      const end = performance.now();
      setPingLatency(Math.round(end - start));
    } catch {
      setPingLatency("Err");
    } finally {
      setIsPinging(false);
    }
  };

  const fetchModelStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/models/status`);
      if (res.ok) {
        const data = await res.json();
        setModelStatus(data);
      }
    } catch (e) {
      console.warn("Model status check skipped", e);
    }
  };

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", currentTheme);
  }, [currentTheme]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        setIsModelModalOpen(false);
        setIsThemeMenuOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    let storedUserId = localStorage.getItem("chat_user_id");
    if (!storedUserId) {
      storedUserId = generateUserId();
      localStorage.setItem("chat_user_id", storedUserId);
    }
    setUserId(storedUserId);
    fetchSessions(storedUserId);
    fetchModelStatus();

    const statusTimer = setInterval(fetchModelStatus, 8000);

    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event) => {
        let currentTranscript = "";
        for (let i = 0; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setMessage(currentTranscript);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => clearInterval(statusTimer);
  }, []);

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      setMessage("");
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const fetchSessions = async (uid) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${uid}`);
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
      }
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  const loadSession = async (sessionId) => {
    if (loading) return;
    setCurrentSessionId(sessionId);
    setMessages([]);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${userId}/${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        const loadedMessages = data.messages.map(m => ({ ...m, isNew: false }));
        setMessages(loadedMessages);
      }
    } catch (e) {
      console.error("Failed to fetch session details", e);
    } finally {
      setLoading(false);
    }
  };

  const startNewChat = () => {
    setCurrentSessionId(null);
    setMessages([]);
    setDocFaqs([]);
    setActiveDocName(null);
  };

  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
        method: "DELETE"
      });
      if (res.ok) {
        setSessions(sessions.filter(s => s.id !== sessionId));
        if (currentSessionId === sessionId) {
          setCurrentSessionId(null);
          setMessages([]);
          setDocFaqs([]);
          setActiveDocName(null);
        }
      }
    } catch (error) {
      console.error("Failed to delete session", error);
    }
  };

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleSpeak = (text, index) => {
    if (!('speechSynthesis' in window)) return;
    if (speakingIndex === index) {
      window.speechSynthesis.cancel();
      setSpeakingIndex(null);
      return;
    }
    window.speechSynthesis.cancel();
    const cleanText = text
      .replace(/```[\s\S]*?```/g, "Code block omitted.")
      .replace(/[#*`_~[\]()]/g, "")
      .trim();
    const utter = new SpeechSynthesisUtterance(cleanText);
    utter.rate = 1.0;
    utter.onend = () => setSpeakingIndex(null);
    utter.onerror = () => setSpeakingIndex(null);
    window.speechSynthesis.speak(utter);
    setSpeakingIndex(index);
  };

  const handleRegenerate = (index) => {
    if (loading) return;
    const prevUser = [...messages.slice(0, index)].reverse().find(m => m.sender === "user");
    if (prevUser && prevUser.text) {
      sendMessage(prevUser.text);
    }
  };

  const handleFeedback = (index, type) => {
    setFeedbackState(prev => ({
      ...prev,
      [index]: prev[index] === type ? null : type
    }));
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (customPrompt = null) => {
    const textToSend = customPrompt || message.trim();
    if ((!textToSend && !selectedDoc) || loading || !userId) return;

    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    }

    const currentMessage = textToSend;
    const currentDoc = selectedDoc;

    // Clear inputs immediately
    setMessage("");
    setSelectedDoc(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setLoading(true);

    // --- Document + prompt flow ---
    if (currentDoc) {
      const userText = currentMessage
        ? `📎 Attached: ${currentDoc.name} — "${currentMessage}"`
        : `📎 Attached: ${currentDoc.name} — "Analyze this document."`;
      setMessages((prev) => [...prev, { sender: "user", text: userText, isNew: false }]);

      const formData = new FormData();
      formData.append("user_id", userId);
      formData.append("file", currentDoc.file);
      formData.append("prompt", currentMessage || "Analyze this document.");
      if (currentSessionId) formData.append("session_id", currentSessionId);

      try {
        const res = await fetch(`${API_BASE_URL}/api/summarize-document`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Upload failed");
        const data = await res.json();
        setMessages((prev) => [...prev, { sender: "bot", text: data.reply, isNew: true }]);
        if (data.faqs && data.faqs.length > 0) {
          setDocFaqs(data.faqs);
          setActiveDocName(data.filename || currentDoc.name);
        }
        if (!currentSessionId) {
          setCurrentSessionId(data.session_id);
          fetchSessions(userId);
        }
      } catch (err) {
        console.error("Document upload error:", err);
        setMessages((prev) => [...prev, { sender: "bot", text: "⚠️ Failed to process the document. Please verify the backend service is active.", isNew: true }]);
      } finally {
        setLoading(false);
      }
      return;
    }

    // --- Normal chat flow ---
    setMessages((prev) => [...prev, { sender: "user", text: currentMessage, isNew: false }]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: currentMessage, session_id: currentSessionId })
      });
      if (!response.ok) throw new Error("Server error");
      const data = await response.json();
      setMessages((prev) => [...prev, { sender: "bot", text: data.reply, isNew: true }]);
      if (!currentSessionId) {
        setCurrentSessionId(data.session_id);
        fetchSessions(userId);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [...prev, { sender: "bot", text: "⚠️ Unable to connect to the assistant server. Please ensure the backend is running.", isNew: true }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const selectDoc = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const sizeInKb = (file.size / 1024).toFixed(1) + " KB";
    setSelectedDoc({ file, name: file.name, size: sizeInKb });
  };

  return (
    <div className="workspace" data-theme={currentTheme}>
      {/* 1. Permanent Left Navigation Sidebar */}
      <aside className={`workspace-sidebar ${isSidebarOpen ? "expanded" : "collapsed"}`}>
        <div className="sidebar-header">
          <div className="brand-badge">
            <div className="brand-icon">
              <Icons.Sparkle />
            </div>
            {isSidebarOpen && (
              <div className="brand-text-wrap">
                <span className="brand-name">Divya Assistant</span>
                <span className="brand-tag">Enterprise AI</span>
              </div>
            )}
          </div>
          <button 
            className="sidebar-toggle-btn" 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            title={isSidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
          >
            <Icons.Sidebar />
          </button>
        </div>

        <div className="sidebar-action-wrap">
          <button className="new-conversation-btn" onClick={startNewChat} title="New Chat">
            <Icons.Plus />
            {isSidebarOpen && <span>New Chat</span>}
          </button>
        </div>

        <div className="conversations-scroll">
          {isSidebarOpen && <div className="section-label">Chat History</div>}
          {sessions.map((s) => (
            <div 
              key={s.id} 
              className={`conv-item ${currentSessionId === s.id ? "active" : ""}`}
              onClick={() => loadSession(s.id)}
              title={s.title}
            >
              <span className="conv-icon">
                <Icons.Chat />
              </span>
              {isSidebarOpen && <span className="conv-title">{s.title}</span>}
              {isSidebarOpen && (
                <button 
                  className="conv-del-btn" 
                  onClick={(e) => handleDeleteSession(e, s.id)}
                  title="Delete conversation"
                >
                  <Icons.Trash />
                </button>
              )}
            </div>
          ))}
          {sessions.length === 0 && isSidebarOpen && (
            <div className="empty-history">
              <p>No recent conversations</p>
            </div>
          )}
        </div>

        {isSidebarOpen && (
          <div className="sidebar-user-footer">
            <div className="user-pill" onClick={() => setIsModelModalOpen(true)} style={{ cursor: "pointer" }} title="Click to view AI Runtime Studio">
              <div className="user-icon-circle">
                <Icons.Cpu />
              </div>
              <div className="user-details">
                <span className="user-label">AI Engine Studio</span>
                <span className="user-status-dot">
                  ● {modelStatus?.is_llm_online ? "Local LLM Online" : "Neural Vector RAG Active"}
                </span>
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* 2. Main Chat Stage */}
      <main className="workspace-main">
        <header className="workspace-header">
          <div className="header-breadcrumbs">
            {!isSidebarOpen && (
              <button 
                className="mobile-expand-btn" 
                onClick={() => setIsSidebarOpen(true)}
                title="Open Sidebar"
              >
                <Icons.Sidebar />
              </button>
            )}
            <span className="crumb-main">Divya Assistant</span>
            <span className="crumb-divider">/</span>
            <span className="crumb-sub">
              {currentSessionId ? "Document & Chat Workspace" : "New Query"}
            </span>
          </div>

          <div className="header-action-group">
            {/* 1. Theme Switcher Dropdown */}
            <div className="theme-switcher-wrapper">
              <button
                className="header-theme-btn"
                onClick={() => setIsThemeMenuOpen(!isThemeMenuOpen)}
                title="Switch Theme"
              >
                <span className="theme-btn-icon">
                  {THEMES.find((t) => t.id === currentTheme)?.icon || "✨"}
                </span>
                <span className="theme-btn-label">
                  {THEMES.find((t) => t.id === currentTheme)?.name || "Theme"}
                </span>
                <Icons.ChevronDown />
              </button>

              {isThemeMenuOpen && (
                <div className="theme-dropdown-menu">
                  <div className="theme-dropdown-header">Workspace Theme</div>
                  {THEMES.map((th) => (
                    <button
                      key={th.id}
                      className={`theme-option-item ${currentTheme === th.id ? "active" : ""}`}
                      onClick={() => handleSelectTheme(th.id)}
                    >
                      <span className="theme-opt-dot" style={{ backgroundColor: th.color }}></span>
                      <span className="theme-opt-icon">{th.icon}</span>
                      <div className="theme-opt-text">
                        <span className="theme-opt-name">{th.name}</span>
                        <span className="theme-opt-tag">{th.tag}</span>
                      </div>
                      {currentTheme === th.id && (
                        <span className="theme-opt-check">
                          <Icons.Check />
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* 2. Interactive AI Model Status & Studio Trigger */}
            <button
              className={`header-model-btn ${modelStatus?.is_llm_online ? "model-online" : "model-offline"}`}
              onClick={() => setIsModelModalOpen(true)}
              title="Click to open AI Model & Runtime Studio"
            >
              <span className={`indicator-dot ${modelStatus?.is_llm_online ? "online" : "fallback"}`}></span>
              <span className="model-btn-provider">
                {modelStatus?.is_llm_online
                  ? (modelStatus.active_provider === "gemini"
                      ? "✨ Google Gemini"
                      : modelStatus.active_provider === "ollama" ? "🦙 Ollama" : "🤖 LM Studio")
                  : "⚡ Local Neural RAG"}
              </span>
              <span className="model-chip-name">
                {(modelStatus?.active_model || "100% Offline").split(":")[0]}
              </span>
              <span className="model-studio-badge">
                <Icons.Sliders />
                <span>Studio</span>
              </span>
            </button>
          </div>
        </header>

        <div className="workspace-chat-scroll">
          {messages.length === 0 ? (
            <div className="welcome-hero-stage">
              <div className="hero-center-badge">
                <div className="hero-avatar">
                  <Icons.Sparkle />
                </div>
                <h1 className="hero-heading">{getTimeGreeting()}, Divya ✨</h1>
                <p className="hero-subheading">
                  I can analyze any uploaded policy, contract, or academic document, rank candidate profiles, and answer your complex questions instantly.
                </p>
              </div>

              <div className="bento-container">
                <div 
                  className="bento-item" 
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="bento-top">
                    <div className="bento-icon-box blue">
                      <Icons.Stethoscope />
                    </div>
                    <span className="bento-link-icon">
                      <Icons.Paperclip />
                    </span>
                  </div>
                  <h3 className="bento-title">Doctor's Prescription & Medicine Verification</h3>
                  <p className="bento-desc">Upload doctor handwritten prescriptions (Images/PDF) to predict medicines, verify dosages, and correct typos.</p>
                </div>

                <div 
                  className="bento-item" 
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="bento-top">
                    <div className="bento-icon-box purple">
                      <Icons.Shield />
                    </div>
                    <span className="bento-link-icon">
                      <Icons.Paperclip />
                    </span>
                  </div>
                  <h3 className="bento-title">Policy & Document Analysis</h3>
                  <p className="bento-desc">Upload health insurance, agreements, or reports for instant schedule & coverage extraction.</p>
                </div>

                <div 
                  className="bento-item" 
                  onClick={() => sendMessage("I want AI ML best candidates in their skills and experience wise")}
                >
                  <div className="bento-top">
                    <div className="bento-icon-box blue">
                      <Icons.Briefcase />
                    </div>
                    <span className="bento-link-icon">
                      <Icons.ArrowUpRight />
                    </span>
                  </div>
                  <h3 className="bento-title">Candidate Intelligence</h3>
                  <p className="bento-desc">Search top talent in AI/ML, Full Stack, and evaluate resumes.</p>
                </div>

                <div 
                  className="bento-item" 
                  onClick={() => sendMessage("Tell me about MCA degree eligibility, syllabus, and fees")}
                >
                  <div className="bento-top">
                    <div className="bento-icon-box indigo">
                      <Icons.GraduationCap />
                    </div>
                    <span className="bento-link-icon">
                      <Icons.ArrowUpRight />
                    </span>
                  </div>
                  <h3 className="bento-title">Academic Programs</h3>
                  <p className="bento-desc">Review MCA, BCA, MBA, BBA fees, eligibility, and course outlines.</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="message-stream-container">
              {messages.map((msg, index) => (
                <div key={index} className={`stream-row ${msg.sender}`}>
                  <div className="stream-avatar">
                    {msg.sender === "user" ? <Icons.User /> : <Icons.Sparkle />}
                  </div>
                  <div className="stream-content">
                    <div className="stream-bubble">
                      <FormattedText text={msg.text} />
                    </div>
                    {msg.sender === "bot" && (
                      <div className="stream-actions">
                        <button 
                          className="action-copy-btn" 
                          onClick={() => copyToClipboard(msg.text, index)}
                          title="Copy response"
                        >
                          {copiedIndex === index ? (
                            <>
                              <Icons.Check /> <span>Copied</span>
                            </>
                          ) : (
                            <>
                              <Icons.Copy /> <span>Copy</span>
                            </>
                          )}
                        </button>

                        <button
                          className={`action-copy-btn ${speakingIndex === index ? "action-active-speak" : ""}`}
                          onClick={() => handleSpeak(msg.text, index)}
                          title={speakingIndex === index ? "Stop speaking" : "Read response aloud"}
                        >
                          {speakingIndex === index ? (
                            <>
                              <Icons.VolumeX /> <span>Stop</span>
                            </>
                          ) : (
                            <>
                              <Icons.Volume /> <span>Read</span>
                            </>
                          )}
                        </button>

                        <button
                          className="action-copy-btn"
                          onClick={() => handleRegenerate(index)}
                          title="Regenerate this response"
                          disabled={loading}
                        >
                          <Icons.RotateCcw /> <span>Retry</span>
                        </button>

                        <div className="stream-feedback-group">
                          <button
                            className={`feedback-pill-btn ${feedbackState[index] === "up" ? "feedback-active-up" : ""}`}
                            onClick={() => handleFeedback(index, "up")}
                            title="Helpful response"
                          >
                            <Icons.ThumbsUp />
                          </button>
                          <button
                            className={`feedback-pill-btn ${feedbackState[index] === "down" ? "feedback-active-down" : ""}`}
                            onClick={() => handleFeedback(index, "down")}
                            title="Unhelpful response"
                          >
                            <Icons.ThumbsDown />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="stream-row bot loading">
                  <div className="stream-avatar">
                    <Icons.Sparkle />
                  </div>
                  <div className="stream-content">
                    <div className="loading-dots-pill">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef}></div>
            </div>
          )}
        </div>

        {/* 3. Composer Island */}
        <footer className="composer-container">
          <div className="composer-inner">
            {docFaqs && docFaqs.length > 0 && !selectedDoc && (
              <div className="doc-faq-island">
                <div className="doc-faq-header">
                  <span className="doc-faq-active-badge">
                    📄 Document FAQ Mode: <strong>{activeDocName || "Uploaded Document"}</strong>
                  </span>
                  <button 
                    className="doc-faq-close-btn"
                    onClick={() => { setDocFaqs([]); setActiveDocName(null); }}
                    title="Exit Document FAQ Mode"
                  >
                    ✕ Clear FAQs
                  </button>
                </div>
                <div className="doc-faq-chips-grid">
                  {docFaqs.map((faq, idx) => (
                    <button 
                      key={idx}
                      className="faq-chip-item"
                      onClick={() => sendMessage(faq.question)}
                      disabled={loading}
                    >
                      💡 {faq.question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {selectedDoc && (
              <div className="attachment-expanded-panel">
                <div className="attachment-chip-bar">
                  <div className="attachment-meta">
                    <div className="attachment-icon-pill">
                      <Icons.FileDoc />
                    </div>
                    <div className="attachment-info-col">
                      <span className="attachment-name">{selectedDoc.name}</span>
                      <span className="attachment-size">Ready for instant vector analysis ({selectedDoc.size})</span>
                    </div>
                  </div>
                  <button 
                    className="attachment-close-btn"
                    onClick={() => { setSelectedDoc(null); if (fileInputRef.current) fileInputRef.current.value = ""; }}
                    title="Remove attachment"
                  >
                    ✕
                  </button>
                </div>

                <div className="quick-doc-prompts">
                  <button 
                    className="doc-prompt-pill"
                    onClick={() => sendMessage("Analyze doctor's prescription and verify all medicines, dosages, and timings")}
                  >
                    🩺 Analyze & Verify Prescription
                  </button>
                  <button 
                    className="doc-prompt-pill"
                    onClick={() => sendMessage("What is the exact medicine schedule, dosages, and before/after food timings?")}
                  >
                    💊 Medicine Schedule & Timings
                  </button>
                  <button 
                    className="doc-prompt-pill"
                    onClick={() => sendMessage("Check for handwriting spelling corrections, antibiotic courses, and safety warnings")}
                  >
                    🔍 Spellings & Safety Warnings
                  </button>
                  <button 
                    className="doc-prompt-pill"
                    onClick={() => sendMessage("What are the key coverage benefits and limits?")}
                  >
                    🏥 Policy Coverage & Limits
                  </button>
                </div>
              </div>
            )}

            <div className="composer-input-row">
              <input
                id="doc-upload-input"
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp,.tiff,.docx,.doc,.xlsx,.xls,.csv,.txt,.md,.json,.log,.xml"
                ref={fileInputRef}
                style={{ display: "none" }}
                onChange={selectDoc}
              />
              <button 
                className="composer-action-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
                title="Attach Document or Prescription Image (PNG, JPG, PDF, Word, Excel)"
              >
                <Icons.Paperclip />
              </button>

              <input
                type="text"
                className={`composer-text-input ${isListening ? "listening" : ""}`}
                placeholder={
                  isListening ? "Listening to your voice..."
                    : selectedDoc ? `Ask any question from "${selectedDoc.name}"...`
                      : "Ask anything, search candidates, or attach a document..."
                }
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
              />

              <button 
                className={`composer-action-btn ${isListening ? "active-mic" : ""}`}
                onClick={toggleListening}
                title="Voice typing"
              >
                <Icons.Mic />
              </button>

              <button 
                className="composer-send-btn"
                onClick={() => sendMessage()}
                disabled={loading || (!message.trim() && !selectedDoc)}
                title="Send message"
              >
                <Icons.Send />
              </button>
            </div>
          </div>
        </footer>
      </main>

      {/* 4. AI Model & Runtime Studio Modal */}
      {isModelModalOpen && (
        <div className="modal-backdrop" onClick={() => setIsModelModalOpen(false)}>
          <div className="modal-dialog model-studio-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <div className="modal-icon-badge">
                  <Icons.Cpu />
                </div>
                <div>
                  <h2 className="modal-title">AI Model & Runtime Studio</h2>
                  <p className="modal-subtitle">Real-time inference engines, local runtimes & theme customizer</p>
                </div>
              </div>
              <button 
                className="modal-close-btn" 
                onClick={() => setIsModelModalOpen(false)}
                title="Close Studio (Esc)"
              >
                <Icons.Cross />
              </button>
            </div>

            <div className="modal-body">
              {/* Active Engine Card */}
              <div className="studio-card active-engine-card">
                <div className="studio-card-header">
                  <div className="studio-card-title">
                    <Icons.Zap />
                    <span>Active Inference Engine</span>
                  </div>
                  <div className="engine-status-pill">
                    <span className={`indicator-dot ${modelStatus?.is_llm_online ? "online" : "fallback"}`}></span>
                    <span>{modelStatus?.is_llm_online ? "Local LLM Online" : "100% Offline Neural Fallback"}</span>
                  </div>
                </div>

                <div className="active-engine-details">
                  <div className="detail-stat">
                    <span className="stat-label">Active Provider</span>
                    <span className="stat-val highlight">
                      {modelStatus?.is_llm_online
                        ? (modelStatus.active_provider === "ollama" ? "🦙 Ollama Local" : "🤖 LM Studio")
                        : "⚡ Local Neural TF-IDF RAG"}
                    </span>
                  </div>
                  <div className="detail-stat">
                    <span className="stat-label">Model Identifier</span>
                    <span className="stat-val code-pill">
                      {modelStatus?.active_model || "neural-tfidf-rag:v2"}
                    </span>
                  </div>
                  <div className="detail-stat">
                    <span className="stat-label">Latency / Response</span>
                    <span className="stat-val">
                      {pingLatency ? `${pingLatency} ms` : "Instant (< 50ms)"}
                    </span>
                  </div>
                  <div className="detail-stat stat-action">
                    <button 
                      className="ping-btn" 
                      onClick={handlePingModel}
                      disabled={isPinging}
                      title="Ping Backend & Providers"
                    >
                      <Icons.Refresh />
                      <span>{isPinging ? "Checking..." : "Ping Engine"}</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Providers Status Breakdown */}
              <div className="studio-card">
                <div className="studio-card-header">
                  <div className="studio-card-title">
                    <Icons.Server />
                    <span>Local Providers Ecosystem</span>
                  </div>
                  <span className="studio-card-tag">3 Supported Engines</span>
                </div>

                <div className="providers-grid">
                  {/* Ollama Provider */}
                  <div className={`provider-item ${modelStatus?.providers?.ollama?.running ? "active" : "dormant"}`}>
                    <div className="provider-top">
                      <div className="provider-info">
                        <span className="provider-icon">🦙</span>
                        <div>
                          <strong className="provider-name">Ollama</strong>
                          <span className="provider-port">Port 11434</span>
                        </div>
                      </div>
                      <span className={`provider-badge ${modelStatus?.providers?.ollama?.running ? "online" : "offline"}`}>
                        {modelStatus?.providers?.ollama?.running ? "Detected" : "Standby"}
                      </span>
                    </div>
                    <div className="provider-models">
                      <span className="models-label">Installed Models:</span>
                      {modelStatus?.providers?.ollama?.models?.length > 0 ? (
                        <div className="models-chips">
                          {modelStatus.providers.ollama.models.map((m, i) => (
                            <span key={i} className="model-mini-chip">{m}</span>
                          ))}
                        </div>
                      ) : (
                        <span className="models-empty">Start Ollama daemon to auto-detect</span>
                      )}
                    </div>
                  </div>

                  {/* LM Studio Provider */}
                  <div className={`provider-item ${modelStatus?.providers?.lm_studio?.running ? "active" : "dormant"}`}>
                    <div className="provider-top">
                      <div className="provider-info">
                        <span className="provider-icon">🤖</span>
                        <div>
                          <strong className="provider-name">LM Studio</strong>
                          <span className="provider-port">Port 1234</span>
                        </div>
                      </div>
                      <span className={`provider-badge ${modelStatus?.providers?.lm_studio?.running ? "online" : "offline"}`}>
                        {modelStatus?.providers?.lm_studio?.running ? "Detected" : "Standby"}
                      </span>
                    </div>
                    <div className="provider-models">
                      <span className="models-label">Loaded Models:</span>
                      {modelStatus?.providers?.lm_studio?.models?.length > 0 ? (
                        <div className="models-chips">
                          {modelStatus.providers.lm_studio.models.map((m, i) => (
                            <span key={i} className="model-mini-chip">{m}</span>
                          ))}
                        </div>
                      ) : (
                        <span className="models-empty">Start LM Studio server to connect</span>
                      )}
                    </div>
                  </div>

                  {/* Built-in Neural RAG */}
                  <div className="provider-item active featured">
                    <div className="provider-top">
                      <div className="provider-info">
                        <span className="provider-icon">🧠</span>
                        <div>
                          <strong className="provider-name">Neural TF-IDF RAG</strong>
                          <span className="provider-port">Built-in Python Engine</span>
                        </div>
                      </div>
                      <span className="provider-badge online">Always Ready</span>
                    </div>
                    <div className="provider-models">
                      <span className="models-label">Capabilities:</span>
                      <div className="models-chips">
                        <span className="model-mini-chip">Zero-Cloud Fallback</span>
                        <span className="model-mini-chip">Prescription OCR</span>
                        <span className="model-mini-chip">Vector Matching</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Theme Customizer Inside Modal */}
              <div className="studio-card">
                <div className="studio-card-header">
                  <div className="studio-card-title">
                    <Icons.Palette />
                    <span>Workspace Visual Theme</span>
                  </div>
                  <span className="studio-card-tag">7 High-Performance Themes</span>
                </div>

                <div className="theme-selection-grid">
                  {THEMES.map((th) => (
                    <div
                      key={th.id}
                      className={`theme-card ${currentTheme === th.id ? "selected" : ""}`}
                      onClick={() => handleSelectTheme(th.id)}
                    >
                      <div className="theme-card-top">
                        <span className="theme-icon">{th.icon}</span>
                        <span className="theme-dot" style={{ backgroundColor: th.color }}></span>
                      </div>
                      <div className="theme-name">{th.name}</div>
                      <div className="theme-tag">{th.tag}</div>
                      <div className="theme-desc">{th.desc}</div>
                      {currentTheme === th.id && (
                        <div className="theme-active-check">
                          <Icons.Check />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Helpful Tip Box */}
              <div className="studio-tip-box">
                <div className="tip-icon">💡</div>
                <div className="tip-content">
                  <strong>Zero-Config Local AI:</strong> To run Llama 3.2 locally, execute <code>ollama run llama3.2</code> in your terminal. Divya AI automatically discovers and connects without needing cloud API keys!
                </div>
              </div>
            </div>

            <div className="modal-footer">
              <span className="footer-status-text">
                User ID: <code>{userId || "Initializing..."}</code>
              </span>
              <button 
                className="modal-done-btn" 
                onClick={() => setIsModelModalOpen(false)}
              >
                Apply & Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;