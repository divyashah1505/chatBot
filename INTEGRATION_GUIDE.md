# Integrating ChatBot API into Other Websites (Option 2: Direct REST API)

This guide shows you how to connect any external website (HTML, React, Next.js, Vue, Angular, Node.js) to this Chatbot backend while running it locally on your computer without external hosting.

---

## 1. Start the ChatBot Backend Locally

Before your other website can communicate with the chatbot, make sure the backend is running on your machine:

1. Open a terminal in the `backend` folder:
   ```bash
   cd backend
   ```
2. Activate your virtual environment (if you use one):
   ```bash
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   ```
3. Start the FastAPI server:
   ```bash
   python main.py
   # Or using uvicorn:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

You will see:
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The API will be accessible at:
- **Base URL:** `http://localhost:8000`
- **Health Check:** `http://localhost:8000/api/health`
- **Chat Endpoint:** `http://localhost:8000/api/chat`
- **Swagger Docs:** `http://localhost:8000/docs`

---

## 2. API Endpoint Specification

### `POST /api/chat`

- **URL:** `http://localhost:8000/api/chat`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Body (JSON):**
  ```json
  {
    "user_id": "unique_user_id",
    "message": "Find candidates with 3 years experience in Python",
    "session_id": null
  }
  ```
  *(Note: Pass `null` or omit `session_id` on the first message. When the backend replies, it will return a `session_id`. Pass that `session_id` in subsequent requests to maintain conversational memory across turns).*

- **Response Body (JSON):**
  ```json
  {
    "reply": "Here are the top candidates matching your query...",
    "session_id": "66d9f481c4e970c679a834b1"
  }
  ```

---

## 3. How to Integrate into Another Website

### A. Vanilla JavaScript (HTML / Any Website)

Using browser native `fetch()`:

```javascript
let currentSessionId = null;
const USER_ID = "web_visitor_" + Math.floor(Math.random() * 10000);

async function sendMessageToBot(messageText) {
  try {
    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        user_id: USER_ID,
        message: messageText,
        session_id: currentSessionId
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const data = await response.json();
    // Save session_id for next message in conversation
    currentSessionId = data.session_id;

    // Display bot's reply in your UI
    console.log("Bot reply:", data.reply);
    return data.reply;
  } catch (err) {
    console.error("Failed to reach ChatBot API:", err);
    return "Sorry, could not connect to chatbot server. Make sure it is running on http://localhost:8000.";
  }
}
```

---

### B. React / Next.js Component

```jsx
import React, { useState } from 'react';

export default function ChatWidget() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setMessages(prev => [...prev, { sender: 'user', text: userText }]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'react_client_user',
          message: userText,
          session_id: sessionId
        })
      });

      const data = await response.json();
      setSessionId(data.session_id);
      setMessages(prev => [...prev, { sender: 'bot', text: data.reply }]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: 'Error connecting to Chatbot API at http://localhost:8000.' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 500, margin: '20px auto', fontFamily: 'sans-serif' }}>
      <div style={{ height: 350, overflowY: 'auto', border: '1px solid #ccc', padding: 12, borderRadius: 8 }}>
        {messages.map((m, idx) => (
          <div key={idx} style={{ textAlign: m.sender === 'user' ? 'right' : 'left', margin: '8px 0' }}>
            <span style={{
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: 6,
              background: m.sender === 'user' ? '#007bff' : '#eee',
              color: m.sender === 'user' ? '#fff' : '#000',
              whiteSpace: 'pre-wrap'
            }}>
              {m.text}
            </span>
          </div>
        ))}
        {loading && <p style={{ color: '#888' }}>Typing...</p>}
      </div>
      <form onSubmit={handleSend} style={{ display: 'flex', marginTop: 10, gap: 8 }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          style={{ flex: 1, padding: 8 }}
        />
        <button type="submit" disabled={loading} style={{ padding: '8px 16px' }}>Send</button>
      </form>
    </div>
  );
}
```

---

### C. Using the Pre-built Client Library (`integration/chatbot-api.js`)

You can copy [`integration/chatbot-api.js`](file:///c:/Users/DIVYA%20B%20SHAH/Desktop/ChatBot/integration/chatbot-api.js) into your other project:

```javascript
import { ChatbotClient } from './chatbot-api.js';

const bot = new ChatbotClient({ apiUrl: 'http://localhost:8000' });

// 1. Send message
const response = await bot.sendMessage('Find candidates for BDE');
console.log(response.reply);

// 2. Multi-turn follow-up (session is maintained automatically!)
const followUp = await bot.sendMessage('What are their contact details?');
console.log(followUp.reply);
```

---

## 4. Testing It Instantly

You can test this right now without writing any code in your other project:
1. Make sure the backend is running (`python backend/main.py`).
2. Double-click or open [`integration/test_api.html`](file:///c:/Users/DIVYA%20B%20SHAH/Desktop/ChatBot/integration/test_api.html) directly in any browser.
3. Type a message and verify the responses in real time!
