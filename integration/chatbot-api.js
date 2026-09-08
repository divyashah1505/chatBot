/**
 * ChatBot API Client
 * Reusable helper library for integrating the ChatBot REST API into any external website.
 * 
 * Usage:
 *   import { ChatbotClient } from './chatbot-api.js';
 *   const bot = new ChatbotClient({ apiUrl: 'http://localhost:8000' });
 *   const response = await bot.sendMessage('Hello');
 *   console.log(response.reply);
 */

export class ChatbotClient {
  /**
   * @param {Object} options
   * @param {string} [options.apiUrl="http://localhost:8000"] - Base URL of your running backend.
   * @param {string} [options.userId] - Optional user identifier. Auto-generated if omitted.
   */
  constructor(options = {}) {
    this.apiUrl = (options.apiUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.userId = options.userId || ('user_' + Math.random().toString(36).substring(2, 9));
    this.sessionId = null;
  }

  /**
   * Check if backend API is running and reachable
   * @returns {Promise<boolean>}
   */
  async checkHealth() {
    try {
      const res = await fetch(`${this.apiUrl}/api/health`);
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Send a message to the chatbot
   * @param {string} message - The question or prompt to send.
   * @returns {Promise<{reply: string, session_id: string}>}
   */
  async sendMessage(message) {
    if (!message || typeof message !== 'string' || !message.trim()) {
      throw new Error('Message cannot be empty.');
    }

    const payload = {
      user_id: this.userId,
      message: message.trim(),
      session_id: this.sessionId,
    };

    const response = await fetch(`${this.apiUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Chatbot API Error (${response.status}): ${errorText}`);
    }

    const data = await response.json();
    // Update active session id to preserve multi-turn context
    if (data.session_id) {
      this.sessionId = data.session_id;
    }

    return data;
  }

  /**
   * Reset the current conversation session
   */
  resetSession() {
    this.sessionId = null;
  }

  /**
   * Fetch previous conversation history for the current user
   * @returns {Promise<Array>}
   */
  async getSessions() {
    const res = await fetch(`${this.apiUrl}/api/sessions/${this.userId}`);
    if (!res.ok) throw new Error('Failed to fetch sessions');
    return await res.json();
  }

  /**
   * Fetch messages for a specific session
   * @param {string} sessionId
   * @returns {Promise<Array>}
   */
  async getSessionMessages(sessionId) {
    const res = await fetch(`${this.apiUrl}/api/sessions/${this.userId}/${sessionId}`);
    if (!res.ok) throw new Error('Failed to fetch session messages');
    const data = await res.json();
    return data.messages || [];
  }
}

// Support CommonJS/Node environments as well if needed
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ChatbotClient };
}
