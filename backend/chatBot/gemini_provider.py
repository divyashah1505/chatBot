"""
Gemini AI Provider for Company Owner & Personal AI Assistant.
Direct REST API Client for Google Gemini (supports gemini-3.6-flash, gemini-2.5-flash, gemini-2.0-flash).
Bypasses gRPC/protobuf compatibility issues and guarantees fast, reliable streaming/completions.
"""

import os
import json
import logging
import requests
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("GeminiProvider")

PRIMARY_MODELS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]

BASE_REST_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._active_model = None

    def is_available(self) -> bool:
        """Returns True if an API key is configured."""
        return bool(self.api_key and len(self.api_key) > 10)

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1500
    ) -> Optional[str]:
        """
        Sends multi-turn chat history to Gemini REST API.
        Converts OpenAI-style messages list to Gemini API format.
        """
        if not self.is_available():
            return None

        # Build Gemini contents payload
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Map roles: 'assistant' -> 'model', 'user' -> 'user'
            gemini_role = "model" if role in ["assistant", "bot"] else "user"
            text_content = msg.get("content", "")
            if text_content.strip():
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": text_content}]
                })

        # If no contents, fallback
        if not contents:
            return None

        # Prepare system instruction if provided
        system_payload = None
        if system_instruction:
            system_payload = {
                "parts": [{"text": system_instruction}]
            }

        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95
        }

        request_body = {
            "contents": contents,
            "generationConfig": generation_config
        }
        if system_payload:
            request_body["systemInstruction"] = system_payload

        # Try available models in priority order
        models_to_try = [self._active_model] if self._active_model else []
        models_to_try.extend([m for m in PRIMARY_MODELS if m not in models_to_try])

        headers = {
            "Content-Type": "application/json"
        }

        for model in models_to_try:
            url = f"{BASE_REST_URL}/{model}:generateContent?key={self.api_key}"
            try:
                resp = requests.post(url, headers=headers, json=request_body, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_chunks = [p.get("text", "") for p in parts if "text" in p]
                        reply_text = "".join(text_chunks).strip()
                        if reply_text:
                            self._active_model = model
                            return reply_text
                elif resp.status_code == 404:
                    # Model not available in this tier, try next
                    continue
                else:
                    logger.warning(f"Gemini API returned status {resp.status_code} for {model}: {resp.text[:200]}")
            except Exception as ex:
                logger.error(f"Error calling Gemini model {model}: {ex}")
                continue

        return None


gemini_provider = GeminiProvider()
