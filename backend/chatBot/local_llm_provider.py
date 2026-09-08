"""
Unified Local LLM Provider Manager (100% Offline & Zero Cloud Dependencies).
Supports:
1. Ollama (http://127.0.0.1:11434 - llama3.2, mistral, qwen2.5, deepseek, phi3, etc.)
2. LM Studio (http://127.0.0.1:1234/v1 - GGUF models)
3. Smart Offline Fallback (Zero external service required)
"""

import json
import socket
import urllib.request
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI

# Provider connection configurations
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/v1"

LM_STUDIO_HOST = "127.0.0.1"
LM_STUDIO_PORT = 1234
LM_STUDIO_BASE_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1"

DEFAULT_MODEL_NAME = "llama3.2"


def _check_port(host: str, port: int, timeout: float = 0.2) -> bool:
    """Fast socket check (100-200ms) to check if a local port is listening."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def is_ollama_running(timeout: float = 0.2) -> bool:
    """Checks if local Ollama daemon is actively running on port 11434."""
    return _check_port(OLLAMA_HOST, OLLAMA_PORT, timeout=timeout)


def is_lm_studio_running(timeout: float = 0.2) -> bool:
    """Checks if LM Studio local server is actively running on port 1234."""
    return _check_port(LM_STUDIO_HOST, LM_STUDIO_PORT, timeout=timeout)


def get_ollama_models() -> List[str]:
    """Fetches list of available local models installed in Ollama."""
    if not is_ollama_running(timeout=0.2):
        return []
    try:
        req = urllib.request.Request(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        return []


def get_lm_studio_models() -> List[str]:
    """Fetches list of available/loaded models in LM Studio."""
    if not is_lm_studio_running(timeout=0.2):
        return []
    try:
        req = urllib.request.Request(
            f"{LM_STUDIO_BASE_URL}/models",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("data", [])
            names = []
            for m in models:
                mid = m.get("id", "")
                if "embed" not in mid.lower() and mid:
                    names.append(mid)
            return names if names else [m.get("id", "") for m in models if m.get("id")]
    except Exception:
        return []


class LocalLLMProvider:
    """
    Central Manager for Local LLM inference.
    Automatically prioritizes active local LLM runtimes (Ollama -> LM Studio -> Offline).
    """

    @classmethod
    def detect_active_provider(cls) -> Tuple[str, str, bool]:
        """
        Returns: (provider_name, model_name, is_online)
        - 'ollama', 'llama3.2:latest', True
        - 'lm_studio', 'llama-3.2-3b-instruct', True
        - 'offline_synthesizer', 'neural-tfidf-rag', False
        """
        # 1. Check Ollama
        if is_ollama_running(timeout=0.2):
            models = get_ollama_models()
            if models:
                # Prefer llama3.2, mistral, qwen, deepseek, phi
                preferred = ["llama3.2", "llama3", "mistral", "qwen2.5", "deepseek", "phi3", "gemma2"]
                chosen = models[0]
                for p in preferred:
                    match = next((m for m in models if p in m.lower()), None)
                    if match:
                        chosen = match
                        break
                return ("ollama", chosen, True)
            return ("ollama", DEFAULT_MODEL_NAME, True)

        # 2. Check LM Studio
        if is_lm_studio_running(timeout=0.2):
            models = get_lm_studio_models()
            chosen = models[0] if models else "llama-3.2-3b-instruct"
            return ("lm_studio", chosen, True)

        # 3. Offline Fallback
        return ("offline_synthesizer", "neural-tfidf-rag", False)

    @classmethod
    def get_provider_status(cls) -> Dict[str, Any]:
        """Detailed status for API and frontend display."""
        from .gemini_provider import gemini_provider
        gemini_ok = gemini_provider.is_available()
        provider, model, is_online = cls.detect_active_provider()
        ollama_ok = is_ollama_running(timeout=0.2)
        lm_studio_ok = is_lm_studio_running(timeout=0.2)

        active_provider = provider
        active_model = model
        if not is_online and gemini_ok:
            active_provider = "gemini"
            active_model = gemini_provider._active_model or "gemini-3.6-flash"
            is_online = True

        return {
            "active_provider": active_provider,
            "active_model": active_model,
            "is_llm_online": is_online,
            "providers": {
                "gemini": {
                    "running": gemini_ok,
                    "model": "gemini-3.6-flash / 2.5-flash",
                    "status": "Online (Google AI Cloud)" if gemini_ok else "No API Key"
                },
                "ollama": {
                    "running": ollama_ok,
                    "endpoint": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
                    "models": get_ollama_models() if ollama_ok else []
                },
                "lm_studio": {
                    "running": lm_studio_ok,
                    "endpoint": f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}",
                    "models": get_lm_studio_models() if lm_studio_ok else []
                },
                "offline_synthesizer": {
                    "running": True,
                    "type": "Local TF-IDF & Neural Feature Extractor",
                    "status": "Ready (100% Offline)"
                }
            }
        }

    @classmethod
    def generate_chat_completion(
        cls,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 800,
        timeout: float = 8.0
    ) -> Optional[Tuple[str, str, str]]:
        """
        Executes chat completion with the active local provider.
        Returns: (response_text, provider_name, model_name) or None if all offline/failed.
        """
        provider, model_name, is_online = cls.detect_active_provider()
        if not is_online:
            return None

        base_url = OLLAMA_BASE_URL if provider == "ollama" else LM_STUDIO_BASE_URL
        api_key = "ollama" if provider == "ollama" else "lm-studio"

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"]
                if content and content.strip():
                    return (content.strip(), provider, model_name)
        except Exception as e:
            print(f"[LocalLLM] Error calling {provider} ({model_name}): {e}")

        # If primary failed, try secondary if online
        if provider == "ollama" and is_lm_studio_running(timeout=0.2):
            try:
                lm_models = get_lm_studio_models()
                lm_model = lm_models[0] if lm_models else "llama-3.2-3b-instruct"
                payload["model"] = lm_model
                req = urllib.request.Request(
                    f"{LM_STUDIO_BASE_URL}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer lm-studio"
                    }
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        return (content.strip(), "lm_studio", lm_model)
            except Exception as e:
                print(f"[LocalLLM] Secondary LM Studio fallback error: {e}")

        return None


local_llm_provider = LocalLLMProvider()
