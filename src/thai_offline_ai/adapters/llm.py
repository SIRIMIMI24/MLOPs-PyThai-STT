from __future__ import annotations

import socket
from urllib.parse import urlparse

from thai_offline_ai.privacy import is_local_url


class OllamaLocalLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        offline_runtime: bool = True,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.offline_runtime = offline_runtime
        self.temperature = temperature
        if offline_runtime and not is_local_url(self.base_url):
            raise ValueError("Offline runtime allows only localhost/loopback LLM endpoints.")

    def chat(self, message: str, system_prompt: str | None = None) -> str:
        if not message.strip():
            raise ValueError("Message is empty.")
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install local LLM extra: pip install -e '.[llm]'") from exc

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message.strip()})

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()

    def status(self) -> tuple[bool, str]:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True, "Local LLM server is reachable."
        except OSError:
            return False, f"Local LLM server is not reachable at {self.base_url}."
