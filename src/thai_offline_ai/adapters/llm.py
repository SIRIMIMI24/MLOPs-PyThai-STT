"""Production boundary for the verified loopback-only Ollama chat API."""
from __future__ import annotations

import socket
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from thai_offline_ai.config import LLMConfig
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import ChatMessage, LLMResult
from thai_offline_ai.privacy import is_local_url


class OllamaLocalLLM:
    """One local Ollama backend; HTTP client construction does not load a model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        offline_runtime: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.offline_runtime = offline_runtime
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._validate_endpoint()

    @classmethod
    def from_config(cls, config: LLMConfig, offline_runtime: bool) -> OllamaLocalLLM:
        if not config.enabled:
            raise WorkbenchError("Local LLM is disabled. Set llm.enabled=true after local Ollama setup.")
        if config.backend != "ollama":
            raise WorkbenchError("Only the Ollama local backend is implemented in Phase 4.")
        return cls(
            config.base_url,
            config.model,
            offline_runtime,
            config.temperature,
            config.max_tokens,
        )

    def _validate_endpoint(self) -> None:
        if not is_local_url(self.base_url):
            raise WorkbenchError("Local LLM endpoint must be an http(s) localhost or loopback URL.")

    def _requests(self):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise WorkbenchError('Requests is missing. Run: python -m pip install -e ".[llm]"') from exc
        return requests

    def status(self) -> tuple[bool, str]:
        """Socket-only loopback reachability check; it does not load a model."""
        self._validate_endpoint()
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True, "Local Ollama service is reachable."
        except OSError:
            return False, f"Local Ollama service is not reachable at {self.base_url}."

    def installed_models(self) -> list[str]:
        """Read Ollama's local model list; never pulls a model."""
        self._validate_endpoint()
        requests = self._requests()
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            if not isinstance(models, list):
                raise TypeError("models is not a list")
            return [str(item["name"]) for item in models if isinstance(item, dict) and item.get("name")]
        except requests.Timeout as exc:
            raise WorkbenchError("Local Ollama service timed out while listing models.") from exc
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise WorkbenchError("Cannot read the local Ollama model list.") from exc

    def chat(self, messages: Sequence[ChatMessage]) -> LLMResult:
        self._validate_endpoint()
        if not messages:
            raise WorkbenchError("Provide at least one chat message.")
        requests = self._requests()
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise WorkbenchError("Local Ollama request timed out. Try a smaller model or shorter prompt.") from exc
        except requests.ConnectionError as exc:
            raise WorkbenchError(
                "Local Ollama service is not running. Start Ollama, then run: ollama list"
            ) from exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise WorkbenchError(
                    f"Configured Ollama model '{self.model}' is not installed locally. "
                    f"Run explicitly: ollama pull {self.model}"
                ) from exc
            raise WorkbenchError("Local Ollama rejected the chat request.") from exc
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise WorkbenchError("Local Ollama returned an invalid chat response.") from exc
        message = data.get("message") if isinstance(data, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise WorkbenchError("Local Ollama response did not include assistant text.")
        prompt_tokens = _optional_int(data.get("prompt_eval_count"))
        completion_tokens = _optional_int(data.get("eval_count"))
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        total_duration = _optional_int(data.get("total_duration"))
        return LLMResult(
            text=text.strip(),
            backend="ollama",
            model=self.model,
            finish_reason=_optional_string(data.get("done_reason")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration=(total_duration / 1_000_000_000)
            if total_duration is not None
            else time.perf_counter() - started,
            metadata={"created_at": data.get("created_at")} if data.get("created_at") else {},
        )


def _optional_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def find_ollama_executable() -> str | None:
    """Find a Windows per-user Ollama install without changing PATH."""
    import os
    import shutil

    found = shutil.which("ollama")
    if found:
        return found
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
    try:
        return str(candidate) if candidate.is_file() else None
    except OSError:
        # A managed Windows install may deny metadata inspection while remaining
        # runnable for the signed-in user. Let the explicit setup command report
        # any actual execution failure.
        return str(candidate)
