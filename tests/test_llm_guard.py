import pytest
import requests

from thai_offline_ai.adapters.llm import OllamaLocalLLM
from thai_offline_ai.config import LLMConfig
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import ChatMessage, LLMResult
from thai_offline_ai.privacy import is_local_url


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"])
def test_local_urls_allowed(url):
    assert is_local_url(url)


@pytest.mark.parametrize("url", ["https://api.example.com", "http://192.168.1.10:11434", "ftp://127.0.0.1:21"])
def test_non_local_urls_rejected(url):
    assert not is_local_url(url)


def test_remote_url_and_disabled_backend_are_rejected():
    with pytest.raises(WorkbenchError, match="loopback"):
        OllamaLocalLLM("https://api.example.com", "some-model")
    with pytest.raises(WorkbenchError, match="disabled"):
        OllamaLocalLLM.from_config(LLMConfig(), True)


def test_chat_message_and_result_contracts():
    assert ChatMessage("user", "สวัสดี").to_dict()["role"] == "user"
    with pytest.raises(ValueError, match="role"):
        ChatMessage("tool", "x")
    with pytest.raises(ValueError, match="non-empty"):
        ChatMessage("user", " ")
    assert LLMResult("ok", "ollama", "model").to_dict()["prompt_tokens"] is None


class Response:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError("bad status")
            error.response = self
            raise error


def test_successful_response_and_usage_metadata(monkeypatch):
    client = OllamaLocalLLM("http://127.0.0.1:11434", "local-model")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: Response(
            {"message": {"content": "คำตอบ"}, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2, "total_duration": 2_000_000_000}
        ),
    )
    result = client.chat([ChatMessage("user", "คำถาม")])
    assert (result.text, result.total_tokens, result.duration) == ("คำตอบ", 5, 2.0)


def test_service_timeout_model_missing_and_malformed_response(monkeypatch):
    client = OllamaLocalLLM("http://127.0.0.1:11434", "local-model")
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout()))
    with pytest.raises(WorkbenchError, match="timed out"):
        client.chat([ChatMessage("user", "x")])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response({}, 404))
    with pytest.raises(WorkbenchError, match="not installed"):
        client.chat([ChatMessage("user", "x")])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response({"message": {}}))
    with pytest.raises(WorkbenchError, match="assistant text"):
        client.chat([ChatMessage("user", "x")])


def test_installed_models_uses_tags_without_model_pull(monkeypatch):
    client = OllamaLocalLLM("http://127.0.0.1:11434", "local-model")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response({"models": [{"name": "local-model"}]}))
    assert client.installed_models() == ["local-model"]


def test_ollama_executable_uses_windows_per_user_location(monkeypatch, tmp_path):
    from thai_offline_ai.adapters.llm import find_ollama_executable

    executable = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert find_ollama_executable() == str(executable)
