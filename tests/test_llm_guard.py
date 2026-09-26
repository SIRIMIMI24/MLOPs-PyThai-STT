import pytest

from thai_offline_ai.adapters.llm import OllamaLocalLLM, is_local_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
    ],
)
def test_local_urls_allowed(url):
    assert is_local_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://192.168.1.10:11434",
        "ftp://127.0.0.1:21",
    ],
)
def test_non_local_urls_rejected(url):
    assert not is_local_url(url)


def test_offline_llm_rejects_remote_url():
    with pytest.raises(ValueError):
        OllamaLocalLLM("https://api.example.com", "some-model", offline_runtime=True)
