import builtins
from unittest.mock import Mock

import pytest

from thai_offline_ai.adapters.nlp import NLPAdapter, _load_functions
from thai_offline_ai.errors import WorkbenchError


def test_empty_text_is_safe():
    result = NLPAdapter().analyze("   ")
    assert result["token_count"] == 0
    assert result["word_tokens"] == []


def test_thai_nlp_smoke_if_installed():
    pytest.importorskip("pythainlp")
    result = NLPAdapter().analyze("สวัสดีครับ วันนี้อากาศดี")
    assert result["token_count"] > 0
    assert result["thai_char_count"] > 0


def test_real_mixed_text_and_punctuation():
    result = NLPAdapter().analyze("  ประเทศไทยมีประชากรประมาณเจ็ดสิบล้านคน Hello!  ")
    assert result["normalized_text"].startswith("ประเทศไทย")
    assert result["latin_char_count"] == 5
    assert "ประชากร" in result["word_tokens"]
    assert result["token_count"] == len(result["word_tokens"])
    assert all(token.strip() for token in result["word_tokens"])
    assert result["sentence_tokens"]
    assert result["metadata"]["version"]


def test_real_normalization():
    assert NLPAdapter().analyze("เเปลก")["normalized_text"] == "แปลก"


@pytest.mark.parametrize("engine", ["newmm", "newmm-safe"])
def test_supported_word_engines(engine):
    assert NLPAdapter(word_engine=engine).analyze("ทดสอบภาษาไทย")["token_count"] > 0


@pytest.mark.parametrize("engine", ["crfcut", "whitespace", "whitespace+newline"])
def test_supported_sentence_engines(engine):
    assert NLPAdapter(sentence_engine=engine).analyze("สวัสดีครับ วันนี้อากาศดี")["sentence_tokens"]


def test_disabled_nlp_does_not_load(monkeypatch):
    loader = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr("thai_offline_ai.adapters.nlp._load_functions", loader)
    with pytest.raises(WorkbenchError, match="disabled"):
        NLPAdapter(enabled=False).analyze("test")
    loader.assert_not_called()


def test_missing_package_is_actionable(monkeypatch):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "pythainlp":
            raise ImportError("unavailable")
        return original(name, *args, **kwargs)

    _load_functions.cache_clear()
    monkeypatch.setattr(builtins, "__import__", unavailable)
    try:
        with pytest.raises(WorkbenchError, match="pip install"):
            NLPAdapter().analyze("test")
    finally:
        _load_functions.cache_clear()


def test_library_exception_does_not_echo_input(monkeypatch):
    def broken(text):
        raise ValueError(text)

    monkeypatch.setattr(
        "thai_offline_ai.adapters.nlp._load_functions", lambda: (broken, None, None, "test")
    )
    with pytest.raises(WorkbenchError) as error:
        NLPAdapter().analyze("PRIVATE_INPUT")
    assert "PRIVATE_INPUT" not in str(error.value)


@pytest.mark.parametrize("text", [None, 123, ["text"]])
def test_non_text_input(text):
    with pytest.raises(WorkbenchError, match="must be text"):
        NLPAdapter().analyze(text)


def test_repeated_requests_reuse_import_cache():
    _load_functions.cache_clear()
    adapter = NLPAdapter()
    first = adapter.analyze("ทดสอบ")
    second = adapter.analyze("ทดสอบ")
    assert first == second
    assert _load_functions.cache_info().misses == 1
    assert _load_functions.cache_info().hits == 1
