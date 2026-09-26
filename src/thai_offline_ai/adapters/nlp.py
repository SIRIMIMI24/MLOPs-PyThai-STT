from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any, TypedDict

from thai_offline_ai.config import SENTENCE_ENGINES, WORD_ENGINES, NLPConfig
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.privacy import configure_privacy


class NLPResult(TypedDict):
    normalized_text: str
    word_tokens: list[str]
    sentence_tokens: list[str]
    token_count: int
    thai_char_count: int
    latin_char_count: int
    metadata: dict[str, str]


# The installed CRF tokenizer reuses a module-level tagger; serialize its use.
_inference_lock = RLock()


@lru_cache(maxsize=1)
def _load_functions() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], str]:
    try:
        import pythainlp
        from pythainlp.tokenize import sent_tokenize, word_tokenize
        from pythainlp.util import normalize
    except (ImportError, OSError) as exc:
        raise WorkbenchError(
            "PyThaiNLP is missing or cannot load. Activate .venv and run: "
            "python -m pip install -e ."
        ) from exc
    return normalize, word_tokenize, sent_tokenize, pythainlp.__version__


@dataclass(slots=True)
class NLPAdapter:
    """Thai NLP with bundled resources, lazy imports, and no Gradio dependency."""

    word_engine: str = "newmm"
    sentence_engine: str = "crfcut"
    enabled: bool = True
    offline_runtime: bool = True

    @classmethod
    def from_config(cls, config: NLPConfig, offline_runtime: bool = True) -> NLPAdapter:
        if config.backend != "pythainlp":
            raise WorkbenchError("NLP backend must be pythainlp in Phase 1.")
        return cls(config.word_engine, config.sentence_engine, config.enabled, offline_runtime)

    def analyze(self, text: str) -> NLPResult:
        """Normalize and tokenize Thai text; never include input text in errors/logs."""
        if not self.enabled:
            raise WorkbenchError("NLP is disabled. Set nlp.enabled=true in your configuration.")
        if not isinstance(text, str):
            raise WorkbenchError("NLP input must be text.")
        if self.word_engine not in WORD_ENGINES or self.sentence_engine not in SENTENCE_ENGINES:
            raise WorkbenchError("Choose a supported local NLP engine in settings.example.yaml.")
        configure_privacy(self.offline_runtime)
        result: NLPResult = {
            "normalized_text": "",
            "word_tokens": [],
            "sentence_tokens": [],
            "token_count": 0,
            "thai_char_count": 0,
            "latin_char_count": 0,
            "metadata": {
                "backend": "pythainlp",
                "word_engine": self.word_engine,
                "sentence_engine": self.sentence_engine,
            },
        }
        if not text.strip():
            return result
        with _inference_lock:
            normalize, word_tokenize, sent_tokenize, version = _load_functions()
            try:
                normalized = normalize(text.strip())
                words = word_tokenize(normalized, engine=self.word_engine, keep_whitespace=False)
                sentences = (
                    sent_tokenize(normalized, engine=self.sentence_engine, keep_whitespace=False)
                    if normalized.strip()
                    else []
                )
            except ImportError as exc:
                raise WorkbenchError(
                    "The sentence tokenizer dependency is missing. Run: python -m pip install -e . "
                    "The crfcut engine requires python-crfsuite."
                ) from exc
            except Exception as exc:
                # Library exceptions can echo private input. Expose only the failure category.
                raise WorkbenchError(
                    f"Thai NLP failed ({type(exc).__name__}). Run python scripts/doctor.py "
                    "and check the configured tokenizer engines."
                ) from exc
        result.update(
            {
                "normalized_text": normalized,
                "word_tokens": words,
                "sentence_tokens": sentences,
                "token_count": len(words),
                "thai_char_count": len(re.findall(r"[\u0E00-\u0E7F]", normalized)),
                "latin_char_count": len(re.findall(r"[A-Za-z]", normalized)),
            }
        )
        result["metadata"]["version"] = version
        return result
