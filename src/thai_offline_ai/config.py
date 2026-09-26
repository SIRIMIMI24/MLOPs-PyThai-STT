from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path, PureWindowsPath
from typing import Any

from thai_offline_ai.errors import ConfigurationError
from thai_offline_ai.privacy import is_local_url, is_loopback_host


@dataclass(slots=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    output_dir: str = "outputs"
    offline_runtime: bool = True
    log_level: str = "WARNING"


@dataclass(slots=True)
class NLPConfig:
    word_engine: str = "newmm"
    enabled: bool = True
    backend: str = "pythainlp"
    sentence_engine: str = "crfcut"


@dataclass(slots=True)
class TTSConfig:
    enabled: bool = False
    backend: str = "pythaitts"
    model: str = "lunarlist_onnx"
    sample_rate: int = 22050
    model_dir: str = "models/pythaitts/lunarlist_onnx"


@dataclass(slots=True)
class ASRConfig:
    enabled: bool = False
    backend: str = "pythaiasr"
    model: str | None = "typhoon_asr"
    device: str = "cpu"
    sampling_rate: int = 16000
    stream_chunk_seconds: float = 1.0
    model_dir: str = "models/pythaiasr"


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    model_path: str | None = None
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass(slots=True)
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


SECTIONS = {
    "app": AppConfig,
    "nlp": NLPConfig,
    "tts": TTSConfig,
    "asr": ASRConfig,
    "llm": LLMConfig,
}
WORD_ENGINES = {"newmm", "newmm-safe"}
SENTENCE_ENGINES = {"crfcut", "whitespace", "whitespace+newline"}
# Restrict NLP to engines using installed resources, with no model downloads.
ENV_FIELDS = {
    "THAI_AI_HOST": ("app", "host"),
    "THAI_AI_PORT": ("app", "port"),
    "THAI_AI_OUTPUT_DIR": ("app", "output_dir"),
    "THAI_AI_OFFLINE_RUNTIME": ("app", "offline_runtime"),
    "THAI_AI_LOG_LEVEL": ("app", "log_level"),
    "THAI_AI_NLP_ENABLED": ("nlp", "enabled"),
    "THAI_AI_NLP_BACKEND": ("nlp", "backend"),
    "THAI_AI_NLP_WORD_ENGINE": ("nlp", "word_engine"),
    "THAI_AI_NLP_SENTENCE_ENGINE": ("nlp", "sentence_engine"),
    "THAI_AI_TTS_ENABLED": ("tts", "enabled"),
    "THAI_AI_TTS_BACKEND": ("tts", "backend"),
    "THAI_AI_TTS_MODEL": ("tts", "model"),
    "THAI_AI_TTS_MODEL_DIR": ("tts", "model_dir"),
    "THAI_AI_ASR_ENABLED": ("asr", "enabled"),
    "THAI_AI_ASR_BACKEND": ("asr", "backend"),
    "THAI_AI_ASR_MODEL_DIR": ("asr", "model_dir"),
    "THAI_AI_LLM_ENABLED": ("llm", "enabled"),
    "THAI_AI_LLM_BACKEND": ("llm", "backend"),
    "THAI_AI_LLM_BASE_URL": ("llm", "base_url"),
}


def _validate_types(section_name: str, section: Any) -> None:
    defaults = SECTIONS[section_name]()
    for item in fields(defaults):
        value, default = getattr(section, item.name), getattr(defaults, item.name)
        if default is None:
            valid = value is None or isinstance(value, str)
        elif isinstance(default, float):
            valid = type(value) in {int, float} and math.isfinite(value)
        else:
            valid = type(value) is type(default)
        if isinstance(value, str) and (not value.strip() or "\x00" in value):
            valid = False
        if not valid:
            expected = "a string or null" if default is None else type(default).__name__
            raise ConfigurationError(f"{section_name}.{item.name} must be {expected}.")


def validate_settings(settings: Settings) -> Settings:
    """Validate configuration from YAML, environment, or programmatic callers."""
    for name, section_type in SECTIONS.items():
        section = getattr(settings, name)
        if not isinstance(section, section_type):
            raise ConfigurationError(f"{name} must be a {section_type.__name__} configuration.")
        _validate_types(name, section)
    if not is_loopback_host(settings.app.host):
        raise ConfigurationError("app.host must be 127.0.0.1, localhost, or a loopback address.")
    if not 1 <= settings.app.port <= 65535:
        raise ConfigurationError("app.port must be an integer from 1 to 65535.")
    if settings.app.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("app.log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    for name, choices in {
        "nlp": {"pythainlp"},
        "tts": {"pythaitts"},
        "asr": {"pythaiasr"},
        "llm": {"ollama", "llama_cpp"},
    }.items():
        if getattr(settings, name).backend not in choices:
            raise ConfigurationError(
                f"{name}.backend must be one of: {', '.join(sorted(choices))}."
            )
    if settings.nlp.word_engine not in WORD_ENGINES:
        raise ConfigurationError("nlp.word_engine must be newmm or newmm-safe in Phase 1.")
    if settings.nlp.sentence_engine not in SENTENCE_ENGINES:
        raise ConfigurationError(
            "nlp.sentence_engine must be crfcut, whitespace, or whitespace+newline."
        )
    if not is_local_url(settings.llm.base_url):
        raise ConfigurationError(
            "llm.base_url must be a localhost/loopback HTTP(S) URL with a valid port, "
            "without credentials, query, or fragment. Cloud endpoints are unsupported."
        )
    if settings.asr.device not in {"cpu", "cuda", "auto"}:
        raise ConfigurationError("asr.device must be cpu, cuda, or auto; Phase 1 uses no ASR.")
    if settings.asr.sampling_rate <= 0:
        raise ConfigurationError("asr.sampling_rate must be positive.")
    if not 0.25 <= settings.asr.stream_chunk_seconds <= 3.0:
        raise ConfigurationError("asr.stream_chunk_seconds must be from 0.25 to 3.0 seconds.")
    if settings.tts.model != "lunarlist_onnx":
        raise ConfigurationError(
            "tts.model must be lunarlist_onnx. It is the verified CPU/ONNX backend for Phase 2."
        )
    if settings.tts.sample_rate != 22050:
        raise ConfigurationError(
            "tts.sample_rate must be 22050 for the verified lunarlist_onnx backend."
        )
    if not 0 <= settings.llm.temperature <= 2 or settings.llm.max_tokens <= 0:
        raise ConfigurationError("llm.temperature must be 0–2 and llm.max_tokens positive.")
    raw = settings.app.output_dir
    if raw.startswith(("\\\\", "//")) or PureWindowsPath(raw).drive.startswith("\\"):
        raise ConfigurationError("app.output_dir must be a local directory, not a network share.")
    try:
        output = Path(raw).expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(
            "app.output_dir is invalid; choose a local writable folder."
        ) from exc
    if output == Path(output.anchor) or str(output).startswith(("\\\\", "//")):
        raise ConfigurationError("app.output_dir must be a local folder, not a drive root/share.")
    settings.app.output_dir = str(output)
    raw_model_dir = settings.tts.model_dir
    if raw_model_dir.startswith(("\\\\", "//")) or PureWindowsPath(raw_model_dir).drive.startswith("\\"):
        raise ConfigurationError("tts.model_dir must be a local directory, not a network share.")
    try:
        model_dir = Path(raw_model_dir).expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigurationError("tts.model_dir is invalid; choose a local models folder.") from exc
    if model_dir == Path(model_dir.anchor) or str(model_dir).startswith(("\\\\", "//")):
        raise ConfigurationError("tts.model_dir must be a local folder, not a drive root/share.")
    settings.tts.model_dir = str(model_dir)
    asr_dir = Path(settings.asr.model_dir).expanduser().resolve()
    if asr_dir == Path(asr_dir.anchor) or str(asr_dir).startswith(("\\\\", "//")):
        raise ConfigurationError("asr.model_dir must be a local folder, not a drive root/share.")
    settings.asr.model_dir = str(asr_dir)
    return settings


def load_settings(path: str | Path | None = None) -> Settings:
    """Load defaults < YAML < THAI_AI_* environment; relative paths use cwd.

    The default YAML is optional. Explicit --config/THAI_AI_CONFIG paths must exist.
    .env files are not executed or loaded implicitly.
    """
    unknown_env = {key for key in os.environ if key.startswith("THAI_AI_")} - (
        ENV_FIELDS.keys() | {"THAI_AI_CONFIG"}
    )
    if unknown_env:
        raise ConfigurationError(
            f"Unknown environment setting: {', '.join(sorted(unknown_env))}. "
            "See README.md for supported overrides."
        )
    explicit = path is not None or "THAI_AI_CONFIG" in os.environ
    raw_path = (
        path if path is not None else os.environ.get("THAI_AI_CONFIG", "config/settings.yaml")
    )
    if not str(raw_path).strip():
        raise ConfigurationError("THAI_AI_CONFIG/--config must name an existing YAML file.")
    config_path = Path(raw_path)
    data: dict[str, Any] = {}
    if explicit or config_path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationError("PyYAML is missing. Run: python -m pip install -e .") from exc

        class UniqueKeyLoader(yaml.SafeLoader):
            def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
                self.flatten_mapping(node)
                seen = set()
                for key_node, _ in node.value:
                    key = self.construct_object(key_node, deep=deep)
                    if not isinstance(key, str):
                        raise ConfigurationError("YAML setting names must be strings.")
                    if key in seen:
                        raise ConfigurationError("Duplicate YAML setting; keep each setting once.")
                    seen.add(key)
                return super().construct_mapping(node, deep=deep)

        try:
            loaded = yaml.load(config_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ConfigurationError(
                f"Cannot read configuration {config_path}. Check the path, permissions, and YAML syntax."
            ) from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigurationError("Configuration must be a YAML mapping of named sections.")
            data = loaded
    if data.keys() - SECTIONS.keys():
        raise ConfigurationError("Unknown configuration section. Use config/settings.example.yaml.")
    sections: dict[str, Any] = {}
    for name, section_type in SECTIONS.items():
        values = data.get(name, {})
        if not isinstance(values, dict):
            raise ConfigurationError(f"{name} must be a YAML mapping.")
        if values.keys() - {item.name for item in fields(section_type)}:
            raise ConfigurationError(f"Unknown setting in {name}; check settings.example.yaml.")
        sections[name] = section_type(**values)
        _validate_types(name, sections[name])
    for variable, (section, key) in ENV_FIELDS.items():
        if variable not in os.environ:
            continue
        raw = os.environ[variable].strip()
        default = getattr(SECTIONS[section](), key)
        value: str | int | bool = raw
        if isinstance(default, bool):
            if raw.lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                raise ConfigurationError(f"{variable} must be true or false.")
            value = raw.lower() in {"true", "1", "yes", "on"}
        elif isinstance(default, int):
            try:
                value = int(raw)
            except ValueError as exc:
                raise ConfigurationError(f"{variable} must be an integer.") from exc
        setattr(sections[section], key, value)
    return validate_settings(Settings(**sections))
