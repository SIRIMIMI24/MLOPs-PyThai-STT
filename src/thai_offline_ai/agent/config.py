"""Configuration loader and validator for Phase 8 JARVIS Agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from thai_offline_ai.errors import ConfigurationError


@dataclass(slots=True)
class AgentCoreConfig:
    enabled: bool = True
    default_mode: str = "offline"
    max_steps: int = 6
    max_parse_retries: int = 2
    tool_timeout_seconds: float = 25.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024


@dataclass(slots=True)
class BrowserAgentConfig:
    headed: bool = True
    profile_dir: str = "outputs/jarvis_browser_profile"
    search_engine: str = "duckduckgo"
    navigation_timeout_ms: int = 20000
    action_timeout_ms: int = 10000
    max_page_chars: int = 4500
    max_tabs: int = 5
    viewport_width: int = 1280
    viewport_height: int = 820


@dataclass(slots=True)
class MemoryConfig:
    db_path: str = "outputs/jarvis_memory.db"
    max_history_messages: int = 16
    max_task_logs: int = 50


@dataclass(slots=True)
class SafetyConfig:
    require_approval_for_interactive: bool = True
    require_approval_for_sensitive: bool = True
    require_approval_for_memory_save: bool = True
    allow_local_test_urls: bool = False
    blocked_schemes: tuple[str, ...] = (
        "file",
        "javascript",
        "data",
        "chrome",
        "edge",
        "about",
        "ftp",
        "vbscript",
    )


@dataclass(slots=True)
class VoiceAgentConfig:
    push_to_talk: bool = True
    auto_speak_responses: bool = True
    wake_word_enabled: bool = False
    max_tts_chars: int = 600


@dataclass(slots=True)
class AgentSettings:
    agent: AgentCoreConfig = field(default_factory=AgentCoreConfig)
    browser: BrowserAgentConfig = field(default_factory=BrowserAgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    voice: VoiceAgentConfig = field(default_factory=VoiceAgentConfig)


_SECTION_TYPES = {
    "agent": AgentCoreConfig,
    "browser": BrowserAgentConfig,
    "memory": MemoryConfig,
    "safety": SafetyConfig,
    "voice": VoiceAgentConfig,
}


def validate_agent_settings(settings: AgentSettings) -> AgentSettings:
    """Validate Phase 8 settings and resolve local profile/database paths."""
    if settings.agent.default_mode not in {"offline", "online"}:
        raise ConfigurationError("agent.default_mode must be 'offline' or 'online'.")
    if not 1 <= settings.agent.max_steps <= 20:
        raise ConfigurationError("agent.max_steps must be between 1 and 20.")
    if not 0 <= settings.agent.max_parse_retries <= 5:
        raise ConfigurationError("agent.max_parse_retries must be between 0 and 5.")
    if not 1.0 <= settings.agent.tool_timeout_seconds <= 120.0:
        raise ConfigurationError("agent.tool_timeout_seconds must be between 1 and 120.")
    if settings.browser.search_engine not in {"duckduckgo"}:
        raise ConfigurationError("browser.search_engine must be 'duckduckgo'.")
    if not 500 <= settings.browser.max_page_chars <= 20000:
        raise ConfigurationError("browser.max_page_chars must be between 500 and 20000.")
    if not 1 <= settings.browser.max_tabs <= 15:
        raise ConfigurationError("browser.max_tabs must be between 1 and 15.")
    if not 2 <= settings.memory.max_history_messages <= 100:
        raise ConfigurationError("memory.max_history_messages must be between 2 and 100.")
    settings.browser.profile_dir = str(Path(settings.browser.profile_dir).expanduser().resolve())
    settings.memory.db_path = str(Path(settings.memory.db_path).expanduser().resolve())
    return settings


def load_agent_settings(path: str | Path | None = None) -> AgentSettings:
    """Load Phase 8 configuration from YAML with safe local defaults."""
    explicit = path is not None
    config_path = Path(path) if path is not None else Path("config/agent_config.yaml")
    data: dict[str, Any] = {}
    if explicit or config_path.exists():
        import yaml

        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Cannot read agent configuration {config_path}.") from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigurationError("Agent configuration must be a YAML mapping.")
            data = loaded
    unknown_sections = set(data.keys()) - set(_SECTION_TYPES.keys())
    if unknown_sections:
        raise ConfigurationError(
            f"Unknown agent config section(s): {', '.join(sorted(unknown_sections))}."
        )
    built: dict[str, Any] = {}
    for name, section_cls in _SECTION_TYPES.items():
        raw_section = data.get(name, {})
        if not isinstance(raw_section, dict):
            raise ConfigurationError(f"agent config section '{name}' must be a mapping.")
        allowed_keys = {item.name for item in fields(section_cls)}
        unknown_keys = set(raw_section.keys()) - allowed_keys
        if unknown_keys:
            raise ConfigurationError(
                f"Unknown setting(s) in {name}: {', '.join(sorted(unknown_keys))}."
            )
        kwargs = dict(raw_section)
        if name == "safety" and "blocked_schemes" in kwargs and isinstance(
            kwargs["blocked_schemes"], list
        ):
            kwargs["blocked_schemes"] = tuple(str(s).lower() for s in kwargs["blocked_schemes"])
        built[name] = section_cls(**kwargs)
    return validate_agent_settings(AgentSettings(**built))
