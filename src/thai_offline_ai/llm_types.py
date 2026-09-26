"""Stable local-chat contracts independent of a particular model server."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VALID_CHAT_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_CHAT_ROLES:
            raise ValueError("Chat message role must be system, user, or assistant.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Chat message content must be non-empty text.")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    backend: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
