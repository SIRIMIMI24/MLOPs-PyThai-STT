"""Stable structured ASR data contract for current and future local backends."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    segments: list[SpeechSegment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    backend: str = "pythaiasr"
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
