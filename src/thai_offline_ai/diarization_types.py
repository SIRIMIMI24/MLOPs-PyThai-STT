"""Structured local speaker-diarization and ASR-composition contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Speaker turn must have non-negative start and end after start.")
        if not self.speaker.strip():
            raise ValueError("Speaker turn must have a speaker label.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    turns: list[SpeakerTurn] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    duration: float | None = None
    backend: str = "pythaiasr"
    model: str = "nemotron-3-diarization"
    device: str = "cpu"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeakerAttributedSegment:
    text: str
    start: float | None
    end: float | None
    speaker: str | None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeakerAttributedTranscript:
    text: str
    segments: list[SpeakerAttributedSegment] = field(default_factory=list)
    asr_backend: str = "pythaiasr"
    asr_model: str | None = None
    diarization_backend: str = "pythaiasr"
    diarization_model: str = "nemotron-3-diarization"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
