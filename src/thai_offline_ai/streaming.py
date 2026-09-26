"""Session-scoped, cache-only streaming support for the local Typhoon ASR model."""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from thai_offline_ai.adapters.asr import model_is_available
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError


@dataclass(frozen=True, slots=True)
class StreamingASREvent:
    """One incremental local-ASR update; timestamps are omitted when unavailable."""

    session_id: str
    sequence: int
    text: str
    is_final: bool
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StreamingSessionMetrics:
    """Local service timing; monotonic values are meaningful only in this process."""

    session_id: str
    state: str
    session_duration_seconds: float
    audio_seconds_processed: float
    inference_seconds: float
    session_rtf: float | None
    finalized_events: int
    latest_chunk_latency_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _Session:
    decoder: Any
    sequence: int = 0
    events: deque[StreamingASREvent] = field(default_factory=lambda: deque(maxlen=256))
    closed: bool = False
    started_at: float = field(default_factory=time.perf_counter)
    audio_seconds_processed: float = 0.0
    inference_seconds: float = 0.0
    latest_chunk_latency_seconds: float | None = None
    lock: Lock = field(default_factory=Lock)


class StreamingASRService:
    """Independent streaming core backed by PyThaiASR's RealtimeStreamASR.

    PyThaiASR 2.2.0 emits newly-decoded text fragments, rather than revisable
    hypotheses. Those fragments are therefore final incremental transcript
    segments. ``finalize`` emits the final remaining fragment and closes the
    session. The model's own rolling audio context is bounded; this service does
    not retain an unbounded application-level audio buffer.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._model_lock = Lock()
        self._inference_lock = Lock()
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = Lock()

    def _load_model(self) -> Any:
        config = self.settings.asr
        if not config.enabled:
            raise WorkbenchError("ASR is disabled. Set asr.enabled=true after local model setup.")
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            if not model_is_available(config.model_dir):
                raise WorkbenchError(
                    "ASR model is not available locally. Run: python scripts/setup_asr_model.py"
                )
            try:
                os.environ["PYTHAIASR_DATA_DIR"] = config.model_dir
                from pythaiasr import ASR

                self._model = ASR(model=config.model or "typhoon_asr", device="cpu").model
            except (ImportError, OSError) as exc:
                raise WorkbenchError(
                    'PyThaiASR is missing. Run: python -m pip install -e ".[asr,dev]"'
                ) from exc
            except Exception as exc:
                raise WorkbenchError(
                    "Thai ASR could not initialize. Confirm the local Typhoon model cache."
                ) from exc
        return self._model

    def start_session(self) -> str:
        """Create isolated decoder state without loading a model until it is needed."""
        session_id = uuid4().hex
        with self._sessions_lock:
            self._sessions[session_id] = _Session(decoder=None)
        return session_id

    def _get_session(self, session_id: str) -> _Session:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise WorkbenchError("Streaming ASR session was not found or has already closed.")
        return session

    def _decoder_for(self, session: _Session) -> Any:
        if session.decoder is None:
            try:
                from pythaiasr import RealtimeStreamASR

                session.decoder = RealtimeStreamASR(
                    model=self._load_model(),
                    sample_rate=self.settings.asr.sampling_rate,
                    step_sec=self.settings.asr.stream_chunk_seconds,
                )
            except WorkbenchError:
                raise
            except Exception as exc:
                raise WorkbenchError("Thai streaming ASR could not start for this session.") from exc
        return session.decoder

    def _normalise_chunk(self, samples: Any, sample_rate: int) -> Any:
        try:
            import numpy as np

            if sample_rate <= 0:
                raise ValueError("invalid sample rate")
            data = np.asarray(samples, dtype=np.float32)
            if data.size == 0 or data.ndim not in {1, 2}:
                raise ValueError("invalid audio")
            mono = data.mean(axis=1) if data.ndim == 2 else data
            if not np.isfinite(mono).all():
                raise ValueError("non-finite audio")
            target_rate = self.settings.asr.sampling_rate
            if sample_rate == target_rate:
                return mono
            target_count = max(1, round(len(mono) * target_rate / sample_rate))
            source_axis = np.linspace(0, len(mono) - 1, len(mono), dtype=np.float64)
            target_axis = np.linspace(0, len(mono) - 1, target_count, dtype=np.float64)
            return np.interp(target_axis, source_axis, mono).astype(np.float32)
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("Streaming audio must be a non-empty readable PCM audio chunk.") from exc

    def _record(
        self,
        session_id: str,
        session: _Session,
        text: str,
        *,
        is_final: bool,
        timing: dict[str, float] | None = None,
    ) -> list[StreamingASREvent]:
        text = str(text).strip()
        if not text:
            return []
        session.sequence += 1
        event = StreamingASREvent(
            session_id=session_id,
            sequence=session.sequence,
            text=text,
            is_final=is_final,
            metadata={
                "backend": "pythaiasr",
                "model": self.settings.asr.model,
                **({"timing": timing} if timing else {}),
            },
        )
        session.events.append(event)
        return [event]

    def push_audio(self, session_id: str, samples: Any, sample_rate: int) -> list[StreamingASREvent]:
        """Accept one audio chunk and return newly finalised incremental text."""
        session = self._get_session(session_id)
        received_at = time.perf_counter()
        audio = self._normalise_chunk(samples, sample_rate)
        audio_seconds = len(audio) / self.settings.asr.sampling_rate
        with session.lock:
            if session.closed:
                raise WorkbenchError("Streaming ASR session is already finalised.")
            decoder = self._decoder_for(session)
            try:
                # PyThaiASR's ONNX model is shared and its thread safety is not documented.
                with self._inference_lock:
                    inference_started_at = time.perf_counter()
                    raw_pieces = decoder.process_chunk(audio)
                    inference_ended_at = time.perf_counter()
            except Exception as exc:
                raise WorkbenchError("Thai streaming ASR failed while processing audio.") from exc
            pieces = [raw_pieces] if isinstance(raw_pieces, str) else raw_pieces
            inference_seconds = inference_ended_at - inference_started_at
            emitted_at = time.perf_counter()
            session.audio_seconds_processed += audio_seconds
            session.inference_seconds += inference_seconds
            session.latest_chunk_latency_seconds = emitted_at - received_at
            timing = {
                "chunk_received_at_monotonic_s": received_at,
                "inference_started_at_monotonic_s": inference_started_at,
                "inference_ended_at_monotonic_s": inference_ended_at,
                "event_emitted_at_monotonic_s": emitted_at,
                "audio_seconds": audio_seconds,
                "inference_seconds": inference_seconds,
                "chunk_processing_latency_seconds": emitted_at - received_at,
                "chunk_rtf": inference_seconds / audio_seconds,
            }
            return [
                event
                for piece in pieces
                for event in self._record(session_id, session, piece, is_final=True, timing=timing)
            ]

    def get_events(self, session_id: str) -> list[StreamingASREvent]:
        session = self._get_session(session_id)
        with session.lock:
            return list(session.events)

    def get_metrics(self, session_id: str) -> StreamingSessionMetrics:
        """Return local timing without retaining or exposing raw microphone audio."""
        session = self._get_session(session_id)
        with session.lock:
            return StreamingSessionMetrics(
                session_id=session_id,
                state="stopped" if session.closed else "listening",
                session_duration_seconds=time.perf_counter() - session.started_at,
                audio_seconds_processed=session.audio_seconds_processed,
                inference_seconds=session.inference_seconds,
                session_rtf=(session.inference_seconds / session.audio_seconds_processed)
                if session.audio_seconds_processed
                else None,
                finalized_events=len(session.events),
                latest_chunk_latency_seconds=session.latest_chunk_latency_seconds,
            )

    def finalize(self, session_id: str) -> list[StreamingASREvent]:
        """Flush the remaining decoder context exactly once."""
        session = self._get_session(session_id)
        received_at = time.perf_counter()
        with session.lock:
            if session.closed:
                return []
            decoder = self._decoder_for(session)
            try:
                with self._inference_lock:
                    inference_started_at = time.perf_counter()
                    text = decoder.flush()
                    inference_ended_at = time.perf_counter()
            except Exception as exc:
                raise WorkbenchError("Thai streaming ASR failed while finalising audio.") from exc
            session.closed = True
            inference_seconds = inference_ended_at - inference_started_at
            session.inference_seconds += inference_seconds
            emitted_at = time.perf_counter()
            return self._record(
                session_id,
                session,
                text,
                is_final=True,
                timing={
                    "finalize_received_at_monotonic_s": received_at,
                    "inference_started_at_monotonic_s": inference_started_at,
                    "inference_ended_at_monotonic_s": inference_ended_at,
                    "event_emitted_at_monotonic_s": emitted_at,
                    "finalization_latency_seconds": emitted_at - received_at,
                    "inference_seconds": inference_seconds,
                },
            )

    def close_session(self, session_id: str) -> None:
        """Release application session state after finalisation or cancellation."""
        with self._sessions_lock:
            self._sessions.pop(session_id, None)
