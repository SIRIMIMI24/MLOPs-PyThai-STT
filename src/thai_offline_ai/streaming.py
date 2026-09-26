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

# Startup audio is held only long enough to make the beginning of a browser
# session deterministic.  The buffer is session-local and is drained as soon as
# it reaches this amount (or when the user stops), so it cannot grow over a
# long-running session.
STARTUP_BUFFER_SECONDS = 0.75


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
    model_initialization_seconds: float | None
    first_audio_to_decoder_seconds: float | None
    first_event_latency_seconds: float | None
    startup_buffered_audio_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _Session:
    decoder: Any
    sequence: int = 0
    events: deque[StreamingASREvent] = field(default_factory=lambda: deque(maxlen=256))
    closed: bool = False
    state: str = "preparing"
    started_at: float = field(default_factory=time.perf_counter)
    model_ready_at: float | None = None
    model_initialization_seconds: float | None = None
    first_audio_received_at: float | None = None
    first_decoder_input_at: float | None = None
    first_event_emitted_at: float | None = None
    startup_audio: Any | None = None
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
        """Prepare an isolated decoder before browser capture is enabled.

        Starting a session is an explicit user action, so loading the cached
        local model here preserves normal application startup laziness while
        avoiding a race in which the first microphone chunk waits for model
        initialisation.
        """
        session_id = uuid4().hex
        session = _Session(decoder=None)
        with self._sessions_lock:
            self._sessions[session_id] = session
        preparation_started_at = time.perf_counter()
        try:
            with session.lock:
                self._decoder_for(session)
                session.model_ready_at = time.perf_counter()
                session.model_initialization_seconds = (
                    session.model_ready_at - preparation_started_at
                )
                session.state = "listening"
        except Exception:
            self.close_session(session_id)
            raise
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
                    # The upstream default energy gate can skip a quiet initial
                    # decoder window. Preserve every accepted microphone sample;
                    # future VAD can add explicit endpoint policy separately.
                    silence_threshold=0.0,
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
        if session.first_event_emitted_at is None:
            session.first_event_emitted_at = time.perf_counter()
        return [event]

    def _process_audio(
        self,
        session_id: str,
        session: _Session,
        audio: Any,
        *,
        received_at: float,
        startup_buffered: bool,
    ) -> list[StreamingASREvent]:
        """Send already-normalised PCM to the prepared session decoder."""
        audio_seconds = len(audio) / self.settings.asr.sampling_rate
        decoder = self._decoder_for(session)
        try:
            # PyThaiASR's ONNX model is shared and its thread safety is not documented.
            with self._inference_lock:
                inference_started_at = time.perf_counter()
                if session.first_decoder_input_at is None:
                    session.first_decoder_input_at = inference_started_at
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
            "startup_buffered": startup_buffered,
            "first_audio_to_decoder_seconds": (
                inference_started_at - session.first_audio_received_at
                if session.first_audio_received_at is not None
                else None
            ),
        }
        return [
            event
            for piece in pieces
            for event in self._record(session_id, session, piece, is_final=True, timing=timing)
        ]

    def push_audio(self, session_id: str, samples: Any, sample_rate: int) -> list[StreamingASREvent]:
        """Accept one audio chunk and return newly finalised incremental text."""
        session = self._get_session(session_id)
        received_at = time.perf_counter()
        audio = self._normalise_chunk(samples, sample_rate)
        with session.lock:
            if session.closed:
                raise WorkbenchError("Streaming ASR session is already finalised.")
            if session.first_audio_received_at is None:
                session.first_audio_received_at = received_at
            startup_limit = round(STARTUP_BUFFER_SECONDS * self.settings.asr.sampling_rate)
            if session.startup_audio is not None or session.audio_seconds_processed == 0.0:
                import numpy as np

                buffered = (
                    audio
                    if session.startup_audio is None
                    else np.concatenate((session.startup_audio, audio))
                )
                if len(buffered) < startup_limit:
                    session.startup_audio = buffered
                    return []
                # Keep the retained start window bounded, then immediately pass
                # every remaining sample through as well. No initial sample is
                # discarded when a browser sends an irregular large chunk.
                session.startup_audio = None
                events = self._process_audio(
                    session_id,
                    session,
                    buffered[:startup_limit],
                    received_at=received_at,
                    startup_buffered=True,
                )
                remainder = buffered[startup_limit:]
                if len(remainder):
                    events.extend(
                        self._process_audio(
                            session_id,
                            session,
                            remainder,
                            received_at=received_at,
                            startup_buffered=True,
                        )
                    )
                return events
            return self._process_audio(
                session_id, session, audio, received_at=received_at, startup_buffered=False
            )

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
                state="stopped" if session.closed else session.state,
                session_duration_seconds=time.perf_counter() - session.started_at,
                audio_seconds_processed=session.audio_seconds_processed,
                inference_seconds=session.inference_seconds,
                session_rtf=(session.inference_seconds / session.audio_seconds_processed)
                if session.audio_seconds_processed
                else None,
                finalized_events=len(session.events),
                latest_chunk_latency_seconds=session.latest_chunk_latency_seconds,
                model_initialization_seconds=session.model_initialization_seconds,
                first_audio_to_decoder_seconds=(
                    session.first_decoder_input_at - session.first_audio_received_at
                    if session.first_decoder_input_at is not None
                    and session.first_audio_received_at is not None
                    else None
                ),
                first_event_latency_seconds=(
                    session.first_event_emitted_at - session.first_audio_received_at
                    if session.first_event_emitted_at is not None
                    and session.first_audio_received_at is not None
                    else None
                ),
                startup_buffered_audio_seconds=(
                    len(session.startup_audio) / self.settings.asr.sampling_rate
                    if session.startup_audio is not None
                    else 0.0
                ),
            )

    def finalize(self, session_id: str) -> list[StreamingASREvent]:
        """Flush the remaining decoder context exactly once."""
        session = self._get_session(session_id)
        received_at = time.perf_counter()
        with session.lock:
            if session.closed:
                return []
            decoder = self._decoder_for(session)
            buffered_events: list[StreamingASREvent] = []
            if session.startup_audio is not None:
                buffered = session.startup_audio
                session.startup_audio = None
                buffered_events = self._process_audio(
                    session_id,
                    session,
                    buffered,
                    received_at=session.first_audio_received_at or received_at,
                    startup_buffered=True,
                )
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
            return buffered_events + self._record(
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
