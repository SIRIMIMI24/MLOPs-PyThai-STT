"""Phase 5.5: session-scoped streaming speaker diarization and live ASR fan-out.

The canonical captured stream is normalized once to mono 16 kHz float32 PCM and
fanned out to both local Typhoon ASR and local Nemotron-3 ONNX diarization.
Speaker labels (``SPEAKER_00``, ``SPEAKER_01``, ...) are anonymous and stable
within one streaming session; business roles are never inferred.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from thai_offline_ai.adapters.diarization import model_is_available, resolve_device
from thai_offline_ai.config import Settings
from thai_offline_ai.diarization_types import SpeakerTurn
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.streaming import (
    StreamingASREvent,
    StreamingASRService,
    StreamingSessionMetrics,
)

NEMOTRON_HOP_LENGTH = 160  # 10 ms per mel frame at 16 kHz
NEMOTRON_FRAME_DURATION = 0.01
LEFT_CONTEXT_MEL_FRAMES = 4  # 640 samples (40 ms), exceeds N_FFT/2 + 1 = 257
DEFAULT_ONSET = 0.5
DEFAULT_OFFSET = 0.5
DEFAULT_MIN_DURATION_ON = 0.3
DEFAULT_MIN_DURATION_OFF = 0.5


@dataclass(frozen=True, slots=True)
class StreamingDiarizationMetrics:
    """Process-local timing and progress metrics for one streaming diarization session."""

    session_id: str
    state: str
    device: str
    session_duration_seconds: float
    audio_seconds_processed: float
    inference_seconds: float
    session_rtf: float | None
    turns_detected: int
    speakers: list[str]
    latest_chunk_latency_seconds: float | None
    model_initialization_seconds: float | None
    pending_audio_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StreamingSpeakerMetrics:
    """Combined process-local metrics for Phase 5.5 live ASR + streaming diarization."""

    session_id: str
    state: str
    asr: StreamingSessionMetrics
    diarization: StreamingDiarizationMetrics
    attributed_events: int
    unattributed_events: int
    ambiguous_events: int
    speakers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state,
            "asr": self.asr.to_dict(),
            "diarization": self.diarization.to_dict(),
            "attributed_events": self.attributed_events,
            "unattributed_events": self.unattributed_events,
            "ambiguous_events": self.ambiguous_events,
            "speakers": list(self.speakers),
        }


def _sigmoid(logits: Any) -> Any:
    import numpy as np

    arr = np.asarray(logits, dtype=np.float32)
    z = np.exp(-np.abs(arr))
    return np.where(arr >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def normalize_stream_pcm(samples: Any, sample_rate: int, target_rate: int = 16000) -> Any:
    """Convert raw PCM samples to mono float32 at ``target_rate`` preserving the first sample."""
    try:
        import numpy as np

        if sample_rate <= 0 or target_rate <= 0:
            raise ValueError("invalid sample rate")
        data = np.asarray(samples, dtype=np.float32)
        if data.size == 0 or data.ndim not in {1, 2}:
            raise ValueError("invalid audio")
        mono = data.mean(axis=1) if data.ndim == 2 else data
        if not np.isfinite(mono).all():
            raise ValueError("non-finite audio")
        if sample_rate == target_rate:
            return mono.astype(np.float32, copy=False)
        target_count = max(1, round(len(mono) * target_rate / sample_rate))
        source_axis = np.linspace(0, len(mono) - 1, len(mono), dtype=np.float64)
        target_axis = np.linspace(0, len(mono) - 1, target_count, dtype=np.float64)
        return np.interp(target_axis, source_axis, mono).astype(np.float32)
    except WorkbenchError:
        raise
    except Exception as exc:
        raise WorkbenchError("Streaming audio must be a non-empty readable PCM audio chunk.") from exc


def attribute_span_to_speaker(
    start: float | None,
    end: float | None,
    turns: Sequence[SpeakerTurn],
    *,
    diarized_until: float | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Deterministically assign one time span to the greatest-overlap speaker turn.

    Matches Phase 5's tie-breaking and ambiguity metadata without rewriting ASR text
    or inventing business roles.
    """
    metadata: dict[str, Any] = {}
    if start is None or end is None or end <= start:
        metadata["speaker_alignment"] = "missing_asr_timestamps"
        return None, metadata
    candidates = [
        (max(0.0, min(end, turn.end) - max(start, turn.start)), turn)
        for turn in turns
    ]
    candidates = [(amount, turn) for amount, turn in candidates if amount > 0]
    if not candidates:
        if diarized_until is not None and end > diarized_until:
            metadata["speaker_alignment"] = "pending_diarization"
        else:
            metadata["speaker_alignment"] = "no_overlap"
        return None, metadata
    highest = max(amount for amount, _ in candidates)
    winners = [turn for amount, turn in candidates if amount == highest]
    winner = min(winners, key=lambda turn: (turn.start, turn.speaker))
    metadata["speaker_alignment"] = "greatest_overlap"
    if len(candidates) > 1:
        metadata["speaker_alignment_ambiguous"] = True
        metadata["overlapping_speakers"] = [turn.speaker for _, turn in candidates]
    if len(winners) > 1:
        metadata["speaker_alignment_tie"] = True
    return winner.speaker, metadata


@dataclass(slots=True)
class _DiarizationSession:
    cache: Any
    device: str
    num_speakers: int
    subsampling_factor: int
    pending_audio: Any | None = None
    left_tail: Any | None = None
    frames_processed: int = 0
    total_samples_received: int = 0
    active_start_frame: dict[int, int] = field(default_factory=dict)
    speaker_labels: dict[int, str] = field(default_factory=dict)
    turns: deque[SpeakerTurn] = field(default_factory=lambda: deque(maxlen=512))
    closed: bool = False
    state: str = "preparing"
    started_at: float = field(default_factory=time.perf_counter)
    model_initialization_seconds: float | None = None
    audio_seconds_processed: float = 0.0
    inference_seconds: float = 0.0
    latest_chunk_latency_seconds: float | None = None
    lock: Lock = field(default_factory=Lock)


class StreamingDiarizationService:
    """Incremental session-scoped Nemotron-3 ONNX speaker diarization service.

    Each session owns an isolated ``NumpySpeakerCache`` so speaker IDs remain
    consistent across streaming chunks with bounded memory. The ONNX sessions
    are loaded lazily on first session start and serialized during inference.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        onset: float = DEFAULT_ONSET,
        offset: float = DEFAULT_OFFSET,
        min_duration_on: float = DEFAULT_MIN_DURATION_ON,
        min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
    ) -> None:
        self.settings = settings
        self.onset = onset
        self.offset = offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self._engine: Any | None = None
        self._device: str = "cpu"
        self._model_lock = Lock()
        self._inference_lock = Lock()
        self._sessions: dict[str, _DiarizationSession] = {}
        self._sessions_lock = Lock()

    def _load_engine(self) -> tuple[Any, str]:
        config = self.settings.diarization
        if not config.enabled:
            raise WorkbenchError(
                "Speaker diarization is disabled. Set diarization.enabled=true after local model setup."
            )
        if self._engine is not None:
            return self._engine, self._device
        with self._model_lock:
            if self._engine is not None:
                return self._engine, self._device
            if not model_is_available(config.model_dir, config.precision):
                raise WorkbenchError(
                    "Diarization model is not available locally. Run: python scripts/setup_diarization_model.py"
                )
            device = resolve_device(config.device)
            try:
                os.environ["PYTHAIASR_DATA_DIR"] = str(Path(config.model_dir).parent)
                from pythaiasr.diarization import NemotronDiarization

                self._engine = NemotronDiarization(
                    model_dir=config.model_dir,
                    precision=config.precision,
                    device=device,
                )
                self._device = device
            except (ImportError, OSError) as exc:
                raise WorkbenchError(
                    'PyThaiASR diarization is unavailable. Run: python -m pip install -e ".[asr,dev]"'
                ) from exc
            except WorkbenchError:
                raise
            except Exception as exc:
                raise WorkbenchError(
                    "Local streaming diarization could not initialize. Confirm the local Nemotron model cache."
                ) from exc
        return self._engine, self._device

    def _create_cache(self, engine: Any) -> Any:
        from pythaiasr.diarization import NumpySpeakerCache

        return NumpySpeakerCache(
            engine.constants,
            fifo_length=int(engine.fifo_length),
            speaker_cache_update_period=int(engine.speaker_cache_update_period),
        )

    def _step_and_right_frames(self, subsampling_factor: int) -> tuple[int, int]:
        step_embeds = max(
            1,
            round(
                self.settings.diarization.stream_chunk_seconds
                / (NEMOTRON_FRAME_DURATION * subsampling_factor)
            ),
        )
        right_embeds = max(
            0,
            round(
                self.settings.diarization.stream_right_context_seconds
                / (NEMOTRON_FRAME_DURATION * subsampling_factor)
            ),
        )
        return step_embeds * subsampling_factor, right_embeds * subsampling_factor

    def start_session(self) -> str:
        """Prepare an isolated diarization session and cache before microphone capture."""
        session_id = uuid4().hex
        started = time.perf_counter()
        engine, device = self._load_engine()
        try:
            cache = self._create_cache(engine)
            num_speakers = int(engine.constants.get("num_speakers", 8))
            subsampling_factor = int(engine.subsampling_factor)
        except Exception as exc:
            raise WorkbenchError("Local streaming diarization could not create session state.") from exc
        ready = time.perf_counter()
        session = _DiarizationSession(
            cache=cache,
            device=device,
            num_speakers=num_speakers,
            subsampling_factor=subsampling_factor,
            state="listening",
            started_at=started,
            model_initialization_seconds=ready - started,
        )
        with self._sessions_lock:
            self._sessions[session_id] = session
        return session_id

    def _get_session(self, session_id: str) -> _DiarizationSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise WorkbenchError("Streaming diarization session was not found or has already closed.")
        return session

    def _label_for(self, session: _DiarizationSession, speaker_slot: int) -> str:
        return session.speaker_labels.setdefault(
            speaker_slot, f"SPEAKER_{len(session.speaker_labels):02d}"
        )

    def _append_or_merge_turn(
        self,
        session: _DiarizationSession,
        speaker_slot: int,
        start_s: float,
        end_s: float,
    ) -> SpeakerTurn | None:
        start_s = round(start_s, 3)
        end_s = round(end_s, 3)
        if end_s <= start_s or (end_s - start_s) < self.min_duration_on:
            return None
        label = self._label_for(session, speaker_slot)
        metadata = {"backend_speaker": f"SPEAKER_{speaker_slot:02d}"}
        if label == metadata["backend_speaker"]:
            metadata = {}
        # Merge brief same-speaker gaps smaller than min_duration_off.
        for idx in range(len(session.turns) - 1, -1, -1):
            prev = session.turns[idx]
            if prev.speaker == label:
                if (start_s - prev.end) < self.min_duration_off:
                    merged = SpeakerTurn(
                        start=prev.start,
                        end=max(prev.end, end_s),
                        speaker=label,
                        metadata=dict(prev.metadata),
                    )
                    session.turns[idx] = merged
                    return merged
                break
        turn = SpeakerTurn(start=start_s, end=end_s, speaker=label, metadata=metadata)
        session.turns.append(turn)
        return turn

    def _update_hysteresis(
        self,
        session: _DiarizationSession,
        frame_probs: Any,
        *,
        close_open: bool = False,
    ) -> list[SpeakerTurn]:
        emitted: list[SpeakerTurn] = []
        base_frame = session.frames_processed
        num_frames = int(frame_probs.shape[0])
        num_slots = int(frame_probs.shape[1])
        for slot in range(num_slots):
            probs = frame_probs[:, slot]
            for offset_frame in range(num_frames):
                frame_idx = base_frame + offset_frame
                prob = float(probs[offset_frame])
                if slot not in session.active_start_frame:
                    if prob >= self.onset:
                        session.active_start_frame[slot] = frame_idx
                elif prob < self.offset:
                    start_frame = session.active_start_frame.pop(slot)
                    turn = self._append_or_merge_turn(
                        session,
                        slot,
                        start_frame * NEMOTRON_FRAME_DURATION,
                        frame_idx * NEMOTRON_FRAME_DURATION,
                    )
                    if turn is not None:
                        emitted.append(turn)
        session.frames_processed += num_frames
        if close_open:
            end_s = max(
                session.frames_processed * NEMOTRON_FRAME_DURATION,
                session.total_samples_received / self.settings.diarization.sampling_rate,
            )
            for slot in sorted(session.active_start_frame):
                start_frame = session.active_start_frame.pop(slot)
                turn = self._append_or_merge_turn(
                    session,
                    slot,
                    start_frame * NEMOTRON_FRAME_DURATION,
                    end_s,
                )
                if turn is not None:
                    emitted.append(turn)
        emitted.sort(key=lambda item: (item.start, item.end, item.speaker))
        return emitted

    def _run_diarization_window(
        self,
        session: _DiarizationSession,
        window_audio: Any,
        *,
        step_mel_frames: int,
        right_mel_frames: int,
        received_at: float,
        close_open: bool,
    ) -> list[SpeakerTurn]:
        import numpy as np

        engine, _ = self._load_engine()
        sub = session.subsampling_factor
        left_samples = LEFT_CONTEXT_MEL_FRAMES * NEMOTRON_HOP_LENGTH
        if session.left_tail is not None and len(session.left_tail) == left_samples:
            combined = np.concatenate((session.left_tail, window_audio))
            left_mel = LEFT_CONTEXT_MEL_FRAMES
        else:
            combined = window_audio
            left_mel = 0

        try:
            with self._inference_lock:
                started = time.perf_counter()
                mel = engine.log_mel(combined)
                available_mel = max(0, int(mel.shape[1]) - left_mel - 1)
                valid_step_mel = min(step_mel_frames, available_mel)
                if valid_step_mel <= 0 and int(mel.shape[1]) > left_mel:
                    valid_step_mel = min(step_mel_frames, int(mel.shape[1]) - left_mel)
                if valid_step_mel <= 0:
                    return []
                desired_mel = min(valid_step_mel + right_mel_frames, int(mel.shape[1]) - left_mel)
                chunk_mel = mel[:, left_mel : left_mel + desired_mel]
                valid_mel_len = min(desired_mel, max(1, available_mel))
                num_chunk_embeds = max(1, math.ceil(valid_step_mel / sub))
                min_required_mel = max(num_chunk_embeds * sub, math.ceil(desired_mel / sub) * sub)
                if chunk_mel.shape[1] < min_required_mel:
                    pad_amount = min_required_mel - chunk_mel.shape[1]
                    chunk_mel = np.pad(chunk_mel, ((0, 0), (0, pad_amount), (0, 0)))
                context_embeds = session.cache.get_embeds()
                context_len = int(context_embeds.shape[1])
                logits, chunk_input_embeds = engine.run_chunk(
                    chunk_mel, valid_mel_len, context_embeds
                )
                session.cache.update(chunk_input_embeds, logits, num_chunk_embeds)
                ended = time.perf_counter()
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("Local streaming diarization failed while processing audio.") from exc

        start_logit = context_len * sub
        step_logits = logits[0, start_logit : start_logit + valid_step_mel]
        probs = _sigmoid(step_logits)
        inference_seconds = ended - started
        session.inference_seconds += inference_seconds
        session.audio_seconds_processed += valid_step_mel * NEMOTRON_FRAME_DURATION
        session.latest_chunk_latency_seconds = time.perf_counter() - received_at
        return self._update_hysteresis(session, probs, close_open=close_open)

    def push_audio(self, session_id: str, samples: Any, sample_rate: int) -> list[SpeakerTurn]:
        """Accept one audio chunk, process ready diarization steps, and return new turns."""
        import numpy as np

        session = self._get_session(session_id)
        received_at = time.perf_counter()
        audio = normalize_stream_pcm(samples, sample_rate, self.settings.diarization.sampling_rate)
        step_mel, right_mel = self._step_and_right_frames(session.subsampling_factor)
        step_samples = step_mel * NEMOTRON_HOP_LENGTH
        right_samples = right_mel * NEMOTRON_HOP_LENGTH
        left_samples = LEFT_CONTEXT_MEL_FRAMES * NEMOTRON_HOP_LENGTH

        emitted: list[SpeakerTurn] = []
        with session.lock:
            if session.closed:
                raise WorkbenchError("Streaming diarization session is already finalised.")
            session.total_samples_received += len(audio)
            pending = (
                audio
                if session.pending_audio is None
                else np.concatenate((session.pending_audio, audio))
            )
            while len(pending) >= step_samples + right_samples:
                window = pending[: step_samples + right_samples]
                emitted.extend(
                    self._run_diarization_window(
                        session,
                        window,
                        step_mel_frames=step_mel,
                        right_mel_frames=right_mel,
                        received_at=received_at,
                        close_open=False,
                    )
                )
                consumed = pending[:step_samples]
                if len(consumed) >= left_samples:
                    session.left_tail = consumed[-left_samples:].copy()
                elif session.left_tail is not None:
                    merged_tail = np.concatenate((session.left_tail, consumed))
                    session.left_tail = merged_tail[-left_samples:].copy()
                pending = pending[step_samples:]
            session.pending_audio = pending if len(pending) > 0 else None
        return emitted

    def finalize(self, session_id: str) -> list[SpeakerTurn]:
        """Flush any remaining buffered audio and close all open speaker turns."""
        session = self._get_session(session_id)
        received_at = time.perf_counter()
        with session.lock:
            if session.closed:
                return []
            emitted: list[SpeakerTurn] = []
            if session.pending_audio is not None and len(session.pending_audio) > 0:
                pending = session.pending_audio
                session.pending_audio = None
                remaining_mel = max(1, len(pending) // NEMOTRON_HOP_LENGTH)
                emitted.extend(
                    self._run_diarization_window(
                        session,
                        pending,
                        step_mel_frames=remaining_mel,
                        right_mel_frames=0,
                        received_at=received_at,
                        close_open=True,
                    )
                )
            else:
                import numpy as np

                empty = np.zeros((0, session.num_speakers), dtype=np.float32)
                emitted.extend(self._update_hysteresis(session, empty, close_open=True))
            session.closed = True
            session.state = "stopped"
            return emitted

    def get_turns(self, session_id: str, *, include_open: bool = False) -> list[SpeakerTurn]:
        """Return chronological speaker turns; optionally include currently-active live turns."""
        session = self._get_session(session_id)
        with session.lock:
            turns = list(session.turns)
            if include_open and not session.closed:
                current_end = round(session.frames_processed * NEMOTRON_FRAME_DURATION, 3)
                for slot, start_frame in sorted(session.active_start_frame.items()):
                    start_s = round(start_frame * NEMOTRON_FRAME_DURATION, 3)
                    if current_end > start_s:
                        label = self._label_for(session, slot)
                        turns.append(
                            SpeakerTurn(
                                start=start_s,
                                end=current_end,
                                speaker=label,
                                metadata={"turn_state": "open"},
                            )
                        )
            turns.sort(key=lambda item: (item.start, item.end, item.speaker))
            return turns

    def get_speakers(self, session_id: str) -> list[str]:
        return sorted({turn.speaker for turn in self.get_turns(session_id, include_open=True)})

    def get_metrics(self, session_id: str) -> StreamingDiarizationMetrics:
        session = self._get_session(session_id)
        with session.lock:
            pending_seconds = (
                len(session.pending_audio) / self.settings.diarization.sampling_rate
                if session.pending_audio is not None
                else 0.0
            )
            turns = list(session.turns)
            speakers = sorted(
                {turn.speaker for turn in turns}
                | {
                    session.speaker_labels[slot]
                    for slot in session.active_start_frame
                    if slot in session.speaker_labels
                }
            )
            return StreamingDiarizationMetrics(
                session_id=session_id,
                state="stopped" if session.closed else session.state,
                device=session.device,
                session_duration_seconds=time.perf_counter() - session.started_at,
                audio_seconds_processed=session.audio_seconds_processed,
                inference_seconds=session.inference_seconds,
                session_rtf=(session.inference_seconds / session.audio_seconds_processed)
                if session.audio_seconds_processed
                else None,
                turns_detected=len(turns),
                speakers=speakers,
                latest_chunk_latency_seconds=session.latest_chunk_latency_seconds,
                model_initialization_seconds=session.model_initialization_seconds,
                pending_audio_seconds=pending_seconds,
            )

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)


@dataclass(slots=True)
class _CombinedSession:
    asr_session_id: str
    diarization_session_id: str
    stream_cursor_seconds: float = 0.0
    last_event_end_seconds: float = 0.0
    events: deque[StreamingASREvent] = field(default_factory=lambda: deque(maxlen=256))
    closed: bool = False
    lock: Lock = field(default_factory=Lock)


class StreamingSpeakerASRService:
    """Fan out one canonical 16 kHz PCM stream to live ASR and live diarization."""

    def __init__(
        self,
        settings: Settings,
        *,
        asr_service: StreamingASRService | None = None,
        diarization_service: StreamingDiarizationService | None = None,
    ) -> None:
        self.settings = settings
        self.asr_service = asr_service or StreamingASRService(settings)
        self.diarization_service = diarization_service or StreamingDiarizationService(settings)
        self._sessions: dict[str, _CombinedSession] = {}
        self._sessions_lock = Lock()

    def start_session(
        self, *, model: str | None = None, prompt_mode: str | None = None
    ) -> str:
        """Prepare both local ASR and diarization sessions before live capture."""
        if model is None and prompt_mode is None:
            asr_session_id = self.asr_service.start_session()
        else:
            asr_session_id = self.asr_service.start_session(model=model, prompt_mode=prompt_mode)
        try:
            diarization_session_id = self.diarization_service.start_session()
        except Exception:
            self.asr_service.close_session(asr_session_id)
            raise
        session_id = uuid4().hex
        with self._sessions_lock:
            self._sessions[session_id] = _CombinedSession(
                asr_session_id=asr_session_id,
                diarization_session_id=diarization_session_id,
            )
        return session_id

    def _get_session(self, session_id: str) -> _CombinedSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise WorkbenchError("Live speaker ASR session was not found or has already closed.")
        return session

    def _attribute_event(
        self,
        session_id: str,
        event: StreamingASREvent,
        start_s: float,
        end_s: float,
        turns: Sequence[SpeakerTurn],
        *,
        diarized_until: float | None,
    ) -> StreamingASREvent:
        speaker, align_meta = attribute_span_to_speaker(
            start_s, end_s, turns, diarized_until=diarized_until
        )
        metadata = dict(event.metadata)
        metadata.pop("speaker_alignment_ambiguous", None)
        metadata.pop("overlapping_speakers", None)
        metadata.pop("speaker_alignment_tie", None)
        metadata.update(align_meta)
        metadata["diarization_backend"] = "pythaiasr"
        metadata["diarization_model"] = self.settings.diarization.model
        return StreamingASREvent(
            session_id=session_id,
            sequence=event.sequence,
            text=event.text,
            is_final=event.is_final,
            start=round(start_s, 3),
            end=round(end_s, 3),
            speaker=speaker,
            confidence=event.confidence,
            metadata=metadata,
        )

    def _refresh_attributions(
        self,
        session_id: str,
        session: _CombinedSession,
        *,
        include_open: bool,
        finalized: bool = False,
    ) -> None:
        turns = self.diarization_service.get_turns(
            session.diarization_session_id, include_open=include_open
        )
        diarized_until = (
            None
            if finalized
            else self.diarization_service.get_metrics(
                session.diarization_session_id
            ).audio_seconds_processed
        )
        refreshed = deque(maxlen=256)
        for existing in session.events:
            if existing.start is not None and existing.end is not None:
                refreshed.append(
                    self._attribute_event(
                        session_id,
                        existing,
                        existing.start,
                        existing.end,
                        turns,
                        diarized_until=diarized_until,
                    )
                )
            else:
                refreshed.append(existing)
        session.events = refreshed

    def push_audio(self, session_id: str, samples: Any, sample_rate: int) -> list[StreamingASREvent]:
        """Normalize one incoming chunk once and fan it out to ASR and diarization."""
        session = self._get_session(session_id)
        target_rate = self.settings.asr.sampling_rate
        canonical = normalize_stream_pcm(samples, sample_rate, target_rate)
        chunk_duration = len(canonical) / target_rate

        with session.lock:
            if session.closed:
                raise WorkbenchError("Live speaker ASR session is already finalised.")
            prev_cursor = session.stream_cursor_seconds
            new_cursor = prev_cursor + chunk_duration
            session.stream_cursor_seconds = new_cursor

            # Fan out the exact same mono 16 kHz PCM stream to both engines.
            self.diarization_service.push_audio(
                session.diarization_session_id, canonical, target_rate
            )
            raw_events = self.asr_service.push_audio(session.asr_session_id, canonical, target_rate)

            turns = self.diarization_service.get_turns(
                session.diarization_session_id, include_open=True
            )
            diarized_until = self.diarization_service.get_metrics(
                session.diarization_session_id
            ).audio_seconds_processed

            attributed: list[StreamingASREvent] = []
            if raw_events:
                span_start = session.last_event_end_seconds
                span_end = max(new_cursor, span_start + 0.001)
                slice_width = (span_end - span_start) / len(raw_events)
                for idx, raw_event in enumerate(raw_events):
                    ev_start = span_start + idx * slice_width
                    ev_end = span_start + (idx + 1) * slice_width
                    item = self._attribute_event(
                        session_id,
                        raw_event,
                        ev_start,
                        ev_end,
                        turns,
                        diarized_until=diarized_until,
                    )
                    session.events.append(item)
                    attributed.append(item)
                session.last_event_end_seconds = span_end

            # Update pending earlier events as new diarization turns arrive.
            self._refresh_attributions(session_id, session, include_open=True, finalized=False)
            return attributed

    def finalize(self, session_id: str) -> list[StreamingASREvent]:
        """Finalize diarization and ASR, then reconcile speaker attribution on all events."""
        session = self._get_session(session_id)
        with session.lock:
            if session.closed:
                return []
            self.diarization_service.finalize(session.diarization_session_id)
            raw_events = self.asr_service.finalize(session.asr_session_id)
            turns = self.diarization_service.get_turns(
                session.diarization_session_id, include_open=False
            )
            attributed: list[StreamingASREvent] = []
            if raw_events:
                if session.last_event_end_seconds < session.stream_cursor_seconds:
                    span_start = session.last_event_end_seconds
                    span_end = session.stream_cursor_seconds
                else:
                    # RealtimeStreamASR.flush() decodes trailing right-context audio
                    # ending at the canonical stream cursor rather than past it.
                    trailing_window = min(session.stream_cursor_seconds, 0.5)
                    span_end = session.stream_cursor_seconds
                    span_start = max(0.0, span_end - trailing_window)
                span_end = max(span_end, span_start + 0.001)
                slice_width = (span_end - span_start) / len(raw_events)
                for idx, raw_event in enumerate(raw_events):
                    ev_start = span_start + idx * slice_width
                    ev_end = span_start + (idx + 1) * slice_width
                    item = self._attribute_event(
                        session_id,
                        raw_event,
                        ev_start,
                        ev_end,
                        turns,
                        diarized_until=None,
                    )
                    session.events.append(item)
                    attributed.append(item)
                session.last_event_end_seconds = span_end
            self._refresh_attributions(session_id, session, include_open=False, finalized=True)
            session.closed = True
            return attributed

    def get_events(self, session_id: str) -> list[StreamingASREvent]:
        session = self._get_session(session_id)
        with session.lock:
            return list(session.events)

    def get_turns(self, session_id: str, *, include_open: bool = False) -> list[SpeakerTurn]:
        session = self._get_session(session_id)
        return self.diarization_service.get_turns(
            session.diarization_session_id, include_open=include_open
        )

    def format_transcript(self, session_id: str) -> str:
        """Render readable speaker-labelled lines without mutating raw ASR text."""
        events = self.get_events(session_id)
        lines: list[str] = []
        current_speaker: str | None = None
        current_parts: list[str] = []
        for event in events:
            label = event.speaker or "UNATTRIBUTED"
            if label == current_speaker:
                current_parts.append(event.text)
            else:
                if current_parts and current_speaker is not None:
                    lines.append(f"[{current_speaker}] {' '.join(current_parts)}")
                current_speaker = label
                current_parts = [event.text]
        if current_parts and current_speaker is not None:
            lines.append(f"[{current_speaker}] {' '.join(current_parts)}")
        return "\n".join(lines)

    def get_metrics(self, session_id: str) -> StreamingSpeakerMetrics:
        session = self._get_session(session_id)
        with session.lock:
            asr_metrics = self.asr_service.get_metrics(session.asr_session_id)
            diar_metrics = self.diarization_service.get_metrics(session.diarization_session_id)
            events = list(session.events)
            attributed = sum(1 for item in events if item.speaker is not None)
            ambiguous = sum(
                1 for item in events if item.metadata.get("speaker_alignment_ambiguous") is True
            )
            return StreamingSpeakerMetrics(
                session_id=session_id,
                state="stopped" if session.closed else asr_metrics.state,
                asr=asr_metrics,
                diarization=diar_metrics,
                attributed_events=attributed,
                unattributed_events=len(events) - attributed,
                ambiguous_events=ambiguous,
                speakers=diar_metrics.speakers,
            )

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            self.asr_service.close_session(session.asr_session_id)
            self.diarization_service.close_session(session.diarization_session_id)
