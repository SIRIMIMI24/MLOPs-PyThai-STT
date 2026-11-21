import json

import numpy as np
import pytest
import soundfile as sf

from thai_offline_ai.cli import main
from thai_offline_ai.config import AppConfig, ConfigurationError, Settings, load_settings
from thai_offline_ai.diarization_types import SpeakerTurn
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.streaming import StreamingASRService
from thai_offline_ai.streaming_diarization import (
    StreamingDiarizationService,
    StreamingSpeakerASRService,
    attribute_span_to_speaker,
)


class FakeSpeakerCache:
    def __init__(self, hidden_size: int = 8) -> None:
        self.hidden_size = hidden_size
        self.frames = 0

    def get_embeds(self) -> np.ndarray:
        return np.zeros((1, self.frames, self.hidden_size), dtype=np.float32)

    def update(self, chunk_input_embeds, chunk_logits, num_chunk_frames: int) -> None:
        self.frames = min(self.frames + num_chunk_frames, 40)


class FakeNemotronEngine:
    """Deterministic fake engine: speaker slot 2 active for [0..2s), slot 5 for [2..4s)."""

    subsampling_factor = 8
    fifo_length = 40
    speaker_cache_update_period = 300

    def __init__(self) -> None:
        self.constants = {"num_speakers": 8, "hidden_size": 8}
        self.calls: list[int] = []

    def log_mel(self, waveform: np.ndarray) -> np.ndarray:
        num_frames = 1 + len(waveform) // 160
        return np.zeros((1, num_frames, 128), dtype=np.float32)

    def run_chunk(self, chunk_mel: np.ndarray, chunk_mel_length: int, context_embeds: np.ndarray):
        self.calls.append(chunk_mel_length)
        context_len = int(context_embeds.shape[1])
        chunk_embeds = chunk_mel.shape[1] // self.subsampling_factor
        total_mel = (context_len + chunk_embeds) * self.subsampling_factor
        logits = np.full((1, total_mel, 8), -6.0, dtype=np.float32)
        start_idx = context_len * self.subsampling_factor
        base_frame = context_len * self.subsampling_factor
        for offset in range(chunk_embeds * self.subsampling_factor):
            global_frame = base_frame + offset
            if 10 <= global_frame < 180:
                logits[0, start_idx + offset, 2] = 6.0
            elif 210 <= global_frame < 380:
                logits[0, start_idx + offset, 5] = 6.0
        embeds = np.zeros((1, context_len + chunk_embeds, 8), dtype=np.float32)
        return logits, embeds


class FakeRealtimeStreamASR:
    def __init__(self, model, sample_rate, step_sec, silence_threshold):
        self.chunks: list[np.ndarray] = []

    def process_chunk(self, chunk: np.ndarray):
        self.chunks.append(chunk)
        idx = len(self.chunks)
        if idx == 1:
            return ["สวัสดีครับ"]
        if idx == 3:
            return ["ยินดีที่ได้คุยกัน"]
        return []

    def flush(self):
        return "ขอบคุณครับ"


def enabled_settings(tmp_path) -> Settings:
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "out")))
    settings.asr.enabled = True
    settings.diarization.enabled = True
    settings.diarization.stream_chunk_seconds = 1.6
    settings.diarization.stream_right_context_seconds = 0.32
    return settings


def fake_diarization_service(tmp_path, monkeypatch) -> tuple[StreamingDiarizationService, FakeNemotronEngine]:
    service = StreamingDiarizationService(enabled_settings(tmp_path))
    engine = FakeNemotronEngine()
    service._engine = engine
    service._device = "cpu"
    monkeypatch.setattr(service, "_create_cache", lambda eng: FakeSpeakerCache())
    return service, engine


def test_attribute_span_to_speaker_overlap_tie_ambiguity_and_pending():
    turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00"), SpeakerTurn(2.0, 4.0, "SPEAKER_01")]
    speaker, meta = attribute_span_to_speaker(0.2, 1.8, turns)
    assert speaker == "SPEAKER_00"
    assert meta["speaker_alignment"] == "greatest_overlap"
    assert "speaker_alignment_ambiguous" not in meta

    speaker, meta = attribute_span_to_speaker(1.0, 3.0, turns)
    assert speaker == "SPEAKER_00"
    assert meta["speaker_alignment_tie"] is True
    assert meta["speaker_alignment_ambiguous"] is True
    assert meta["overlapping_speakers"] == ["SPEAKER_00", "SPEAKER_01"]

    speaker, meta = attribute_span_to_speaker(4.5, 5.0, turns, diarized_until=4.0)
    assert speaker is None
    assert meta["speaker_alignment"] == "pending_diarization"

    speaker, meta = attribute_span_to_speaker(4.5, 5.0, turns, diarized_until=6.0)
    assert speaker is None
    assert meta["speaker_alignment"] == "no_overlap"

    speaker, meta = attribute_span_to_speaker(None, None, turns)
    assert speaker is None
    assert meta["speaker_alignment"] == "missing_asr_timestamps"


def test_streaming_diarization_disabled_and_missing_cache_are_actionable(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "out")))
    with pytest.raises(WorkbenchError, match="disabled"):
        StreamingDiarizationService(settings).start_session()
    settings.diarization.enabled = True
    settings.diarization.model_dir = str(tmp_path / "missing-diarization")
    with pytest.raises(WorkbenchError, match="setup_diarization"):
        StreamingDiarizationService(settings).start_session()


def test_streaming_diarization_tracks_two_speakers_incrementally_and_flushes(tmp_path, monkeypatch):
    service, _ = fake_diarization_service(tmp_path, monkeypatch)
    session_id = service.start_session()

    # Feed 4 seconds of stereo 22050 Hz audio in 1-second chunks.
    stereo_1s = np.zeros((22050, 2), dtype=np.float32)
    service.push_audio(session_id, stereo_1s, 22050)
    # Still buffering before step (1.6s) + right context (0.32s) = 1.92s.
    assert service.get_turns(session_id, include_open=True) == []

    service.push_audio(session_id, stereo_1s, 22050)
    # First 1.6s window has been processed; speaker slot 2 is active and still open at 1.6s.
    live_turns = service.get_turns(session_id, include_open=True)
    assert len(live_turns) == 1
    assert live_turns[0].speaker == "SPEAKER_00"
    assert live_turns[0].metadata["turn_state"] == "open"

    service.push_audio(session_id, stereo_1s, 22050)
    service.push_audio(session_id, stereo_1s, 22050)
    service.finalize(session_id)

    final_turns = service.get_turns(session_id, include_open=False)
    assert [turn.speaker for turn in final_turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert final_turns[0].metadata["backend_speaker"] == "SPEAKER_02"
    assert final_turns[1].metadata["backend_speaker"] == "SPEAKER_05"

    metrics = service.get_metrics(session_id)
    assert metrics.state == "stopped"
    assert metrics.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert metrics.audio_seconds_processed >= 3.9
    assert metrics.session_rtf is not None

    with pytest.raises(WorkbenchError, match="already finalised"):
        service.push_audio(session_id, np.ones(1600, dtype=np.float32), 16000)

    service.close_session(session_id)
    with pytest.raises(WorkbenchError, match="not found"):
        service.get_turns(session_id)


def test_streaming_diarization_short_session_flushes_without_error(tmp_path, monkeypatch):
    service, engine = fake_diarization_service(tmp_path, monkeypatch)
    session_id = service.start_session()
    service.push_audio(session_id, np.zeros(800, dtype=np.float32), 16000)
    service.finalize(session_id)
    assert len(engine.calls) == 1
    assert service.get_metrics(session_id).state == "stopped"


def test_streaming_speaker_asr_single_fanout_and_reconciliation(tmp_path, monkeypatch):
    import pythaiasr

    monkeypatch.setattr(pythaiasr, "RealtimeStreamASR", FakeRealtimeStreamASR)
    settings = enabled_settings(tmp_path)
    asr_service = StreamingASRService(settings)
    asr_service._model = object()
    diar_service, _ = fake_diarization_service(tmp_path, monkeypatch)
    combined = StreamingSpeakerASRService(
        settings, asr_service=asr_service, diarization_service=diar_service
    )

    session_id = combined.start_session()
    # Chunk 1 (1.0s): ASR emits "สวัสดีครับ" before diarization's 1.92s first window.
    first_events = combined.push_audio(session_id, np.ones(16000, dtype=np.float32), 16000)
    assert len(first_events) == 1
    assert first_events[0].text == "สวัสดีครับ"
    assert first_events[0].speaker is None
    assert first_events[0].metadata["speaker_alignment"] == "pending_diarization"

    # Chunk 2 (1.0s): diarization runs its first 1.6s window -> reconciles event 1 to SPEAKER_00!
    combined.push_audio(session_id, np.ones(16000, dtype=np.float32), 16000)
    reconciled = combined.get_events(session_id)
    assert reconciled[0].text == "สวัสดีครับ"
    assert reconciled[0].speaker == "SPEAKER_00"

    # Chunks 3-4 (2.0s more): second speaker arrives.
    combined.push_audio(session_id, np.ones(16000, dtype=np.float32), 16000)
    combined.push_audio(session_id, np.ones(16000, dtype=np.float32), 16000)
    combined.finalize(session_id)

    all_events = combined.get_events(session_id)
    assert [ev.text for ev in all_events] == ["สวัสดีครับ", "ยินดีที่ได้คุยกัน", "ขอบคุณครับ"]
    assert all_events[0].speaker == "SPEAKER_00"
    assert all_events[-1].speaker == "SPEAKER_01"

    transcript = combined.format_transcript(session_id)
    assert "[SPEAKER_00]" in transcript and "[SPEAKER_01]" in transcript
    metrics = combined.get_metrics(session_id)
    assert metrics.state == "stopped"
    assert metrics.attributed_events == 3
    assert metrics.speakers == ["SPEAKER_00", "SPEAKER_01"]

    combined.close_session(session_id)
    with pytest.raises(WorkbenchError, match="not found"):
        combined.get_events(session_id)


def test_finalize_flush_attributes_trailing_right_context_within_stream_bounds(tmp_path, monkeypatch):
    """When push_audio already emitted at stream end, finalize() flush must still overlap the trailing speaker."""
    import pythaiasr

    class TrailingFlushASR(FakeRealtimeStreamASR):
        def process_chunk(self, chunk: np.ndarray):
            self.chunks.append(chunk)
            return [f"ชิ้นที่{len(self.chunks)}"]

        def flush(self):
            return "คำท้าย"

    monkeypatch.setattr(pythaiasr, "RealtimeStreamASR", TrailingFlushASR)
    settings = enabled_settings(tmp_path)
    asr_service = StreamingASRService(settings)
    asr_service._model = object()
    diar_service, _ = fake_diarization_service(tmp_path, monkeypatch)
    combined = StreamingSpeakerASRService(
        settings, asr_service=asr_service, diarization_service=diar_service
    )
    session_id = combined.start_session()
    for _ in range(4):
        combined.push_audio(session_id, np.ones(16000, dtype=np.float32), 16000)
    flushed = combined.finalize(session_id)
    assert len(flushed) == 1
    assert flushed[0].text == "คำท้าย"
    assert flushed[0].end <= 4.0
    assert flushed[0].speaker == "SPEAKER_01"
    combined.close_session(session_id)


def test_streaming_speaker_start_cleans_up_asr_session_if_diarization_fails(tmp_path, monkeypatch):
    import pythaiasr

    monkeypatch.setattr(pythaiasr, "RealtimeStreamASR", FakeRealtimeStreamASR)
    settings = enabled_settings(tmp_path)
    settings.diarization.enabled = False
    asr_service = StreamingASRService(settings)
    asr_service._model = object()
    combined = StreamingSpeakerASRService(settings, asr_service=asr_service)
    with pytest.raises(WorkbenchError, match="disabled"):
        combined.start_session()
    assert asr_service._sessions == {}


def test_diarization_streaming_config_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THAI_AI_DIARIZATION_STREAM_CHUNK_SECONDS", "2.4")
    monkeypatch.setenv("THAI_AI_DIARIZATION_STREAM_RIGHT_CONTEXT_SECONDS", "0.4")
    settings = load_settings()
    assert settings.diarization.stream_chunk_seconds == pytest.approx(2.4)
    assert settings.diarization.stream_right_context_seconds == pytest.approx(0.4)

    monkeypatch.setenv("THAI_AI_DIARIZATION_STREAM_CHUNK_SECONDS", "0.1")
    with pytest.raises(ConfigurationError, match="stream_chunk_seconds"):
        load_settings()


def test_cli_stream_asr_with_speakers(tmp_path, monkeypatch, capsys):
    import pythaiasr

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("THAI_AI_ASR_ENABLED", "true")
    monkeypatch.setenv("THAI_AI_DIARIZATION_ENABLED", "true")
    monkeypatch.setattr(pythaiasr, "RealtimeStreamASR", FakeRealtimeStreamASR)
    monkeypatch.setattr(StreamingASRService, "_load_model", lambda self: object())
    engine = FakeNemotronEngine()
    monkeypatch.setattr(
        StreamingDiarizationService,
        "_load_engine",
        lambda self: (engine, "cpu"),
    )
    monkeypatch.setattr(
        StreamingDiarizationService,
        "_create_cache",
        lambda self, eng: FakeSpeakerCache(),
    )
    wav_path = tmp_path / "two_speakers.wav"
    sf.write(wav_path, np.zeros(16000 * 4, dtype=np.float32), 16000)

    assert main(["stream-asr", str(wav_path), "--with-speakers"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    summary = next(item for item in lines if "attributed_events" in item)
    assert [turn["speaker"] for turn in summary["speaker_turns"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert summary["session_metrics"]["attributed_events"] >= 2
    assert "[SPEAKER_00]" in summary["formatted_transcript"]
