import numpy as np
import pytest

from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.streaming import StreamingASREvent, StreamingASRService


class FakeRealtimeStreamASR:
    def __init__(self, model, sample_rate, step_sec, silence_threshold):
        self.model = model
        self.sample_rate = sample_rate
        self.step_sec = step_sec
        self.silence_threshold = silence_threshold
        self.chunks = []

    def process_chunk(self, chunk):
        self.chunks.append(chunk)
        return ["ส่วนแรก"] if len(self.chunks) == 1 else []

    def flush(self):
        return "ส่วนสุดท้าย"


class StringRealtimeStreamASR(FakeRealtimeStreamASR):
    def process_chunk(self, chunk):
        self.chunks.append(chunk)
        return "ข้อความเดียว"


def enabled_settings(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "out")))
    settings.asr.enabled = True
    return settings


def fake_service(tmp_path, monkeypatch):
    import pythaiasr

    monkeypatch.setattr(pythaiasr, "RealtimeStreamASR", FakeRealtimeStreamASR)
    service = StreamingASRService(enabled_settings(tmp_path))
    service._model = object()
    return service


def test_streaming_event_is_serializable_and_defaults_are_independent():
    first = StreamingASREvent("one", 1, "ข้อความ", True)
    second = StreamingASREvent("two", 1, "อื่น", False)
    assert first.speaker is None and first.confidence is None
    assert first.metadata is not second.metadata
    assert first.to_dict()["is_final"] is True


def test_start_session_prepares_model_before_audio_capture(tmp_path, monkeypatch):
    service = StreamingASRService(enabled_settings(tmp_path))
    import pythaiasr

    monkeypatch.setattr(pythaiasr, "RealtimeStreamASR", FakeRealtimeStreamASR)
    model = object()
    monkeypatch.setattr(service, "_load_model", lambda: model)
    session_id = service.start_session()
    session = service._get_session(session_id)
    assert session.decoder.model is model
    assert session.decoder.silence_threshold == 0.0
    assert service.get_metrics(session_id).state == "listening"


def test_streaming_resamples_and_finalizes_in_order(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    session = service.start_session()
    events = service.push_audio(session, np.ones(22050, dtype=np.float32), 22050)
    finished = service.finalize(session)
    assert [event.sequence for event in events + finished] == [1, 2]
    assert all(event.is_final for event in events + finished)
    decoder = service._get_session(session).decoder
    assert decoder.sample_rate == 16000
    assert len(np.concatenate(decoder.chunks)) == 16000
    assert [event.text for event in service.get_events(session)] == ["ส่วนแรก", "ส่วนสุดท้าย"]


def test_startup_buffer_preserves_immediate_irregular_chunks(tmp_path, monkeypatch):
    """The first speech samples must survive browser-sized irregular chunks."""
    service = fake_service(tmp_path, monkeypatch)
    session = service.start_session()
    chunks = [
        np.arange(3200, dtype=np.float32),
        np.arange(3200, 8800, dtype=np.float32),
        np.arange(8800, 16000, dtype=np.float32),
        np.arange(16000, 32000, dtype=np.float32),
    ]
    for chunk in chunks:
        service.push_audio(session, chunk, 16000)
    decoder = service._get_session(session).decoder
    decoded = np.concatenate(decoder.chunks)
    expected = np.concatenate(chunks)
    assert np.array_equal(decoded, expected)
    metrics = service.get_metrics(session)
    assert metrics.first_audio_to_decoder_seconds is not None
    assert metrics.startup_buffered_audio_seconds == 0


def test_finalise_flushes_a_short_first_chunk_without_discarding_it(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    session = service.start_session()
    first_chunk = np.arange(3200, dtype=np.float32)
    assert service.push_audio(session, first_chunk, 16000) == []
    events = service.finalize(session)
    decoder = service._get_session(session).decoder
    assert np.array_equal(decoder.chunks[0], first_chunk)
    assert [event.text for event in events] == ["ส่วนแรก", "ส่วนสุดท้าย"]


def test_streaming_resampler_keeps_the_first_input_sample(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    samples = np.zeros(22050, dtype=np.float32)
    samples[0] = 0.75
    normalised = service._normalise_chunk(samples, 22050)
    assert len(normalised) == 16000
    assert normalised[0] == pytest.approx(0.75)


def test_streaming_records_local_latency_and_session_rtf(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    session = service.start_session()
    event = service.push_audio(session, np.ones(16000), 16000)[0]
    timing = event.metadata["timing"]
    assert timing["audio_seconds"] == 0.75
    assert timing["inference_seconds"] >= 0
    assert timing["chunk_rtf"] >= 0
    assert timing["chunk_received_at_monotonic_s"] <= timing["event_emitted_at_monotonic_s"]
    service.finalize(session)
    metrics = service.get_metrics(session)
    assert metrics.audio_seconds_processed == 1
    assert metrics.session_rtf is not None
    assert metrics.state == "stopped"


def test_streaming_sessions_are_isolated(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    left, right = service.start_session(), service.start_session()
    service.push_audio(left, np.ones(12000), 16000)
    service.push_audio(right, np.ones(12000), 16000)
    assert service.get_events(left)[0].session_id == left
    assert service.get_events(right)[0].session_id == right
    service.close_session(left)
    with pytest.raises(WorkbenchError, match="not found"):
        service.get_events(left)


def test_streaming_string_result_is_one_event(tmp_path, monkeypatch):
    import pythaiasr

    monkeypatch.setattr(pythaiasr, "RealtimeStreamASR", StringRealtimeStreamASR)
    service = StreamingASRService(enabled_settings(tmp_path))
    service._model = object()
    events = service.push_audio(service.start_session(), np.ones(12000), 16000)
    assert [event.text for event in events] == ["ข้อความเดียว"]


@pytest.mark.parametrize("samples,rate", [(np.array([], dtype=np.float32), 16000), (np.ones(2), 0)])
def test_streaming_rejects_invalid_audio(tmp_path, monkeypatch, samples, rate):
    service = fake_service(tmp_path, monkeypatch)
    with pytest.raises(WorkbenchError, match="audio"):
        service.push_audio(service.start_session(), samples, rate)


def test_streaming_closed_session_cannot_accept_more_audio(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    session = service.start_session()
    service.finalize(session)
    with pytest.raises(WorkbenchError, match="already finalised"):
        service.push_audio(session, np.ones(10), 16000)
