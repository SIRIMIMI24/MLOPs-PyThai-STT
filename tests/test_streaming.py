import numpy as np
import pytest

from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.streaming import StreamingASREvent, StreamingASRService


class FakeRealtimeStreamASR:
    def __init__(self, model, sample_rate, step_sec):
        self.model = model
        self.sample_rate = sample_rate
        self.step_sec = step_sec
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


def test_start_session_does_not_load_model(tmp_path, monkeypatch):
    service = StreamingASRService(enabled_settings(tmp_path))
    monkeypatch.setattr(service, "_load_model", lambda: (_ for _ in ()).throw(AssertionError()))
    assert service.start_session()


def test_streaming_resamples_and_finalizes_in_order(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    session = service.start_session()
    events = service.push_audio(session, np.ones(22050, dtype=np.float32), 22050)
    finished = service.finalize(session)
    assert [event.sequence for event in events + finished] == [1, 2]
    assert all(event.is_final for event in events + finished)
    decoder = service._get_session(session).decoder
    assert decoder.sample_rate == 16000
    assert len(decoder.chunks[0]) == 16000
    assert [event.text for event in service.get_events(session)] == ["ส่วนแรก", "ส่วนสุดท้าย"]


def test_streaming_sessions_are_isolated(tmp_path, monkeypatch):
    service = fake_service(tmp_path, monkeypatch)
    left, right = service.start_session(), service.start_session()
    service.push_audio(left, np.ones(160), 16000)
    service.push_audio(right, np.ones(160), 16000)
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
    events = service.push_audio(service.start_session(), np.ones(160), 16000)
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
