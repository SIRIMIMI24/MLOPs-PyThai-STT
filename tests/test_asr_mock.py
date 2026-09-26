import sys
from types import ModuleType

from thai_offline_ai.adapters.asr import ASRAdapter


def test_asr_adapter_with_fake_module(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    fake = ModuleType("pythaiasr")
    fake.asr = lambda path, **kwargs: "ทดสอบถอดเสียง"
    monkeypatch.setitem(sys.modules, "pythaiasr", fake)

    result = ASRAdapter().transcribe_file(audio)
    assert result == "ทดสอบถอดเสียง"
