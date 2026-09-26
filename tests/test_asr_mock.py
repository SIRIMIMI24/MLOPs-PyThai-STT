from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.asr_types import ASRResult, SpeechSegment
from thai_offline_ai.config import AppConfig, Settings


def test_asr_result_contract_has_independent_defaults():
    first, second = ASRResult(text="a"), ASRResult(text="b")
    assert first.segments is not second.segments
    assert SpeechSegment(text="x").speaker is None


def test_asr_adapter_with_fake_backend(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    import numpy as np
    import soundfile as sf
    sf.write(audio, np.zeros(160), 16000)
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "out")))
    settings.asr.enabled = True
    settings.asr.model_dir = str(tmp_path / "models")
    adapter = ASRAdapter(settings)
    adapter._backend = lambda path, **kwargs: "ทดสอบถอดเสียง"
    result = adapter.transcribe_file(audio)
    assert result.text == "ทดสอบถอดเสียง"
    assert result.segments[0].start is None
