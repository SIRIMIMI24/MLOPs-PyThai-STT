import wave
from pathlib import Path

import pytest

from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError


class FakeTTS:
    def tts(self, text, **kwargs):
        path = Path(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"\x00\x00" * 100)
        return str(path)


def settings_for(tmp_path, enabled=True):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    settings.tts.enabled = enabled
    settings.tts.model_dir = str(tmp_path / "models")
    return settings


def test_tts_adapter_with_fake_model(tmp_path):
    adapter = TTSAdapter(settings_for(tmp_path))
    adapter._model = FakeTTS()
    result = adapter.synthesize("สวัสดี")
    assert Path(result.audio_path).exists()
    assert Path(result.audio_path).parent.name == "tts"
    assert result.sample_rate == 22050


def test_tts_disabled_does_not_load(tmp_path):
    with pytest.raises(WorkbenchError, match="disabled"):
        TTSAdapter(settings_for(tmp_path, enabled=False)).synthesize("สวัสดี")


def test_missing_tts_model_is_actionable(tmp_path):
    with pytest.raises(WorkbenchError, match="setup_tts_model"):
        TTSAdapter(settings_for(tmp_path)).synthesize("สวัสดี")
