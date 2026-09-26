from pathlib import Path

from thai_offline_ai.adapters.tts import TTSAdapter


class FakeTTS:
    def tts(self, text, **kwargs):
        path = Path(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
        return str(path)


def test_tts_adapter_with_fake_model(tmp_path):
    adapter = TTSAdapter()
    adapter._model = FakeTTS()
    output = adapter.synthesize("สวัสดี", tmp_path / "out.wav")
    assert Path(output).exists()
