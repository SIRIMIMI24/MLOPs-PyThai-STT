import json
import wave

import numpy as np
import pytest

from thai_offline_ai.adapters.kokoro_tts import (
    DEFAULT_KOKORO_VOICE,
    KOKORO_MODEL_NAME,
    KOKORO_VOICES,
    KokoroThaiONNXModel,
    english_run_to_kokoro_phonemes,
    harmonic_source,
    is_kokoro_model,
    kokoro_model_is_available,
    normalize_thai_for_kokoro,
    thai_text_to_kokoro_phonemes,
)
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.cli import main
from thai_offline_ai.config import AppConfig, ConfigurationError, Settings, load_settings
from thai_offline_ai.errors import WorkbenchError


def test_kokoro_model_detection_and_voice_catalog(tmp_path):
    assert is_kokoro_model("kokoro_thai_onnx")
    assert is_kokoro_model("kunato/wayu-kokoro-thai-v1")
    assert is_kokoro_model("lunarlist_onnx", backend="kokoro_onnx")
    assert not is_kokoro_model("lunarlist_onnx", backend="pythaitts")
    assert len(KOKORO_VOICES) == 12
    assert any(v.voice_id == DEFAULT_KOKORO_VOICE for v in KOKORO_VOICES)

    model_dir = tmp_path / "kokoro"
    model_dir.mkdir()
    assert not kokoro_model_is_available(model_dir)
    for name in (
        "onnx_manifest.json",
        "prosody_fp32.onnx",
        "curves_fp32.onnx",
        "decoder_fp32.onnx",
        "source_params.npz",
        "styles.npz",
    ):
        (model_dir / name).write_bytes(b"ok")
    assert kokoro_model_is_available(model_dir)


def test_thai_normalization_and_kokoro_phonemizer():
    assert normalize_thai_for_kokoro("มากกก จ้าา") == "มาก จ้า"
    vocab = {
        "s": 1, "a": 2, "w": 3, "t": 4, "d": 5, "i": 6, "ɔ": 7, " ": 8,
        "→": 9, "˩": 10, "↘": 11, "ˈ": 12, "A": 13, "I": 14, "p": 15,
        "θ": 16, "ɑ": 17, "n": 18, "f": 19, "æ": 20, "k": 21, "ˌ": 22,
    }
    phonemes = thai_text_to_kokoro_phonemes(
        "สวัสดี AI Python Fastwork",
        vocab,
        th2ipa_fn=lambda chunk: "sa2.wat2.di1<s/>ᴐ3",
        en_lexicon={"python": "pˈIθɑn"},
        en_fallback=lambda tok: ("fˈæstwk", 2),
    )
    assert "˩" in phonemes and "→" in phonemes and "ɔ↘" in phonemes
    assert "ˈAˌI" in phonemes
    assert "pˈIθɑn" in phonemes
    assert "fˈæstwk" in phonemes
    assert "<s/>" not in phonemes and "." not in phonemes

    en_only = english_run_to_kokoro_phonemes(
        "AI Python", vocab, lexicon={"python": "pˈIθɑn"}, espeak_fallback=None
    )
    assert en_only == "ˈAˌI pˈIθɑn"


def test_harmonic_source_shape_and_finite_output():
    f0 = np.full((1, 4), 220.0, dtype=np.float32)
    weight = np.ones((1, 9), dtype=np.float64) * 0.1
    bias = np.zeros((1,), dtype=np.float64)
    window = np.hanning(20).astype(np.float32)
    har = harmonic_source(f0, weight, bias, window, seed=42)
    assert har.ndim == 3 and har.shape[0] == 1 and har.shape[1] == 22
    assert np.isfinite(har).all()


def test_tts_adapter_routes_to_kokoro_and_writes_24khz_wav(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "out")))
    settings.tts.enabled = True
    settings.tts.kokoro_model_dir = str(tmp_path / "missing-kokoro")

    adapter = TTSAdapter.from_settings(settings)
    with pytest.raises(WorkbenchError, match="setup_kokoro_tts_model.py"):
        adapter.synthesize("สวัสดีค่ะ", model="kokoro_thai_onnx")

    class FakeKokoro:
        def synthesize(self, text: str, *, voice: str, speed: float | None = None):
            assert voice == "f_young_bright"
            assert speed == pytest.approx(1.1)
            return np.zeros(2400, dtype=np.float32), 24000

    adapter._kokoro_model = FakeKokoro()
    result = adapter.synthesize(
        "สวัสดีค่ะ", model="kokoro_thai_onnx", voice="f_young_bright", speed=1.1
    )
    assert result.backend == "kokoro_onnx"
    assert result.model == KOKORO_MODEL_NAME
    assert result.sample_rate == 24000
    assert result.voice == "f_young_bright"
    assert result.speed == pytest.approx(1.1)
    with wave.open(result.audio_path, "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 2400


def test_kokoro_config_validation_and_cli_override(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("THAI_AI_TTS_ENABLED", "true")
    monkeypatch.setenv("THAI_AI_TTS_BACKEND", "kokoro_onnx")
    monkeypatch.setenv("THAI_AI_TTS_MODEL", "kokoro_thai_onnx")
    monkeypatch.setenv("THAI_AI_TTS_VOICE", "f_young_clear")
    monkeypatch.setenv("THAI_AI_TTS_SPEED", "1.15")
    settings = load_settings()
    assert settings.tts.model == "kokoro_thai_onnx"
    assert settings.tts.voice == "f_young_clear"
    assert settings.tts.speed == pytest.approx(1.15)

    monkeypatch.setenv("THAI_AI_TTS_SPEED", "2.5")
    with pytest.raises(ConfigurationError, match="tts.speed"):
        load_settings()

    monkeypatch.setenv("THAI_AI_TTS_SPEED", "1.05")
    monkeypatch.setattr(
        KokoroThaiONNXModel,
        "synthesize",
        lambda self, text, *, voice=DEFAULT_KOKORO_VOICE, speed=None: (
            np.zeros(1200, dtype=np.float32),
            24000,
        ),
    )
    monkeypatch.setattr(
        TTSAdapter,
        "_get_kokoro_model",
        lambda self: object.__new__(KokoroThaiONNXModel),
    )
    assert (
        main(["tts", "สวัสดีค่ะ", "--model", "kokoro_thai_onnx", "--voice", "f_young_bright", "--speed", "1.1"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "kokoro_onnx"
    assert payload["voice"] == "f_young_bright"
    assert payload["sample_rate"] == 24000
