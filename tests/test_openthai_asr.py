import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.adapters.openthai_asr import (
    OPENTHAI_MODEL_NAME,
    OpenThai2ONNXModel,
    OpenThai2RealtimeStreamASR,
    extract_log_mel_128,
    is_openthai_model,
    openthai_model_is_available,
    resolve_available_precision,
    split_audio_at_pauses,
)
from thai_offline_ai.asr_types import ASRResult, SpeechSegment
from thai_offline_ai.cli import main
from thai_offline_ai.config import AppConfig, ConfigurationError, Settings, load_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.streaming import StreamingASRService


class FakeEncoderSession:
    def __init__(self) -> None:
        self.seen_prompts: list[int] = []

    def get_inputs(self):
        return [
            SimpleNamespace(name="audio_signal", type="tensor(float)", shape=[1, 128, "T"]),
            SimpleNamespace(name="length", type="tensor(int64)", shape=[1]),
            SimpleNamespace(name="prompt_index", type="tensor(int64)", shape=[1]),
        ]

    def run(self, _outputs, feeds):
        self.seen_prompts.append(int(feeds["prompt_index"][0]))
        seq_len = int(feeds["length"][0])
        enc_len = max(1, seq_len // 8)
        encoded = np.zeros((1, enc_len, 1024), dtype=np.float32)
        return [encoded, np.array([enc_len], dtype=np.int64)]


class FakeDecoderSession:
    """Emits token 1 (<|th|>), token 2 (▁สวัสดี), token 3 (ครับ) then blank (4)."""

    def __init__(self, vocab_size: int = 4) -> None:
        self.vocab_size = vocab_size
        self.calls = 0

    def get_inputs(self):
        return [
            SimpleNamespace(name="encoder_outputs", type="tensor(float)", shape=[1, 1024, 1]),
            SimpleNamespace(name="targets", type="tensor(int32)", shape=[1, 1]),
            SimpleNamespace(name="target_length", type="tensor(int32)", shape=[1]),
            SimpleNamespace(name="input_states_1", type="tensor(float)", shape=[2, 1, 640]),
            SimpleNamespace(name="input_states_2", type="tensor(float)", shape=[2, 1, 640]),
        ]

    def run(self, _outputs, feeds):
        logits = np.full((1, 1, 1, self.vocab_size + 1), -5.0, dtype=np.float32)
        if self.calls == 0:
            logits[0, 0, 0, 1] = 2.0  # <|th|> tag (stripped)
        elif self.calls == 1:
            logits[0, 0, 0, 2] = 2.0  # ▁สวัสดี
        elif self.calls == 2:
            logits[0, 0, 0, 3] = 2.0  # ครับ
        else:
            # Blank logit 3.0 stays winning even after -2.5 short-clip bias (0.5 > -5.0).
            logits[0, 0, 0, self.vocab_size] = 3.0
        self.calls += 1
        return [logits, np.array([1], dtype=np.int32), feeds["input_states_1"], feeds["input_states_2"]]


def _make_fake_openthai_dir(tmp_path, precision: str = "int8"):
    model_dir = tmp_path / "openthai"
    model_dir.mkdir()
    enc = "encoder.int8.onnx" if precision == "int8" else "encoder.onnx"
    for name in (enc, f"{enc}.data", "decoder_joint.onnx"):
        (model_dir / name).write_bytes(b"fake-onnx")
    (model_dir / "tokens.txt").write_text("<unk>\n<th-TH>\n\u2581สวัสดี\nครับ\n", encoding="utf-8")
    return model_dir


def test_openthai_model_detection_and_cache_check(tmp_path):
    assert is_openthai_model("openthai2_asr")
    assert is_openthai_model("iapp/OpenThai2.0-ASR-Streaming-ONNX")
    assert is_openthai_model("typhoon_asr", backend="openthai_onnx")
    assert not is_openthai_model("typhoon_asr", backend="pythaiasr")

    model_dir = _make_fake_openthai_dir(tmp_path, precision="int8")
    assert openthai_model_is_available(model_dir, "int8")
    assert not openthai_model_is_available(model_dir, "fp32")
    assert resolve_available_precision(model_dir, "fp32") == "int8"


def test_extract_log_mel_128_and_pause_splitter():
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.05
    mel, seq_len = extract_log_mel_128(audio, sample_rate=16000)
    assert mel.shape == (1, 128, seq_len)
    assert seq_len == 101
    assert np.isfinite(mel).all()

    empty_mel, empty_len = extract_log_mel_128(np.array([], dtype=np.float32))
    assert empty_len == 0 and empty_mel.shape == (1, 128, 0)

    # 25s signal with an explicit silent pause around 12s should split near 12s.
    long_audio = np.ones(16000 * 25, dtype=np.float32) * 0.2
    long_audio[16000 * 11 : 16000 * 13] = 0.0
    pieces = split_audio_at_pauses(long_audio, sample_rate=16000, max_chunk_seconds=20.0)
    assert len(pieces) == 2
    assert 11.0 <= pieces[1][0] <= 13.0


def test_openthai_onnx_model_transcribe_prompt_and_streaming(tmp_path, monkeypatch):
    model_dir = _make_fake_openthai_dir(tmp_path, precision="int8")
    enc_session = FakeEncoderSession()
    dec_session = FakeDecoderSession(vocab_size=4)

    class FakeORT:
        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = 99

        class SessionOptions:
            graph_optimization_level = 0
            log_severity_level = 3

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            return enc_session if "encoder" in str(path) else dec_session

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeORT)
    model = OpenThai2ONNXModel(model_dir, precision="int8", device="cpu")
    result = model.transcribe(np.zeros(16000, dtype=np.float32), prompt_mode="th-mix")
    assert enc_session.seen_prompts == [63]
    assert result["text"] == "สวัสดีครับ"
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["start"] == 0.0
    assert result["chunks"][0]["end"] > 0.0

    # Verify streaming wrapper contract (process_chunk + flush).
    dec_session.calls = 0
    streamer = OpenThai2RealtimeStreamASR(model, step_sec=0.5, left_context_sec=0.2, right_context_sec=0.2, prompt_mode="th-TH")
    pieces = streamer.process_chunk(np.zeros(16000, dtype=np.float32))
    trailing = streamer.flush()
    assert enc_session.seen_prompts[-1] == 32
    assert "สวัสดีครับ" in " ".join([*pieces, trailing])


def test_asr_adapter_and_streaming_service_select_openthai_model(tmp_path):
    wav_path = tmp_path / "sample.wav"
    sf.write(wav_path, np.zeros(16000, dtype=np.float32), 16000)

    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "out")))
    settings.asr.enabled = True
    settings.asr.openthai_model_dir = str(tmp_path / "missing-openthai")

    adapter = ASRAdapter(settings)
    with pytest.raises(WorkbenchError, match="setup_openthai_asr_model.py"):
        adapter.transcribe_file(wav_path, model="openthai2_asr")

    class FakeModel:
        precision = "int8"

        def transcribe(self, audio, sample_rate=16000, *, prompt_mode="th-TH", return_timestamps=True):
            return {
                "text": f"ทดสอบ {prompt_mode}",
                "chunks": [{"text": f"ทดสอบ {prompt_mode}", "start": 0.0, "end": 0.8}],
            }

    adapter._openthai_model = FakeModel()
    result = adapter.transcribe_file(wav_path, model="openthai2_asr", prompt_mode="th-mix")
    assert result.backend == "openthai_onnx"
    assert result.model == OPENTHAI_MODEL_NAME
    assert result.text == "ทดสอบ th-mix"
    assert result.metadata["openthai_prompt"] == "th-mix"
    assert result.segments[0].end == 0.8

    # StreamingASRService rejects missing OpenThai cache cleanly without downloading.
    stream_svc = StreamingASRService(settings)
    with pytest.raises(WorkbenchError, match="setup_openthai_asr_model.py"):
        stream_svc.start_session(model="openthai2_asr")


def test_openthai_config_and_cli_override(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("THAI_AI_ASR_ENABLED", "true")
    monkeypatch.setenv("THAI_AI_ASR_MODEL", "openthai2_asr")
    monkeypatch.setenv("THAI_AI_ASR_OPENTHAI_PROMPT", "th-mix")
    monkeypatch.setenv("THAI_AI_ASR_OPENTHAI_PRECISION", "fp32")
    settings = load_settings()
    assert settings.asr.model == "openthai2_asr"
    assert settings.asr.openthai_prompt == "th-mix"
    assert settings.asr.openthai_precision == "fp32"

    monkeypatch.setenv("THAI_AI_ASR_OPENTHAI_PROMPT", "invalid")
    with pytest.raises(ConfigurationError, match="openthai_prompt"):
        load_settings()

    monkeypatch.delenv("THAI_AI_ASR_OPENTHAI_PROMPT")
    monkeypatch.delenv("THAI_AI_ASR_OPENTHAI_PRECISION")
    monkeypatch.setattr(
        "thai_offline_ai.cli.ASRAdapter.transcribe_file",
        lambda self, path: ASRResult(
            text="สวัสดี OpenThai",
            segments=[SpeechSegment("สวัสดี OpenThai", 0.0, 1.0)],
            backend="openthai_onnx",
            model=self.settings.asr.model,
            metadata={"openthai_prompt": self.settings.asr.openthai_prompt},
        ),
    )
    assert main(["asr", "sample.wav", "--model", "openthai2_asr", "--prompt", "th-mix"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "openthai_onnx"
    assert payload["model"] == "openthai2_asr"
    assert payload["metadata"]["openthai_prompt"] == "th-mix"
