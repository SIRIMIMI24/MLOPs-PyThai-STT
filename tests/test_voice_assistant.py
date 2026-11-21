"""Phase 6 — Automated tests for 100% offline LocalVoiceAssistantService and CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from thai_offline_ai.adapters.tts import TTSResult
from thai_offline_ai.asr_types import ASRResult
from thai_offline_ai.cli import main as cli_main
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import ChatMessage, LLMResult
from thai_offline_ai.voice_assistant import LocalVoiceAssistantService, prepare_speech_text


@dataclass
class _StubASR:
    transcript: str = "สวัสดีครับ ช่วยแนะนำการเขียน Python หน่อย"
    calls: int = 0

    def transcribe_file(self, audio_path: str, *, model=None, prompt_mode=None) -> ASRResult:
        self.calls += 1
        return ASRResult(
            text=self.transcript,
            segments=[],
            duration=2.4,
            backend="openthai_onnx",
            model=model or "openthai2_asr",
        )


@dataclass
class _StubLLM:
    replies: list[str] = field(
        default_factory=lambda: ["สวัสดีครับ เริ่มจากติดตั้ง Python 3.10 แล้วลองเขียนฟังก์ชันสั้นๆ ก่อนครับ"]
    )
    seen_messages: list[list[ChatMessage]] = field(default_factory=list)

    def chat(self, messages) -> LLMResult:
        self.seen_messages.append(list(messages))
        text = self.replies.pop(0) if self.replies else "ครับผม"
        return LLMResult(text=text, backend="ollama", model="typhoon2.5-qwen3-4b")


@dataclass
class _StubTTS:
    output_dir: object
    fail: bool = False
    spoken: list[str] = field(default_factory=list)

    def synthesize(self, text: str, *, model=None, voice=None, speed=None) -> TTSResult:
        if self.fail:
            raise WorkbenchError("TTS model cache missing")
        self.spoken.append(text)
        wav = self.output_dir / f"va_{len(self.spoken)}.wav"
        wav.write_bytes(b"RIFF")
        return TTSResult(
            audio_path=str(wav),
            sample_rate=24000,
            backend="kokoro_onnx",
            model=model or "kokoro_thai_onnx",
            voice=voice or "f_young_bright",
            speed=speed or 1.05,
        )


def test_phase6_voice_assistant_end_to_end_and_session_continuity(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.asr.enabled = True
    settings.llm.enabled = True
    settings.tts.enabled = True

    asr = _StubASR()
    llm = _StubLLM(
        replies=[
            "เริ่มต้นด้วย **Python 3.10** และดูคู่มือที่ https://docs.python.org ได้เลยครับ",
            "ต่อจากเมื่อกี้ แนะนำให้ใช้ `pytest` สำหรับทดสอบโค้ดครับ",
        ]
    )
    tts = _StubTTS(tmp_path)
    svc = LocalVoiceAssistantService(
        settings,
        asr_adapter=asr,  # type: ignore[arg-type]
        tts_adapter=tts,  # type: ignore[arg-type]
        llm_client=llm,
    )

    t1 = svc.process_audio_turn("s1", str(tmp_path / "q1.wav"))
    assert asr.calls == 1
    assert t1.transcript == "สวัสดีครับ ช่วยแนะนำการเขียน Python หน่อย"
    assert t1.audio_path is not None
    assert t1.tts_error is None
    assert "https://" not in t1.spoken_text
    assert t1.metrics.asr_seconds >= 0.0
    assert t1.metrics.llm_seconds >= 0.0
    assert t1.metrics.tts_seconds >= 0.0
    assert t1.metrics.total_seconds >= 0.0
    assert t1.metrics.audio_duration_seconds == 2.4
    assert len(t1.history) == 2

    # Second turn reuses session history automatically
    t2 = svc.process_text_turn("s1", "แล้วควรใช้อะไรเทสต์โค้ดดี")
    assert len(t2.history) == 4
    assert any("Python 3.10" in msg.content for msg in llm.seen_messages[1])

    svc.clear_session("s1")
    assert svc.get_history("s1") == []


def test_phase6_voice_assistant_degrades_gracefully_when_tts_fails_or_disabled(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.tts.enabled = True
    svc = LocalVoiceAssistantService(
        settings,
        asr_adapter=_StubASR(),  # type: ignore[arg-type]
        tts_adapter=_StubTTS(tmp_path, fail=True),  # type: ignore[arg-type]
        llm_client=_StubLLM(),
    )
    turn = svc.process_audio_turn("s-err", str(tmp_path / "q.wav"))
    assert turn.response_text
    assert turn.audio_path is None
    assert turn.tts_error == "TTS model cache missing"

    # Validation checks
    with pytest.raises(WorkbenchError, match="บันทึกเสียง"):
        svc.process_audio_turn("s-err", None)
    assert prepare_speech_text("Hello [World](https://example.com) **ok**!") == "Hello World ok!"


def test_phase6_voice_assistant_cli_outputs_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF")

    monkeypatch.setattr(
        "thai_offline_ai.adapters.asr.ASRAdapter.transcribe_file",
        lambda self, audio, *, model=None, prompt_mode=None: ASRResult(
            text="ทดสอบเสียง", segments=[], duration=1.2, backend="openthai_onnx", model="openthai2_asr"
        ),
    )
    monkeypatch.setattr(
        "thai_offline_ai.adapters.llm.OllamaLocalLLM.chat",
        lambda self, messages: LLMResult(text="รับทราบครับ", backend="ollama", model=self.model),
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "app:\n  output_dir: outputs\nasr:\n  enabled: true\nllm:\n  enabled: true\ntts:\n  enabled: false\n",
        encoding="utf-8",
    )

    rc = cli_main(["--config", str(cfg_path), "voice-assistant", str(wav), "--no-tts"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript"] == "ทดสอบเสียง"
    assert payload["response_text"] == "รับทราบครับ"
    assert payload["metrics"]["asr_model"] == "openthai2_asr"
