"""Phase 8.3 — Push-to-talk Voice Pipeline reusing shared ASRAdapter and TTSAdapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.agent.orchestrator import AgentOrchestrator
from thai_offline_ai.agent.schemas import AgentResponse, OperatingMode
from thai_offline_ai.errors import WorkbenchError

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_MD_SYMBOL_RE = re.compile(r"[*_`#>]+")


class WakeWordHook(Protocol):
    """Optional future wake-word interface (disabled by default in push-to-talk mode)."""

    def is_triggered(self, audio_chunk: Any, sample_rate: int) -> bool: ...


@dataclass(slots=True)
class VoiceTurnResult:
    transcript: str
    agent_response: AgentResponse
    audio_path: str | None = None
    tts_error: str | None = None


def prepare_speech_text(answer: str, max_chars: int = 600) -> str:
    """Convert Markdown/URL-heavy answers into clean spoken Thai/English sentences for TTS."""
    text = _MD_LINK_RE.sub(r"\1", answer or "")
    text = _URL_RE.sub("", text)
    text = _MD_SYMBOL_RE.sub("", text)
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    joined = " ".join(lines)
    joined = re.sub(r"\s{2,}", " ", joined).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return joined


class JarvisVoicePipeline:
    """End-to-end Push-to-Talk pipeline: Audio -> ASR -> Orchestrator -> TTS."""

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        asr_adapter: ASRAdapter,
        tts_adapter: TTSAdapter,
        *,
        max_tts_chars: int = 600,
        wake_word_hook: WakeWordHook | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.asr_adapter = asr_adapter
        self.tts_adapter = tts_adapter
        self.max_tts_chars = max_tts_chars
        self.wake_word_hook = wake_word_hook

    def speak_answer(
        self,
        answer_text: str,
        *,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        tts_speed: float | None = None,
    ) -> tuple[str | None, str | None]:
        """Synthesize spoken audio via the shared TTSAdapter without loading duplicate models."""
        spoken = prepare_speech_text(answer_text, max_chars=self.max_tts_chars)
        if not spoken:
            return None, None
        try:
            res = self.tts_adapter.synthesize(
                spoken,
                model=tts_model,
                voice=tts_voice,
                speed=tts_speed,
            )
            return res.audio_path, None
        except WorkbenchError as exc:
            return None, str(exc)

    def process_voice_command(
        self,
        session_id: str,
        audio_path: str | None,
        *,
        mode: OperatingMode | None = None,
        speak_response: bool = True,
        asr_model: str | None = None,
        asr_prompt: str | None = None,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        tts_speed: float | None = None,
    ) -> VoiceTurnResult:
        if not audio_path:
            raise WorkbenchError("กรุณากดบันทึกเสียงผ่านไมโครโฟนหรือเลือกไฟล์เสียงก่อนสั่ง JARVIS")
        asr_result = self.asr_adapter.transcribe_file(
            audio_path,
            model=asr_model,
            prompt_mode=asr_prompt,
        )
        transcript = (asr_result.text or "").strip()
        if not transcript:
            raise WorkbenchError("ไม่สามารถถอดเสียงเป็นข้อความได้ ลองพูดให้ชัดขึ้นอีกครั้งครับ")

        agent_response = self.orchestrator.run_turn(session_id, transcript, mode=mode)
        reply_audio: str | None = None
        tts_error: str | None = None
        if speak_response and agent_response.answer:
            reply_audio, tts_error = self.speak_answer(
                agent_response.answer,
                tts_model=tts_model,
                tts_voice=tts_voice,
                tts_speed=tts_speed,
            )
        return VoiceTurnResult(
            transcript=transcript,
            agent_response=agent_response,
            audio_path=reply_audio,
            tts_error=tts_error,
        )
