"""Phase 6 — 100% Offline Local Voice Assistant (ASR -> Local LLM -> TTS).

Reuses shared ASRAdapter and TTSAdapter instances so models are never loaded twice,
measures real stage-by-stage latency, and keeps conversation history session-local.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import ChatMessage, LLMResult

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_MD_SYMBOL_RE = re.compile(r"[*_`#>]+")


class VoiceLLMClient(Protocol):
    """Minimal local chat client protocol matching OllamaLocalLLM."""

    def chat(self, messages: Sequence[ChatMessage]) -> LLMResult: ...


def prepare_speech_text(answer: str, max_chars: int = 600) -> str:
    """Convert Markdown/URL text into clean spoken Thai/English sentences for local TTS."""
    text = _MD_LINK_RE.sub(r"\1", answer or "")
    text = _URL_RE.sub("", text)
    text = _MD_SYMBOL_RE.sub("", text)
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    joined = " ".join(lines)
    joined = re.sub(r"\s{2,}", " ", joined).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return joined


@dataclass(frozen=True, slots=True)
class VoiceAssistantMetrics:
    """Measured wall-clock latency for each stage of a local voice turn."""

    asr_seconds: float
    llm_seconds: float
    tts_seconds: float
    total_seconds: float
    audio_duration_seconds: float
    asr_model: str
    llm_model: str
    tts_model: str | None = None
    tts_voice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asr_seconds": round(self.asr_seconds, 3),
            "llm_seconds": round(self.llm_seconds, 3),
            "tts_seconds": round(self.tts_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "audio_duration_seconds": round(self.audio_duration_seconds, 3),
            "asr_model": self.asr_model,
            "llm_model": self.llm_model,
            "tts_model": self.tts_model,
            "tts_voice": self.tts_voice,
        }


@dataclass(slots=True)
class VoiceAssistantTurnResult:
    """Result of one offline Local Voice Assistant turn."""

    session_id: str
    transcript: str
    response_text: str
    spoken_text: str
    audio_path: str | None
    metrics: VoiceAssistantMetrics
    history: list[dict[str, str]] = field(default_factory=list)
    tts_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "transcript": self.transcript,
            "response_text": self.response_text,
            "spoken_text": self.spoken_text,
            "audio_path": self.audio_path,
            "tts_error": self.tts_error,
            "metrics": self.metrics.to_dict(),
        }


class LocalVoiceAssistantService:
    """Direct offline voice-to-voice service reusing existing workbench adapters."""

    def __init__(
        self,
        settings: Settings,
        *,
        asr_adapter: ASRAdapter | None = None,
        tts_adapter: TTSAdapter | None = None,
        llm_client: VoiceLLMClient | None = None,
        max_history_messages: int = 20,
        max_tts_chars: int = 600,
    ) -> None:
        self.settings = settings
        self.asr_adapter = asr_adapter or ASRAdapter(settings)
        self.tts_adapter = tts_adapter or TTSAdapter.from_settings(settings)
        self._llm_client = llm_client
        self.max_history_messages = max(2, max_history_messages)
        self.max_tts_chars = max(100, max_tts_chars)
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = Lock()

    def _get_llm(self) -> VoiceLLMClient:
        if self._llm_client is not None:
            return self._llm_client
        from thai_offline_ai.adapters.llm import OllamaLocalLLM

        return OllamaLocalLLM.from_config(
            self.settings.llm, self.settings.app.offline_runtime
        )

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _build_messages(
        self,
        session_id: str,
        user_text: str,
        *,
        system_prompt: str | None,
        external_history: Sequence[dict[str, str]] | None,
    ) -> tuple[list[ChatMessage], list[dict[str, str]]]:
        if external_history is not None:
            base_history = [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in external_history
                if isinstance(item, dict) and item.get("role") and item.get("content")
            ]
        else:
            base_history = self.get_history(session_id)
        trimmed = base_history[-self.max_history_messages :]
        prompt = (system_prompt or self.settings.llm.system_prompt).strip()
        messages = [ChatMessage("system", prompt)]
        messages.extend(ChatMessage(item["role"], item["content"]) for item in trimmed)
        messages.append(ChatMessage("user", user_text))
        return messages, trimmed

    def _commit_turn(
        self,
        session_id: str,
        base_history: list[dict[str, str]],
        user_text: str,
        assistant_text: str,
    ) -> list[dict[str, str]]:
        updated = [
            *base_history,
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ][-self.max_history_messages :]
        with self._lock:
            self._sessions[session_id] = list(updated)
        return updated

    def process_audio_turn(
        self,
        session_id: str,
        audio_path: str | None,
        *,
        system_prompt: str | None = None,
        external_history: Sequence[dict[str, str]] | None = None,
        speak_response: bool = True,
        asr_model: str | None = None,
        asr_prompt: str | None = None,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        tts_speed: float | None = None,
    ) -> VoiceAssistantTurnResult:
        """Transcribe local audio, query local Ollama LLM, and optionally speak reply via TTS."""
        if not audio_path:
            raise WorkbenchError("กรุณาบันทึกเสียงจากไมโครโฟนหรือเลือกไฟล์เสียงก่อนคุยกับผู้ช่วยเสียง")

        total_started = time.perf_counter()

        # Stage 1: Local ASR
        asr_started = time.perf_counter()
        asr_result = self.asr_adapter.transcribe_file(
            audio_path,
            model=asr_model,
            prompt_mode=asr_prompt,
        )
        asr_elapsed = time.perf_counter() - asr_started
        transcript = (asr_result.text or "").strip()
        if not transcript:
            raise WorkbenchError("ไม่สามารถถอดเสียงเป็นข้อความได้ ลองพูดให้ชัดเจนขึ้นอีกครั้ง")

        return self._run_llm_and_tts(
            session_id=session_id,
            user_text=transcript,
            asr_elapsed=asr_elapsed,
            audio_duration=float(asr_result.duration or 0.0),
            asr_model_name=asr_result.model,
            total_started=total_started,
            system_prompt=system_prompt,
            external_history=external_history,
            speak_response=speak_response,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
        )

    def process_text_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        system_prompt: str | None = None,
        external_history: Sequence[dict[str, str]] | None = None,
        speak_response: bool = True,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        tts_speed: float | None = None,
    ) -> VoiceAssistantTurnResult:
        """Send text directly to Local LLM and optionally synthesize spoken reply via TTS."""
        cleaned = (user_text or "").strip()
        if not cleaned:
            raise WorkbenchError("พิมพ์ข้อความก่อนส่งไปยัง Local Voice Assistant")
        return self._run_llm_and_tts(
            session_id=session_id,
            user_text=cleaned,
            asr_elapsed=0.0,
            audio_duration=0.0,
            asr_model_name="text-input",
            total_started=time.perf_counter(),
            system_prompt=system_prompt,
            external_history=external_history,
            speak_response=speak_response,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
        )

    def _run_llm_and_tts(
        self,
        *,
        session_id: str,
        user_text: str,
        asr_elapsed: float,
        audio_duration: float,
        asr_model_name: str,
        total_started: float,
        system_prompt: str | None,
        external_history: Sequence[dict[str, str]] | None,
        speak_response: bool,
        tts_model: str | None,
        tts_voice: str | None,
        tts_speed: float | None,
    ) -> VoiceAssistantTurnResult:
        messages, base_history = self._build_messages(
            session_id,
            user_text,
            system_prompt=system_prompt,
            external_history=external_history,
        )

        # Stage 2: Local LLM (Ollama loopback)
        llm_started = time.perf_counter()
        llm_result = self._get_llm().chat(messages)
        llm_elapsed = time.perf_counter() - llm_started
        response_text = llm_result.text.strip()

        updated_history = self._commit_turn(session_id, base_history, user_text, response_text)
        spoken_text = prepare_speech_text(response_text, max_chars=self.max_tts_chars)

        # Stage 3: Local TTS (graceful degradation if disabled or unavailable)
        audio_path: str | None = None
        tts_error: str | None = None
        tts_elapsed = 0.0
        used_tts_model: str | None = None
        used_tts_voice: str | None = None

        if speak_response and spoken_text:
            if not self.settings.tts.enabled:
                tts_error = "TTS is disabled in settings (tts.enabled=false)."
            else:
                tts_started = time.perf_counter()
                try:
                    if tts_model is None and tts_voice is None and tts_speed is None:
                        tts_res = self.tts_adapter.synthesize(spoken_text)
                    else:
                        tts_res = self.tts_adapter.synthesize(
                            spoken_text,
                            model=tts_model,
                            voice=tts_voice,
                            speed=tts_speed,
                        )
                    audio_path = tts_res.audio_path
                    used_tts_model = tts_res.model
                    used_tts_voice = tts_res.voice
                except WorkbenchError as exc:
                    tts_error = str(exc)
                finally:
                    tts_elapsed = time.perf_counter() - tts_started

        total_elapsed = time.perf_counter() - total_started
        metrics = VoiceAssistantMetrics(
            asr_seconds=asr_elapsed,
            llm_seconds=llm_elapsed,
            tts_seconds=tts_elapsed,
            total_seconds=total_elapsed,
            audio_duration_seconds=audio_duration,
            asr_model=asr_model_name,
            llm_model=llm_result.model,
            tts_model=used_tts_model,
            tts_voice=used_tts_voice,
        )
        return VoiceAssistantTurnResult(
            session_id=session_id,
            transcript=user_text,
            response_text=response_text,
            spoken_text=spoken_text,
            audio_path=audio_path,
            metrics=metrics,
            history=updated_history,
            tts_error=tts_error,
        )
