"""Offline CPU Thai text-to-speech supporting PyThaiTTS Lunarlist and Kokoro Thai ONNX."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from thai_offline_ai.adapters.kokoro_tts import (
    DEFAULT_KOKORO_VOICE,
    KOKORO_MODEL_NAME,
    KokoroThaiONNXModel,
    is_kokoro_model,
    kokoro_model_is_available,
)
from thai_offline_ai.config import Settings, TTSConfig
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import output_artifact_path

LUNARLIST_ONNX_FILES = (
    "tacotron2encoder-th.onnx",
    "tacotron2decoder-th.onnx",
    "tacotron2postnet-th.onnx",
    "vocoder.onnx",
)


@dataclass(frozen=True, slots=True)
class TTSResult:
    """A locally generated WAV and metadata that the UI and CLI can safely expose."""

    audio_path: str
    sample_rate: int
    backend: str
    model: str
    voice: str | None = None
    speed: float | None = None


def model_is_available(model_dir: str | Path) -> bool:
    """Check only local cache files; never import a model or contact a model host."""
    root = Path(model_dir)
    return root.is_dir() and all(any(root.rglob(name)) for name in LUNARLIST_ONNX_FILES)


class TTSAdapter:
    """Lazy, cache-only CPU Thai TTS adapter supporting Lunarlist & Kokoro Thai ONNX."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config: TTSConfig = settings.tts
        self._model: Any | None = None
        self._kokoro_model: KokoroThaiONNXModel | None = None
        self._lock = Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> TTSAdapter:
        return cls(settings)

    def _get_model(self) -> Any:
        if not self._config.enabled:
            raise WorkbenchError("TTS is disabled. Set tts.enabled=true in your configuration.")
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if not model_is_available(self._config.model_dir):
                raise WorkbenchError(
                    "TTS model is not available locally. Run: "
                    "python scripts/setup_tts_model.py (internet is required only for that setup step)."
                )
            try:
                from huggingface_hub import hf_hub_download
                from pythaitts import TTS
                from pythaitts.pretrained import lunarlist_onnx
            except (ImportError, OSError) as exc:
                raise WorkbenchError(
                    "PyThaiTTS or ONNX Runtime is not installed. Run: "
                    "python -m pip install -e \".[tts,dev]\""
                ) from exc

            # PyThaiTTS 0.4.2 calls hf_hub_download without cache/local flags. Its
            # verified ONNX module is patched at this narrow boundary so normal
            # inference is cache-only and uses the configured project-local cache.
            def local_model_file(*args: Any, **kwargs: Any) -> str:
                kwargs["cache_dir"] = self._config.model_dir
                kwargs["local_files_only"] = True
                return hf_hub_download(*args, **kwargs)

            lunarlist_onnx.hf_hub_download = local_model_file
            original_session = lunarlist_onnx.ort.InferenceSession

            def cpu_session(*args: Any, **kwargs: Any) -> Any:
                kwargs["providers"] = ["CPUExecutionProvider"]
                return original_session(*args, **kwargs)

            lunarlist_onnx.ort.InferenceSession = cpu_session
            try:
                self._model = TTS(pretrained="lunarlist_onnx", device="cpu")
            except Exception as exc:
                raise WorkbenchError(
                    "The local TTS model could not be loaded. Run python scripts/setup_tts_model.py "
                    "to populate the configured model cache, then run python scripts/doctor.py."
                ) from exc
        return self._model

    def _get_kokoro_model(self) -> KokoroThaiONNXModel:
        if not self._config.enabled:
            raise WorkbenchError("TTS is disabled. Set tts.enabled=true in your configuration.")
        if self._kokoro_model is not None:
            return self._kokoro_model
        with self._lock:
            if self._kokoro_model is not None:
                return self._kokoro_model
            if not kokoro_model_is_available(self._config.kokoro_model_dir):
                raise WorkbenchError(
                    "Kokoro Thai TTS model is not available locally. "
                    "Run: python scripts/setup_kokoro_tts_model.py"
                )
            try:
                self._kokoro_model = KokoroThaiONNXModel(self._config.kokoro_model_dir)
            except WorkbenchError:
                raise
            except Exception as exc:
                raise WorkbenchError(
                    "Kokoro Thai TTS could not initialize. Confirm the local ONNX cache."
                ) from exc
        return self._kokoro_model

    def synthesize(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
    ) -> TTSResult:
        """Create one unique WAV below ``outputs/tts`` without model downloads."""
        if not isinstance(text, str) or not text.strip():
            raise WorkbenchError("TTS input must contain text.")

        effective_model = model or self._config.model
        use_kokoro = is_kokoro_model(
            effective_model, None if model is not None else self._config.backend
        )

        filename = f"tts_{datetime.now().astimezone():%Y%m%d_%H%M%S}_{uuid4().hex[:10]}.wav"
        destination = output_artifact_path(self._settings, "tts", filename)

        if use_kokoro:
            import numpy as np

            effective_voice = voice or self._config.voice or DEFAULT_KOKORO_VOICE
            effective_speed = self._config.speed if speed is None else float(speed)
            engine = self._get_kokoro_model()
            try:
                with self._lock:
                    audio, sample_rate = engine.synthesize(
                        text.strip(), voice=effective_voice, speed=effective_speed
                    )
                pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
                with wave.open(str(destination), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(pcm16.tobytes())
            except WorkbenchError:
                raise
            except Exception as exc:
                raise WorkbenchError(
                    "Kokoro Thai speech synthesis failed. Confirm the local model cache and tltk installation."
                ) from exc
            return TTSResult(
                audio_path=str(destination),
                sample_rate=sample_rate,
                backend="kokoro_onnx",
                model=KOKORO_MODEL_NAME,
                voice=effective_voice,
                speed=effective_speed,
            )

        tts_model = self._get_model()
        try:
            returned = tts_model.tts(
                text.strip(), return_type="file", filename=str(destination), preprocess=True
            )
            generated = Path(returned or destination)
            if generated.resolve() != destination or not destination.is_file() or destination.stat().st_size == 0:
                raise OSError("backend did not create the requested WAV output")
            with wave.open(str(destination), "rb") as wav:
                sample_rate = wav.getframerate()
                if wav.getnframes() <= 0 or sample_rate <= 0:
                    raise OSError("generated WAV has no audio frames")
            if sample_rate != 22050:
                raise OSError(f"unexpected sample rate {sample_rate}")
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError(
                "Thai speech synthesis failed. Confirm the local TTS model with "
                "python scripts/doctor.py and try shorter Thai text."
            ) from exc
        return TTSResult(
            audio_path=str(destination),
            sample_rate=sample_rate,
            backend="pythaitts",
            model="lunarlist_onnx",
        )
