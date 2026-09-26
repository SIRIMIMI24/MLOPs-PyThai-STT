from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from thai_offline_ai.asr_types import ASRResult, SpeechSegment
from thai_offline_ai.audio import prepare_audio, write_temporary_wav
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError

REQUIRED_MODEL_FILES = ("encoder-fastconformer-quran-ar.onnx", "decoder_joint-fastconformer-quran-ar.onnx", "vocab.json")


def model_is_available(model_dir: str | Path) -> bool:
    root = Path(model_dir)
    return root.is_dir() and all(any(root.rglob(name)) for name in REQUIRED_MODEL_FILES)


class ASRAdapter:
    """Lazy cache-only PyThaiASR 2.2.0 file-transcription adapter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backend: Any | None = None
        self._lock = Lock()

    def _load(self) -> Any:
        config = self.settings.asr
        if not config.enabled:
            raise WorkbenchError("ASR is disabled. Set asr.enabled=true after local model setup.")
        if self._backend is not None:
            return self._backend
        with self._lock:
            if not model_is_available(config.model_dir):
                raise WorkbenchError("ASR model is not available locally. Run: python scripts/setup_asr_model.py")
            try:
                os.environ["PYTHAIASR_DATA_DIR"] = config.model_dir
                from pythaiasr import asr
            except (ImportError, OSError) as exc:
                raise WorkbenchError("PyThaiASR is missing. Run: python -m pip install -e \".[asr,dev]\"") from exc
            self._backend = asr
        return self._backend

    def transcribe_file(self, audio_path: str | Path) -> ASRResult:
        path = Path(audio_path)
        normalized: Path | None = None
        try:
            audio = prepare_audio(path, target_rate=self.settings.asr.sampling_rate)
            normalized = write_temporary_wav(audio, self.settings)
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("ASR input is not a readable supported audio file.") from exc
        try:
            raw = self._load()(str(normalized), model=self.settings.asr.model, device="cpu", sampling_rate=16000, return_timestamps=True)
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("Thai ASR failed. Confirm the local model cache and WAV input.") from exc
        finally:
            if normalized is not None:
                normalized.unlink(missing_ok=True)
        text = str(raw.get("text", "") if isinstance(raw, dict) else raw).strip()
        chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
        segments = [SpeechSegment(text=str(c.get("text", "")).strip(), start=c.get("start"), end=c.get("end")) for c in chunks if str(c.get("text", "")).strip()]
        if text and not segments:
            segments = [SpeechSegment(text=text)]
        return ASRResult(
            text=text,
            segments=segments,
            language="th",
            duration=audio.original.duration,
            model=self.settings.asr.model,
            metadata={
                "original_audio": audio.original.to_dict(),
                "normalized_audio": audio.normalized.to_dict(),
            },
        )
