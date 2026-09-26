from __future__ import annotations

import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

from thai_offline_ai.asr_types import ASRResult, SpeechSegment
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_output_dir

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
        if path.suffix.lower() != ".wav" or not path.is_file() or path.stat().st_size == 0:
            raise WorkbenchError("ASR input must be a non-empty readable WAV file.")
        normalized: Path | None = None
        try:
            import numpy as np
            import soundfile as sf

            info = sf.info(str(path))
            if info.frames <= 0 or info.samplerate <= 0:
                raise RuntimeError("empty audio")
            duration = info.frames / info.samplerate
            samples, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
            if samples.size == 0:
                raise RuntimeError("empty audio")
            mono = samples.mean(axis=1)
            if source_rate != self.settings.asr.sampling_rate:
                target_count = max(1, round(len(mono) * self.settings.asr.sampling_rate / source_rate))
                source_axis = np.linspace(0, len(mono) - 1, len(mono), dtype=np.float64)
                target_axis = np.linspace(0, len(mono) - 1, target_count, dtype=np.float64)
                mono = np.interp(target_axis, source_axis, mono).astype("float32")
            directory = prepare_output_dir(self.settings) / "asr-tmp"
            directory.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".wav", delete=False) as handle:
                normalized = Path(handle.name)
            sf.write(str(normalized), mono, self.settings.asr.sampling_rate, subtype="PCM_16")
        except Exception as exc:
            raise WorkbenchError("ASR input is not a readable WAV file.") from exc
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
        return ASRResult(text=text, segments=segments, language="th", duration=duration, model=self.settings.asr.model)
