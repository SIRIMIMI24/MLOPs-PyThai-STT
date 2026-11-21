from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from thai_offline_ai.adapters.openthai_asr import (
    OPENTHAI_MODEL_NAME,
    OpenThai2ONNXModel,
    is_openthai_model,
    resolve_available_precision,
)
from thai_offline_ai.asr_types import ASRResult, SpeechSegment
from thai_offline_ai.audio import prepare_audio, write_temporary_wav
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError

REQUIRED_MODEL_FILES = (
    "encoder-fastconformer-quran-ar.onnx",
    "decoder_joint-fastconformer-quran-ar.onnx",
    "vocab.json",
)


def model_is_available(model_dir: str | Path) -> bool:
    root = Path(model_dir)
    return root.is_dir() and all(any(root.rglob(name)) for name in REQUIRED_MODEL_FILES)


class ASRAdapter:
    """Lazy cache-only file-transcription adapter supporting Typhoon & OpenThai 2.0 ONNX."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backend: Any | None = None
        self._openthai_model: OpenThai2ONNXModel | None = None
        self._lock = Lock()

    def _load(self) -> Any:
        config = self.settings.asr
        if not config.enabled:
            raise WorkbenchError("ASR is disabled. Set asr.enabled=true after local model setup.")
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            if not model_is_available(config.model_dir):
                raise WorkbenchError(
                    "ASR model is not available locally. Run: python scripts/setup_asr_model.py"
                )
            try:
                os.environ["PYTHAIASR_DATA_DIR"] = config.model_dir
                from pythaiasr import asr
            except (ImportError, OSError) as exc:
                raise WorkbenchError(
                    'PyThaiASR is missing. Run: python -m pip install -e ".[asr,dev]"'
                ) from exc
            self._backend = asr
        return self._backend

    def _load_openthai(self) -> OpenThai2ONNXModel:
        config = self.settings.asr
        if not config.enabled:
            raise WorkbenchError("ASR is disabled. Set asr.enabled=true after local model setup.")
        if self._openthai_model is not None:
            return self._openthai_model
        with self._lock:
            if self._openthai_model is not None:
                return self._openthai_model
            precision = resolve_available_precision(
                config.openthai_model_dir, config.openthai_precision
            )
            if precision is None:
                raise WorkbenchError(
                    "OpenThai2.0 ASR ONNX model is not available locally. "
                    f"Run: python scripts/setup_openthai_asr_model.py --precision {config.openthai_precision}"
                )
            try:
                self._openthai_model = OpenThai2ONNXModel(
                    config.openthai_model_dir,
                    precision=precision,
                    device=config.device,
                )
            except WorkbenchError:
                raise
            except Exception as exc:
                raise WorkbenchError(
                    "OpenThai2.0 ASR ONNX could not initialize. Confirm the local model cache."
                ) from exc
        return self._openthai_model

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        model: str | None = None,
        prompt_mode: str | None = None,
    ) -> ASRResult:
        path = Path(audio_path)
        effective_model = model or self.settings.asr.model
        use_openthai = is_openthai_model(
            effective_model, None if model is not None else self.settings.asr.backend
        )
        try:
            audio = prepare_audio(path, target_rate=self.settings.asr.sampling_rate)
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("ASR input is not a readable supported audio file.") from exc

        if use_openthai:
            effective_prompt = prompt_mode or self.settings.asr.openthai_prompt
            try:
                engine = self._load_openthai()
                with self._lock:
                    raw = engine.transcribe(
                        audio.samples,
                        sample_rate=audio.normalized.sample_rate,
                        prompt_mode=effective_prompt,
                        return_timestamps=True,
                    )
            except WorkbenchError:
                raise
            except Exception as exc:
                raise WorkbenchError(
                    "OpenThai2.0 ASR failed. Confirm the local ONNX cache and audio input."
                ) from exc
            text = str(raw.get("text", "")).strip()
            chunks = raw.get("chunks", [])
            segments = [
                SpeechSegment(
                    text=str(c.get("text", "")).strip(),
                    start=c.get("start"),
                    end=c.get("end"),
                )
                for c in chunks
                if str(c.get("text", "")).strip()
            ]
            if text and not segments:
                segments = [SpeechSegment(text=text)]
            return ASRResult(
                text=text,
                segments=segments,
                language="th",
                duration=audio.original.duration,
                backend="openthai_onnx",
                model=OPENTHAI_MODEL_NAME,
                metadata={
                    "original_audio": audio.original.to_dict(),
                    "normalized_audio": audio.normalized.to_dict(),
                    "openthai_precision": engine.precision,
                    "openthai_prompt": effective_prompt,
                },
            )

        normalized: Path | None = None
        try:
            normalized = write_temporary_wav(audio, self.settings)
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("ASR input is not a readable supported audio file.") from exc
        try:
            raw = self._load()(
                str(normalized),
                model=effective_model,
                device="cpu",
                sampling_rate=16000,
                return_timestamps=True,
            )
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError("Thai ASR failed. Confirm the local model cache and WAV input.") from exc
        finally:
            if normalized is not None:
                normalized.unlink(missing_ok=True)
        text = str(raw.get("text", "") if isinstance(raw, dict) else raw).strip()
        chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
        segments = [
            SpeechSegment(
                text=str(c.get("text", "")).strip(),
                start=c.get("start"),
                end=c.get("end"),
            )
            for c in chunks
            if str(c.get("text", "")).strip()
        ]
        if text and not segments:
            segments = [SpeechSegment(text=text)]
        return ASRResult(
            text=text,
            segments=segments,
            language="th",
            duration=audio.original.duration,
            backend="pythaiasr",
            model=effective_model,
            metadata={
                "original_audio": audio.original.to_dict(),
                "normalized_audio": audio.normalized.to_dict(),
            },
        )
