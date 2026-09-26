"""Lazy, cache-only local PyThaiASR speaker-diarization adapter."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Any

from thai_offline_ai.audio import prepare_audio, write_temporary_wav
from thai_offline_ai.config import Settings
from thai_offline_ai.diarization_types import DiarizationResult, SpeakerTurn
from thai_offline_ai.errors import WorkbenchError

REQUIRED_INT8_FILES = ("preprocessor_core.onnx", "model.int8.onnx", "constants.npz")
REQUIRED_FP32_FILES = ("preprocessor_core.onnx", "model.onnx", "constants.npz")


def model_is_available(model_dir: str | Path, precision: str = "int8") -> bool:
    root = Path(model_dir)
    names = REQUIRED_FP32_FILES if precision == "fp32" else REQUIRED_INT8_FILES
    return root.is_dir() and all((root / name).is_file() and (root / name).stat().st_size > 0 for name in names)


def cuda_is_usable() -> bool:
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except (ImportError, OSError):
        return False


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_is_usable():
            raise WorkbenchError(
                "CUDA diarization is unavailable. Install compatible ONNX Runtime GPU or use diarization.device=cpu."
            )
        return "cuda"
    return "cuda" if cuda_is_usable() else "cpu"


def nvidia_gpu_summary() -> str | None:
    """Return lightweight local GPU metadata without initializing a model."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    return None


class DiarizationAdapter:
    """Diarize a local file; backend model construction occurs only on first call."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backend: Any | None = None
        self._lock = Lock()

    def _load(self) -> Any:
        config = self.settings.diarization
        if not config.enabled:
            raise WorkbenchError("Speaker diarization is disabled. Set diarization.enabled=true after local model setup.")
        if self._backend is not None:
            return self._backend
        with self._lock:
            if not model_is_available(config.model_dir, config.precision):
                raise WorkbenchError(
                    "Diarization model is not available locally. Run: python scripts/setup_diarization_model.py"
                )
            try:
                os.environ["PYTHAIASR_DATA_DIR"] = str(Path(config.model_dir).parent)
                from pythaiasr import diarize
            except (ImportError, OSError) as exc:
                raise WorkbenchError(
                    "PyThaiASR diarization is unavailable. Run: python -m pip install -e \".[asr,dev]\""
                ) from exc
            self._backend = diarize
        return self._backend

    def diarize_file(self, audio_path: str | Path) -> DiarizationResult:
        config = self.settings.diarization
        normalized: Path | None = None
        started = time.perf_counter()
        try:
            audio = prepare_audio(audio_path, target_rate=config.sampling_rate)
            normalized = write_temporary_wav(audio, self.settings)
            device = resolve_device(config.device)
            backend = self._load()
            with self._lock:
                raw_turns = backend(
                    str(normalized),
                    model=config.model,
                    device=device,
                    precision=config.precision,
                    sampling_rate=config.sampling_rate,
                    model_path=config.model_dir,
                )
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError(
                "Local speaker diarization failed. Confirm the local model cache and input audio."
            ) from exc
        finally:
            if normalized is not None:
                normalized.unlink(missing_ok=True)
        turns = normalize_turns(raw_turns)
        processing_seconds = time.perf_counter() - started
        return DiarizationResult(
            turns=turns,
            speakers=sorted({turn.speaker for turn in turns}),
            duration=audio.original.duration,
            backend="pythaiasr",
            model=config.model,
            device=device,
            metadata={
                "original_audio": audio.original.to_dict(),
                "normalized_audio": audio.normalized.to_dict(),
                "precision": config.precision,
                "processing_seconds": processing_seconds,
                "rtf": (processing_seconds / audio.original.duration)
                if audio.original.duration > 0
                else None,
            },
        )


def normalize_turns(raw_turns: object) -> list[SpeakerTurn]:
    """Normalize verified PyThaiASR dictionaries to deterministic anonymous labels."""
    if not isinstance(raw_turns, list):
        raise WorkbenchError("Diarization backend returned an invalid turn list.")
    ordered = sorted(
        (item for item in raw_turns if isinstance(item, dict)),
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    labels: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for item in ordered:
        try:
            start, end = float(item["start"]), float(item["end"])
            original = str(item["speaker"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkbenchError("Diarization backend returned an invalid speaker turn.") from exc
        if end <= start:
            continue
        normalized = labels.setdefault(original, f"SPEAKER_{len(labels):02d}")
        metadata = {"backend_speaker": original} if normalized != original else {}
        turns.append(SpeakerTurn(start=start, end=end, speaker=normalized, metadata=metadata))
    return turns
