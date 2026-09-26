"""Reusable, local-only audio decoding and canonical ASR preparation."""
from __future__ import annotations

import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_output_dir

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg"})


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    format: str
    sample_rate: int
    channels: int
    duration: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CanonicalAudio:
    samples: object
    original: AudioMetadata
    normalized: AudioMetadata


def is_supported_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def _read(path: Path) -> tuple[object, AudioMetadata]:
    if not path.is_file() or path.stat().st_size == 0:
        raise WorkbenchError("ASR input must be a non-empty readable audio file.")
    if not is_supported_audio(path):
        supported = ", ".join(sorted(item.removeprefix(".").upper() for item in SUPPORTED_AUDIO_EXTENSIONS))
        raise WorkbenchError(f"Unsupported audio format. Supported formats: {supported}.")
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError("empty audio")
        samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        if samples.size == 0:
            raise ValueError("empty audio")
        return samples, AudioMetadata(
            format=path.suffix.lower().removeprefix("."),
            sample_rate=int(sample_rate),
            channels=int(info.channels),
            duration=float(info.frames / info.samplerate),
        )
    except WorkbenchError:
        raise
    except Exception as exc:
        raise WorkbenchError("ASR input is not a readable supported audio file.") from exc


def prepare_audio(path: str | Path, *, target_rate: int = 16000) -> CanonicalAudio:
    """Decode then convert a local file to mono float PCM at ``target_rate``."""
    if target_rate <= 0:
        raise WorkbenchError("ASR target sample rate must be positive.")
    samples, original = _read(Path(path))
    try:
        import numpy as np
        from scipy.signal import resample_poly

        mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
        if not np.isfinite(mono).all():
            raise ValueError("non-finite samples")
        if original.sample_rate != target_rate:
            divisor = math.gcd(original.sample_rate, target_rate)
            mono = resample_poly(mono, target_rate // divisor, original.sample_rate // divisor)
            mono = np.asarray(mono, dtype=np.float32)
        normalized = AudioMetadata(
            format="wav-pcm16",
            sample_rate=target_rate,
            channels=1,
            duration=float(len(mono) / target_rate),
        )
        return CanonicalAudio(samples=mono, original=original, normalized=normalized)
    except WorkbenchError:
        raise
    except Exception as exc:
        raise WorkbenchError("Audio normalization to mono 16 kHz PCM failed.") from exc


def write_temporary_wav(audio: CanonicalAudio, settings: object) -> Path:
    """Write a short-lived canonical WAV only beneath the configured output root."""
    try:
        import soundfile as sf

        directory = prepare_output_dir(settings) / "asr-tmp"
        directory.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".wav", delete=False) as handle:
            temporary = Path(handle.name)
        sf.write(str(temporary), audio.samples, audio.normalized.sample_rate, subtype="PCM_16")
        return temporary
    except Exception as exc:
        raise WorkbenchError("Cannot create the local temporary ASR audio file.") from exc
