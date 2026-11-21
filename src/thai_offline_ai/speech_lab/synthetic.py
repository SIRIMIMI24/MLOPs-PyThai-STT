"""Local synthetic speech generation and deterministic CPU audio augmentation."""

from __future__ import annotations

import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.speech_lab.datasets import (
    DatasetManifest,
    DatasetSample,
    compute_file_sha256,
)

GENERATOR_VERSION = "speech-lab-synth-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Deterministic CPU audio augmentation profile for synthetic ASR training data."""

    profile_name: str = "clean"
    snr_db: float | None = None
    telephone_bandpass: bool = False
    gain_db: float = 0.0
    pad_silence_ms: int = 0
    speed_factor: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.snr_db is not None and not 0.0 <= self.snr_db <= 50.0:
            raise WorkbenchError("AugmentationConfig.snr_db must be between 0 and 50 dB.")
        if not -24.0 <= self.gain_db <= 18.0:
            raise WorkbenchError("AugmentationConfig.gain_db must be between -24 and 18 dB.")
        if not 0 <= self.pad_silence_ms <= 3000:
            raise WorkbenchError("AugmentationConfig.pad_silence_ms must be 0–3000 ms.")
        if not 0.8 <= self.speed_factor <= 1.25:
            raise WorkbenchError("AugmentationConfig.speed_factor must be 0.80–1.25.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AUGMENTATION_PRESETS: dict[str, AugmentationConfig] = {
    "clean": AugmentationConfig(profile_name="clean"),
    "gaussian_noise": AugmentationConfig(profile_name="gaussian_noise", snr_db=20.0, pad_silence_ms=150),
    "telephone_bandpass": AugmentationConfig(
        profile_name="telephone_bandpass",
        telephone_bandpass=True,
        snr_db=18.0,
        pad_silence_ms=200,
    ),
    "fast_quiet": AugmentationConfig(
        profile_name="fast_quiet",
        gain_db=-4.0,
        speed_factor=1.08,
        pad_silence_ms=120,
    ),
}


def augment_waveform(
    samples: np.ndarray, sample_rate: int, config: AugmentationConfig
) -> np.ndarray:
    """Apply deterministic CPU audio augmentations to a 1-D float32 waveform in [-1, 1]."""
    if sample_rate <= 0:
        raise WorkbenchError("sample_rate must be positive.")
    wav = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
    if wav.size == 0:
        return wav

    # 1. Optional speed perturbation via linear interpolation (deterministic, zero external deps).
    if abs(config.speed_factor - 1.0) > 1e-4:
        new_len = max(1, round(wav.size / config.speed_factor))
        old_idx = np.linspace(0.0, wav.size - 1, num=new_len, dtype=np.float64)
        wav = np.interp(old_idx, np.arange(wav.size, dtype=np.float64), wav).astype(np.float32)

    # 2. Optional telephone bandpass (300–3400 Hz) using SciPy Butterworth filter.
    if config.telephone_bandpass and sample_rate > 7200 and wav.size > 32:
        from scipy.signal import butter, sosfilt

        nyquist = sample_rate / 2.0
        low = max(0.005, 300.0 / nyquist)
        high = min(0.98, 3400.0 / nyquist)
        if low < high:
            sos = butter(4, [low, high], btype="bandpass", output="sos")
            wav = sosfilt(sos, wav).astype(np.float32)

    # 3. Optional additive Gaussian noise at target SNR dB.
    if config.snr_db is not None:
        sig_power = float(np.mean(wav * wav))
        if sig_power > 1e-12:
            rng = np.random.default_rng(config.seed)
            noise_power = sig_power / (10.0 ** (config.snr_db / 10.0))
            noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
            wav = wav + noise

    # 4. Optional volume gain scaling.
    if abs(config.gain_db) > 1e-4:
        wav = wav * np.float32(10.0 ** (config.gain_db / 20.0))

    # 5. Optional leading & trailing silence padding.
    if config.pad_silence_ms > 0:
        pad = round(sample_rate * (config.pad_silence_ms / 1000.0))
        if pad > 0:
            zeros = np.zeros(pad, dtype=np.float32)
            wav = np.concatenate([zeros, wav, zeros])

    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def _read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise WorkbenchError(f"Expected 16-bit PCM WAV from local TTS, got {width * 8}-bit.")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


@dataclass(frozen=True, slots=True)
class SyntheticCorpusItem:
    sample_id: str
    text: str
    domain_tags: tuple[str, ...] = ()
    voice: str | None = None
    speed: float | None = None


@dataclass(slots=True)
class SyntheticBatchSummary:
    manifest: DatasetManifest
    manifest_path: Path
    output_dir: Path
    generated_count: int
    failed_items: list[dict[str, str]] = field(default_factory=list)


class SyntheticSpeechGenerator:
    """Adapter-based local synthetic speech pipeline producing provenance-rich manifests."""

    def __init__(self, settings: Settings, tts_adapter: TTSAdapter | None = None) -> None:
        self.settings = settings
        self.tts_adapter = tts_adapter or TTSAdapter.from_settings(settings)

    def generate_dataset(
        self,
        items: Sequence[SyntheticCorpusItem],
        *,
        dataset_id: str,
        dataset_version: str = "v1",
        output_dir: str | Path,
        augmentations: Sequence[AugmentationConfig] = (AugmentationConfig(),),
        tts_model: str | None = None,
    ) -> SyntheticBatchSummary:
        if not items:
            raise WorkbenchError("Provide at least one approved text item to generate synthetic audio.")
        if not augmentations:
            raise WorkbenchError("Provide at least one AugmentationConfig profile.")

        out_root = Path(output_dir).expanduser().resolve()
        audio_dir = out_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        samples: list[DatasetSample] = []
        failed: list[dict[str, str]] = []

        for item_idx, item in enumerate(items):
            text = item.text.strip()
            if not text:
                failed.append({"sample_id": item.sample_id, "error": "Empty source text."})
                continue
            try:
                if tts_model is None and item.voice is None and item.speed is None:
                    tts_res = self.tts_adapter.synthesize(text)
                else:
                    tts_res = self.tts_adapter.synthesize(
                        text, model=tts_model, voice=item.voice, speed=item.speed
                    )
                base_wav, sr = _read_pcm16_wav(Path(tts_res.audio_path))
            except WorkbenchError as exc:
                failed.append({"sample_id": item.sample_id, "error": str(exc)})
                continue

            for aug_idx, aug in enumerate(augmentations):
                aug_cfg = AugmentationConfig(
                    profile_name=aug.profile_name,
                    snr_db=aug.snr_db,
                    telephone_bandpass=aug.telephone_bandpass,
                    gain_db=aug.gain_db,
                    pad_silence_ms=aug.pad_silence_ms,
                    speed_factor=aug.speed_factor,
                    seed=aug.seed + item_idx * 100 + aug_idx,
                )
                processed = augment_waveform(base_wav, sr, aug_cfg)
                syn_id = f"{item.sample_id}__{aug_cfg.profile_name}"
                out_wav = audio_dir / f"{syn_id}.wav"
                _write_pcm16_wav(out_wav, processed, sr)
                sha = compute_file_sha256(out_wav)
                duration_s = round(processed.size / sr, 3)
                samples.append(
                    DatasetSample(
                        sample_id=syn_id,
                        split="train_synthetic",
                        audio_path=str(out_wav),
                        source_text=text,
                        synthetic=True,
                        provenance=f"local_tts:{tts_res.backend}:{tts_res.model}",
                        consent_status="synthetic_authorized_voice",
                        transcript_state="synthetic_source",
                        audio_sha256=sha,
                        dataset_version=dataset_version,
                        domain_tags=item.domain_tags,
                        metadata={
                            "tts_backend": tts_res.backend,
                            "tts_model": tts_res.model,
                            "voice": tts_res.voice,
                            "speed": tts_res.speed,
                            "sample_rate": sr,
                            "duration_seconds": duration_s,
                            "augmentation": aug_cfg.to_dict(),
                            "generator_version": GENERATOR_VERSION,
                            "created_at": _utc_now(),
                        },
                    )
                )

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            split="train_synthetic",
            version=dataset_version,
            samples=samples,
        )
        manifest_path = manifest.save_jsonl(out_root / f"{dataset_id}_{dataset_version}.jsonl")
        return SyntheticBatchSummary(
            manifest=manifest,
            manifest_path=manifest_path,
            output_dir=out_root,
            generated_count=len(samples),
            failed_items=failed,
        )
