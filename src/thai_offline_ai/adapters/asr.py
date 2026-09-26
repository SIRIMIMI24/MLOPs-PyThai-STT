from __future__ import annotations

from pathlib import Path
from typing import Any


class ASRAdapter:
    """PyThaiASR adapter; exact backend API must be verified against installed version."""

    def __init__(
        self,
        model: str | None = None,
        device: str = "cpu",
        sampling_rate: int = 16000,
    ) -> None:
        self.model = model
        self.device = device
        self.sampling_rate = sampling_rate

    def transcribe_file(self, audio_path: str | Path) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(path)

        try:
            from pythaiasr import asr
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "PyThaiASR is not installed. Install a verified PyThaiASR version/backend first."
            ) from exc

        kwargs: dict[str, Any] = {
            "device": self.device,
            "sampling_rate": self.sampling_rate,
        }
        if self.model:
            kwargs["model"] = self.model

        result = asr(str(path), **kwargs)
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        return str(result)
