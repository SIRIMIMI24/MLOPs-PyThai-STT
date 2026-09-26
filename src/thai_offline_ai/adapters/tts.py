from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any


class TTSAdapter:
    """Lazy PyThaiTTS adapter."""

    def __init__(self, model: str = "lunarlist_onnx", voice: str | None = None) -> None:
        self.model_name = model
        self.voice = voice
        self._model: Any | None = None
        self._lock = Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from pythaitts import TTS
                except ImportError as exc:
                    raise RuntimeError(
                        "PyThaiTTS is not installed. Run: pip install 'PyThaiTTS>=0.4.2,<0.5'"
                    ) from exc
                self._model = TTS(pretrained=self.model_name)
        return self._model

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
    ) -> str:
        if not text.strip():
            raise ValueError("Text is empty.")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model = self._get_model()
        selected_voice = voice or self.voice

        kwargs: dict[str, Any] = {"filename": str(destination)}
        if selected_voice:
            kwargs["speaker_idx"] = selected_voice

        result = model.tts(text.strip(), **kwargs)
        return str(result or destination)
