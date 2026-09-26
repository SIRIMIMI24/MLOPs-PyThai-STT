"""Explicitly download the verified PyThaiTTS Lunarlist ONNX files into a local cache."""

from __future__ import annotations

import sys
from pathlib import Path

FILES = (
    "tacotron2encoder-th.onnx",
    "tacotron2decoder-th.onnx",
    "tacotron2postnet-th.onnx",
    "vocoder.onnx",
)


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download

        from thai_offline_ai.adapters.tts import model_is_available
        from thai_offline_ai.config import load_settings
    except ImportError:
        print("[FAIL] Install Phase 2 dependencies first: python -m pip install -e \".[tts,dev]\"")
        return 1
    settings = load_settings()
    destination = Path(settings.tts.model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading PyThaiTTS Lunarlist ONNX files to {destination}")
    for name in FILES:
        try:
            path = hf_hub_download("pythainlp/thaitts-onnx", name, cache_dir=str(destination))
        except Exception as exc:  # noqa: BLE001 - setup must name the failed component
            print(f"[FAIL] Could not retrieve {name}: {type(exc).__name__}: {exc}")
            return 1
        print(f"[OK] {name}: {path}")
    if not model_is_available(destination):
        print("[FAIL] Download completed but the local model cache is incomplete.")
        return 1
    print("[OK] TTS model setup complete. Future app runs use local_files_only=True.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
