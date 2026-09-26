"""Explicitly download PyThaiASR's verified Typhoon ONNX cache."""
from __future__ import annotations

import os

from thai_offline_ai.config import load_settings


def main() -> int:
    settings = load_settings()
    os.environ["PYTHAIASR_DATA_DIR"] = settings.asr.model_dir
    from pythaiasr.download import get_typhoon_model_files

    print(get_typhoon_model_files())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
