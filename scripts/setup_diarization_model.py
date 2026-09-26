"""Explicitly acquire the configured PyThaiASR Nemotron ONNX files once."""
from __future__ import annotations

import sys

from thai_offline_ai.adapters.diarization import model_is_available
from thai_offline_ai.config import load_settings
from thai_offline_ai.errors import WorkbenchError


def main() -> int:
    try:
        settings = load_settings()
        config = settings.diarization
        if config.backend != "pythaiasr" or config.model != "nemotron-3-diarization":
            raise WorkbenchError("Phase 5 setup supports PyThaiASR nemotron-3-diarization only.")
        if model_is_available(config.model_dir, config.precision):
            print(f"Diarization model is already ready: {config.model_dir}")
            return 0
        try:
            from pythaiasr.download import get_nemotron_diarization_model_files
        except (ImportError, OSError) as exc:
            raise WorkbenchError(
                'PyThaiASR is missing. Run: python -m pip install -e ".[asr,dev]"'
            ) from exc
        print(f"Explicitly downloading {config.model} ({config.precision}) to {config.model_dir}")
        get_nemotron_diarization_model_files(config.model_dir, precision=config.precision)
        if not model_is_available(config.model_dir, config.precision):
            raise WorkbenchError("Diarization setup finished without all required non-empty model files.")
        print("Diarization model setup complete.")
        return 0
    except WorkbenchError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - setup must surface upstream download failure
        print(f"[FAIL] Diarization model setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
