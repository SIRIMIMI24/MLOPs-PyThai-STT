"""Shared runtime preparation for CLI, UI and diagnostics."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from thai_offline_ai.config import Settings, validate_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.privacy import configure_privacy


def prepare_output_dir(settings: Settings) -> Path:
    """Create and test the configured local folder; leave no probe file behind."""
    validate_settings(settings)
    destination = Path(settings.app.output_dir)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=destination, prefix=".write-check-") as probe:
            probe.write(b"ok")
            probe.flush()
    except OSError as exc:
        raise WorkbenchError(
            f"Cannot write to output directory {destination}. "
            "Set app.output_dir or THAI_AI_OUTPUT_DIR to a writable local folder."
        ) from exc
    return destination


def output_path(settings: Settings, filename: str) -> Path:
    """Resolve one artifact name strictly inside the configured output folder."""
    if not filename or filename in {".", ".."} or any(c in filename for c in "/\\:\x00"):
        raise WorkbenchError("Output filename must be a simple name inside app.output_dir.")
    directory = prepare_output_dir(settings)
    target = (directory / filename).resolve()
    if target.parent != directory:
        raise WorkbenchError(
            "Output path escapes app.output_dir; remove the link or change the name."
        )
    return target


def output_artifact_path(settings: Settings, category: str, filename: str) -> Path:
    """Create a category folder and return a safe, unique artifact destination."""
    if not category or any(c in category for c in "/\\:\x00") or category in {".", ".."}:
        raise WorkbenchError("Output category must be a simple folder name.")
    if not filename or any(c in filename for c in "/\\:\x00") or filename in {".", ".."}:
        raise WorkbenchError("Output filename must be a simple name inside app.output_dir.")
    directory = prepare_output_dir(settings)
    artifact_dir = (directory / category).resolve()
    if artifact_dir.parent != directory:
        raise WorkbenchError("Output category escapes app.output_dir.")
    try:
        artifact_dir.mkdir(parents=False, exist_ok=True)
    except OSError as exc:
        raise WorkbenchError("Cannot create local output category folder.") from exc
    target = (artifact_dir / filename).resolve()
    if target.parent != artifact_dir:
        raise WorkbenchError("Output path escapes app.output_dir.")
    return target


def prepare_runtime(settings: Settings) -> Path:
    """Apply privacy before UI/backend imports, then validate local output storage."""
    validate_settings(settings)
    configure_privacy(settings.app.offline_runtime)
    logging.getLogger("thai_offline_ai").setLevel(settings.app.log_level)
    directory = prepare_output_dir(settings)
    os.environ["GRADIO_TEMP_DIR"] = str(output_path(settings, ".gradio"))
    return directory
