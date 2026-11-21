"""Portable Windows desktop packaging and launcher generator."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thai_offline_ai.config import Settings
from thai_offline_ai.model_manager import ModelManager

START_BAT_CONTENT = """@echo off
setlocal
cd /d "%~dp0"
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHAINLP_OFFLINE=1
if not exist ".venv\\Scripts\\python.exe" (
    echo [FAIL] .venv\\Scripts\\python.exe not found. Run setup_windows.ps1 first.
    pause
    exit /b 1
)
echo Starting Thai Offline AI Workbench on http://127.0.0.1:7860 ...
".venv\\Scripts\\python.exe" run_app.py
"""

DOCTOR_BAT_CONTENT = """@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\\Scripts\\python.exe" (
    echo [FAIL] .venv\\Scripts\\python.exe not found.
    pause
    exit /b 1
)
".venv\\Scripts\\python.exe" scripts\\doctor.py
pause
"""


@dataclass(frozen=True, slots=True)
class PortableBundleResult:
    bundle_dir: str
    manifest_path: str
    launcher_bat: str
    doctor_bat: str
    copied_files: list[str]
    ready_models: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_portable_bundle(
    settings: Settings,
    *,
    project_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> PortableBundleResult:
    """Create a clean portable desktop launcher bundle without copying private user outputs."""
    root = Path(project_root or Path.cwd()).resolve()
    bundle_root = Path(
        output_dir or (Path(settings.app.output_dir) / "portable_bundle")
    ).expanduser().resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)

    launcher_path = bundle_root / "start_workbench.bat"
    doctor_path = bundle_root / "doctor_check.bat"
    launcher_path.write_text(START_BAT_CONTENT, encoding="utf-8")
    doctor_path.write_text(DOCTOR_BAT_CONTENT, encoding="utf-8")

    copied: list[str] = ["start_workbench.bat", "doctor_check.bat"]
    for rel in (
        "run_app.py",
        "pyproject.toml",
        "README.md",
        "config/settings.example.yaml",
        "config/agent_config.yaml",
    ):
        src = root / rel
        if src.is_file():
            dst = bundle_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)

    statuses = ModelManager(settings, project_root=root).list_statuses()
    ready_models = [m.key for m in statuses if m.ready]
    manifest = {
        "name": "Thai Offline AI Workbench — Portable Desktop Bundle",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": settings.app.host,
        "port": settings.app.port,
        "offline_runtime": settings.app.offline_runtime,
        "launcher_bat": str(launcher_path),
        "doctor_bat": str(doctor_path),
        "copied_files": copied,
        "ready_models": ready_models,
        "model_statuses": [m.to_dict() for m in statuses],
    }
    manifest_path = bundle_root / "portable_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return PortableBundleResult(
        bundle_dir=str(bundle_root),
        manifest_path=str(manifest_path),
        launcher_bat=str(launcher_path),
        doctor_bat=str(doctor_path),
        copied_files=copied,
        ready_models=ready_models,
    )
