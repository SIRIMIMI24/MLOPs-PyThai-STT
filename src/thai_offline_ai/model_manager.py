"""One-Click Local Model Manager & Cache Inspector.

Inspects local model directories without loading weights into memory. Network
downloads are never triggered during normal runtime; explicit setup execution
requires `allow_setup_network=True` and runs the verified setup script in an
isolated child process so the parent process offline guard remains intact.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from thai_offline_ai.adapters.asr import model_is_available as typhoon_is_available
from thai_offline_ai.adapters.diarization import (
    model_is_available as diarization_is_available,
)
from thai_offline_ai.adapters.kokoro_tts import kokoro_model_is_available
from thai_offline_ai.adapters.openthai_asr import resolve_available_precision
from thai_offline_ai.adapters.tts import model_is_available as lunarlist_is_available
from thai_offline_ai.agent.browser.browser_manager import playwright_is_installed
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError


@dataclass(frozen=True, slots=True)
class ManagedModelStatus:
    key: str
    title: str
    category: str
    ready: bool
    cache_dir: str
    file_count: int
    size_mb: float
    setup_script: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dir_stats(path: str | Path) -> tuple[int, float]:
    target = Path(path)
    if not target.is_dir():
        return 0, 0.0
    files = [p for p in target.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    return len(files), round(total_bytes / (1024 * 1024), 2)


class ModelManager:
    """Inspect and explicitly run setup scripts for local offline AI models."""

    SCRIPT_MAP: ClassVar[dict[str, str]] = {
        "kokoro_thai_onnx": "scripts/setup_kokoro_tts_model.py",
        "lunarlist_onnx": "scripts/setup_tts_model.py",
        "openthai2_asr": "scripts/setup_openthai_asr_model.py",
        "typhoon_asr": "scripts/setup_asr_model.py",
        "nemotron_diarization": "scripts/setup_diarization_model.py",
        "playwright_chromium": "scripts/setup_playwright.py",
    }

    def __init__(self, settings: Settings, project_root: str | Path | None = None) -> None:
        self.settings = settings
        self.project_root = Path(project_root or Path.cwd()).resolve()

    def list_statuses(self) -> list[ManagedModelStatus]:
        s = self.settings

        kokoro_ready = kokoro_model_is_available(s.tts.kokoro_model_dir)
        k_files, k_mb = _dir_stats(s.tts.kokoro_model_dir)

        lunar_ready = lunarlist_is_available(s.tts.model_dir)
        l_files, l_mb = _dir_stats(s.tts.model_dir)

        ot_prec = resolve_available_precision(s.asr.openthai_model_dir, s.asr.openthai_precision)
        ot_files, ot_mb = _dir_stats(s.asr.openthai_model_dir)

        ty_ready = typhoon_is_available(s.asr.model_dir)
        ty_files, ty_mb = _dir_stats(s.asr.model_dir)

        diar_ready = diarization_is_available(s.diarization.model_dir, s.diarization.precision)
        d_files, d_mb = _dir_stats(s.diarization.model_dir)

        pw_ready = playwright_is_installed()

        return [
            ManagedModelStatus(
                key="kokoro_thai_onnx",
                title="Kokoro Thai ONNX (12 Voices · 24 kHz · Thai+English)",
                category="TTS",
                ready=kokoro_ready,
                cache_dir=s.tts.kokoro_model_dir,
                file_count=k_files,
                size_mb=k_mb,
                setup_script=self.SCRIPT_MAP["kokoro_thai_onnx"],
                detail="พร้อมใช้งาน" if kokoro_ready else "ยังไม่ได้ดาวน์โหลดโมเดล",
            ),
            ManagedModelStatus(
                key="lunarlist_onnx",
                title="PyThaiTTS Lunarlist ONNX (22.05 kHz · Thai)",
                category="TTS",
                ready=lunar_ready,
                cache_dir=s.tts.model_dir,
                file_count=l_files,
                size_mb=l_mb,
                setup_script=self.SCRIPT_MAP["lunarlist_onnx"],
                detail="พร้อมใช้งาน" if lunar_ready else "ยังไม่ได้ดาวน์โหลดโมเดล",
            ),
            ManagedModelStatus(
                key="openthai2_asr",
                title="OpenThai 2.0 ASR Streaming ONNX (th-mix / th-TH)",
                category="ASR",
                ready=bool(ot_prec),
                cache_dir=s.asr.openthai_model_dir,
                file_count=ot_files,
                size_mb=ot_mb,
                setup_script=self.SCRIPT_MAP["openthai2_asr"],
                detail=f"พร้อมใช้งาน ({ot_prec})" if ot_prec else "ยังไม่ได้ดาวน์โหลดโมเดล",
            ),
            ManagedModelStatus(
                key="typhoon_asr",
                title="Typhoon ASR Realtime ONNX (PyThaiASR)",
                category="ASR",
                ready=ty_ready,
                cache_dir=s.asr.model_dir,
                file_count=ty_files,
                size_mb=ty_mb,
                setup_script=self.SCRIPT_MAP["typhoon_asr"],
                detail="พร้อมใช้งาน" if ty_ready else "ยังไม่ได้ดาวน์โหลดโมเดล",
            ),
            ManagedModelStatus(
                key="nemotron_diarization",
                title="Nemotron-3 Speaker Diarization ONNX",
                category="Diarization",
                ready=diar_ready,
                cache_dir=s.diarization.model_dir,
                file_count=d_files,
                size_mb=d_mb,
                setup_script=self.SCRIPT_MAP["nemotron_diarization"],
                detail="พร้อมใช้งาน" if diar_ready else "ยังไม่ได้ดาวน์โหลดโมเดล",
            ),
            ManagedModelStatus(
                key="playwright_chromium",
                title="Playwright + Chromium (JARVIS Browser Agent)",
                category="Agent",
                ready=pw_ready,
                cache_dir=str(Path(s.app.output_dir) / "jarvis_browser_profile"),
                file_count=0,
                size_mb=0.0,
                setup_script=self.SCRIPT_MAP["playwright_chromium"],
                detail="ติดตั้ง Playwright แล้ว" if pw_ready else "ยังไม่ได้ติดตั้ง Playwright",
            ),
        ]

    def to_table_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for item in self.list_statuses():
            rows.append(
                [
                    item.key,
                    item.category,
                    "READY" if item.ready else "MISSING",
                    f"{item.size_mb:.1f} MB ({item.file_count} files)",
                    item.detail,
                    f"python {item.setup_script}",
                ]
            )
        return rows

    def run_setup(
        self,
        model_key: str,
        *,
        allow_setup_network: bool = False,
        timeout_seconds: float = 600.0,
    ) -> dict[str, Any]:
        """Run the verified setup script in an isolated child process when explicitly authorized."""
        if model_key not in self.SCRIPT_MAP:
            raise WorkbenchError(
                f"ไม่รู้จักโมเดล '{model_key}' (ตัวเลือก: {', '.join(self.SCRIPT_MAP)})"
            )
        if not allow_setup_network:
            raise WorkbenchError(
                "Offline Runtime ปิดการดาวน์โหลดอัตโนมัติไว้ — "
                "กรุณาติ๊กยืนยัน 'เปิด Setup Mode (อนุญาตดาวน์โหลดโมเดล)' ก่อนกดติดตั้ง"
            )
        script_rel = self.SCRIPT_MAP[model_key]
        script_path = (self.project_root / script_rel).resolve()
        if not script_path.is_file():
            raise WorkbenchError(f"ไม่พบสคริปต์ติดตั้ง: {script_path}")

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkbenchError(
                f"การติดตั้ง '{model_key}' ใช้เวลานานเกิน {timeout_seconds:.0f} วินาที"
            ) from exc
        except OSError as exc:
            raise WorkbenchError(f"ไม่สามารถรันสคริปต์ติดตั้ง '{model_key}': {exc}") from exc

        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            raise WorkbenchError(
                f"การติดตั้ง '{model_key}' ล้มเหลว (exit code {proc.returncode}): {output[:400]}"
            )
        updated = next(item for item in self.list_statuses() if item.key == model_key)
        return {
            "key": model_key,
            "ready": updated.ready,
            "size_mb": updated.size_mb,
            "output": output[-1200:],
        }
