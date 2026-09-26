"""Lightweight environment checks shared by the doctor command and System tab."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata

from thai_offline_ai.config import Settings, load_settings, validate_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.privacy import configure_privacy, is_local_url
from thai_offline_ai.runtime import prepare_output_dir


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def _package_check(distribution: str, module: str, *, required: bool, check_imports: bool) -> Check:
    failure_status = "FAIL" if required else "SKIP"
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        remedy = "Run: python -m pip install -e ." if required else "Optional; future phase."
        return Check(failure_status, distribution, f"Not installed. {remedy}")
    try:
        if importlib.util.find_spec(module) is None:
            raise ImportError("Module cannot be discovered")
        if required and check_imports:
            importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report broken third-party imports, never suppress them
        return Check(
            "FAIL" if required else "WARN",
            distribution,
            f"Installed {version}, but import/discovery failed ({type(exc).__name__}). "
            "Activate the project .venv; run python -m pip check and reinstall the project.",
        )
    detail = "import OK" if required and check_imports else "metadata only; import not tested"
    return Check("OK", distribution, f"{version}; {detail}")


def collect_diagnostics(
    settings: Settings | None = None, *, check_imports: bool = True
) -> list[Check]:
    """Check required imports and optional metadata, never optional model readiness."""
    results = [Check("OK", "Operating system", platform.platform())]
    supported = (3, 10) <= sys.version_info[:2] < (3, 13)
    python_status = (
        "FAIL" if not supported else ("OK" if sys.version_info[:2] == (3, 10) else "WARN")
    )
    results.append(
        Check(
            python_status,
            "Python",
            f"{platform.python_version()}; Python 3.10 is the primary supported/tested version.",
        )
    )
    results.append(Check("OK", "Interpreter", sys.executable))
    try:
        settings = validate_settings(settings) if settings is not None else load_settings()
    except WorkbenchError as exc:
        results.append(Check("FAIL", "Configuration", str(exc)))
        # Continue package checks with safe defaults; still return a failure exit code.
        settings = Settings()
    configure_privacy(settings.app.offline_runtime)
    for distribution, module, required in [
        ("thai-offline-ai-workbench", "thai_offline_ai", True),
        ("PyYAML", "yaml", True),
        ("pythainlp", "pythainlp", settings.nlp.enabled),
        (
            "python-crfsuite",
            "pycrfsuite",
            settings.nlp.enabled and settings.nlp.sentence_engine == "crfcut",
        ),
        ("gradio", "gradio", True),
        ("PyThaiTTS", "pythaitts", False),
        ("pythaiasr", "pythaiasr", False),
        ("onnxruntime", "onnxruntime", False),
        ("onnxruntime-gpu", "onnxruntime", False),
    ]:
        results.append(
            _package_check(distribution, module, required=required, check_imports=check_imports)
        )
    if settings.nlp.enabled and check_imports and not any(c.status == "FAIL" for c in results):
        try:
            from thai_offline_ai.adapters.nlp import NLPAdapter

            NLPAdapter.from_config(settings.nlp, settings.app.offline_runtime).analyze("สวัสดีครับ")
            results.append(Check("OK", "NLP resources", "Bundled tokenizer smoke test passed."))
        except WorkbenchError as exc:
            results.append(Check("FAIL", "NLP resources", str(exc)))
    try:
        directory = prepare_output_dir(settings)
        results.append(Check("OK", "Output directory", f"{directory}; write probe passed."))
    except WorkbenchError as exc:
        results.append(Check("FAIL", "Output directory", str(exc)))
    results.append(
        Check(
            "OK" if settings.app.offline_runtime else "WARN",
            "Offline runtime",
            f"{settings.app.offline_runtime}; external Python socket guard "
            + ("enabled." if settings.app.offline_runtime else "not requested; use true."),
        )
    )
    results.append(
        Check("OK", "UI bind", f"{settings.app.host}:{settings.app.port}; sharing disabled.")
    )
    results.append(Check("OK", "Log level", settings.app.log_level))
    for name in ("nlp", "tts", "asr", "llm"):
        component = getattr(settings, name)
        status = (
            ("OK" if component.enabled else "SKIP")
            if name == "nlp"
            else ("WARN" if component.enabled else "SKIP")
        )
        detail = f"enabled={component.enabled}; backend={component.backend}"
        if name != "nlp":
            detail += "; future phase, inference unavailable even when enabled."
        results.append(Check(status, f"Component {name}", detail))
    results.append(
        Check(
            "OK" if is_local_url(settings.llm.base_url) else "FAIL",
            "LLM URL",
            f"{settings.llm.base_url}; local={is_local_url(settings.llm.base_url)}; "
            "no server contacted.",
        )
    )
    gpu_hint = (
        "nvidia-smi found; hardware not probed" if shutil.which("nvidia-smi") else "not probed"
    )
    results.append(Check("SKIP", "GPU", f"{gpu_hint}; no GPU is required for Phase 1."))
    return results


def format_report(checks: list[Check]) -> str:
    return "\n".join(f"[{check.status}] {check.name}: {check.detail}" for check in checks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1 without loading optional models")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    if args.config:
        try:
            settings = load_settings(args.config)
        except WorkbenchError as exc:
            print(f"[FAIL] Configuration: {exc}")
            return 1
    else:
        settings = None
    checks = collect_diagnostics(settings)
    print(format_report(checks))
    return int(any(check.status == "FAIL" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
