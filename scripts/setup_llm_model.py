"""Explicitly pull the configured Ollama model; never called by normal runtime."""
from __future__ import annotations

import subprocess
import sys

from thai_offline_ai.adapters.llm import find_ollama_executable
from thai_offline_ai.config import load_settings
from thai_offline_ai.errors import WorkbenchError


def main() -> int:
    try:
        settings = load_settings()
        if settings.llm.backend != "ollama":
            raise WorkbenchError("Phase 4 setup supports the Ollama backend only.")
        executable = find_ollama_executable()
        if executable is None:
            raise WorkbenchError("Ollama is not installed. Install Ollama for Windows, then run this script again.")
        print(f"Pulling local Ollama model explicitly: {settings.llm.model}")
        return subprocess.run([executable, "pull", settings.llm.model], check=False).returncode
    except WorkbenchError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
