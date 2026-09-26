import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell setup test")
def test_setup_stops_on_native_prerequisite_failure(tmp_path):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile("scripts/setup_windows.ps1", scripts / "setup_windows.ps1")
    fake_python = project / "failed-python.cmd"
    fake_python.write_text("@echo off\nexit /b 37\n", encoding="ascii")
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(scripts / "setup_windows.ps1"),
            "-PythonExe",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1
    assert "[FAIL]" in result.stdout and "exit 37" in result.stdout
    assert "Install Phase 1" not in result.stdout
    assert not (project / ".venv").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell parser test")
def test_setup_has_valid_powershell_syntax():
    script = Path("scripts/setup_windows.ps1").resolve()
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "'" + str(script).replace("'", "''") + "',[ref]$tokens,[ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors; exit 1 }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
