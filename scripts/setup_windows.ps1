[CmdletBinding()]
param(
    # Optional path for an existing Python 3.10 that is not registered with py.exe.
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments, [string]$Step)
    Write-Host "[INFO] $Step"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE). Fix the error above and rerun setup."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    Write-Host "Thai Offline AI Workbench - Phase 1 setup"
    $pythonArguments = @()
    if ($PythonExe) {
        $pythonCommand = Get-Command -Name $PythonExe -CommandType Application -ErrorAction Stop
    } else {
        $pythonCommand = Get-Command -Name py -CommandType Application -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {
            throw "Python launcher not found. Install Python 3.10 with the launcher, or pass -PythonExe PATH. See README.md."
        }
        $pythonArguments = @("-3.10")
    }
    $versionCheck = "import sys; print(sys.executable); print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)"
    Invoke-Checked -Executable $pythonCommand.Source -Arguments ($pythonArguments + @("-c", $versionCheck)) -Step "Check Python 3.10"

    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath ".venv")) {
        Invoke-Checked -Executable $pythonCommand.Source -Arguments ($pythonArguments + @("-m", "venv", ".venv")) -Step "Create project .venv"
    }
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "The project .venv is incomplete. Rename the old .venv folder and rerun setup; no files were deleted."
    }
    Invoke-Checked -Executable $venvPython -Arguments @("-c", $versionCheck) -Step "Check project .venv interpreter"
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "--version") -Step "Check pip (repair with .venv\Scripts\python.exe -m ensurepip --upgrade if missing)"
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-e", ".[dev]") -Step "Install Phase 1 core and test dependencies"
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "check") -Step "Validate installed dependencies"
    if (-not (Test-Path -LiteralPath "config\settings.yaml")) {
        Copy-Item -LiteralPath "config\settings.example.yaml" -Destination "config\settings.yaml"
    }
    Invoke-Checked -Executable $venvPython -Arguments @("scripts/doctor.py") -Step "Check Phase 1 readiness"
    Write-Host "[OK] Phase 1 setup complete. No optional inference packages or large models were installed."
    Write-Host "Run: .\.venv\Scripts\python.exe run_app.py"
    Write-Host "Open: http://127.0.0.1:7860 (or your configured local port)"
} catch {
    Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
