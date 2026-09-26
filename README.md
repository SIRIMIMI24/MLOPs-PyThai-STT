# Thai Offline AI Workbench — Phase 1

A local, CPU-first Thai NLP application using **PyThaiNLP** and **Gradio**.
Phase 1 supports Thai normalization, word and sentence tokenization, token counts,
and Thai/Latin character counts. The UI preserves Home, NLP, TTS, ASR, Local LLM,
and System tabs. **Only NLP is functional in this phase.**

TTS, ASR, LLM inference, and microphone input are reserved for later phases.
No GPU, NVIDIA/CUDA, WSL, Docker, API key, or separately downloaded model is needed.
Python **3.10 (64-bit)** is the primary supported version.

## Windows: fresh folder to working NLP UI

Open PowerShell in the project folder containing `run_app.py`.

### 1. Check Python

```powershell
py -0p
py -3.10 --version
```

If Python 3.10 is missing, install it during setup, then open a new PowerShell:

```powershell
winget install --exact --id Python.Python.3.10 --source winget
```

This is the official Python package identifier in the
[Microsoft WinGet repository](https://github.com/microsoft/winget-pkgs/tree/master/manifests/p/Python/Python/3/10).
If WinGet is unavailable, use the [Python Windows downloads](https://www.python.org/downloads/windows/)
and include the Python launcher. The setup script also accepts `-PythonExe` for
an existing Python 3.10 executable that is not registered with the launcher.

### 2. Set up the project

Internet access is needed **during this installation step**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

The script checks Python, creates or validates `.venv`, checks pip, installs
`.[dev]`, runs `pip check`, copies the example configuration only if the real
configuration is absent, and runs doctor. It stops on every critical failure.
It does not install optional inference extras or download large models.

For an unregistered Python executable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -PythonExe "C:\Path\To\Python310\python.exe"
```

Manual equivalent after verifying Python 3.10:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip check
if (-not (Test-Path -LiteralPath config/settings.yaml)) {
    Copy-Item -LiteralPath config/settings.example.yaml -Destination config/settings.yaml
}
python scripts/doctor.py
```

Use `python -m pip install -e .` if you do not need the test tools.

### 3. Run NLP

Activation is optional when using the virtual environment executable directly:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli nlp "ประเทศไทยมีประชากรประมาณเจ็ดสิบล้านคน"
.\.venv\Scripts\python.exe run_app.py
```

Open **http://127.0.0.1:7860**, select **NLP**, enter Thai text, and click
**วิเคราะห์ข้อความ**. Normalized text, word tokens, sentence tokens, and token
count appear separately. Stop the application with **Ctrl+C**.

Subsequent runs require only `.\.venv\Scripts\python.exe run_app.py`.

## CLI and configuration

The original `--text` syntax remains supported:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli nlp --text "สวัสดีครับ วันนี้อากาศดี"
.\.venv\Scripts\python.exe -m thai_offline_ai.cli --config config/settings.yaml nlp "ภาษาไทย"
.\.venv\Scripts\python.exe -m thai_offline_ai.cli doctor
```

NLP prints indented JSON with readable Thai text. The existing `tts`, `asr`, and
`chat` command syntax is retained, but returns an actionable Phase 1 message and
exit code 2 without invoking a model.

Configuration order: **defaults → YAML → environment variables**.
`config/settings.yaml` is optional; an explicitly supplied `--config` or
`THAI_AI_CONFIG` file must exist. Unknown fields, invalid types, unsupported
backends, duplicate YAML keys, unknown `THAI_AI_*` overrides, and unsafe endpoints
are errors. `.env.example` documents shell
variables; **`.env` is not loaded automatically**.

Relative configuration and output paths resolve from the current working
directory. Run the commands above from the project root. You can explicitly
configure another local output directory; network shares and drive roots are rejected.
Output creation/permissions are checked. Phase 1 does not save NLP inputs/results.

Examples of PowerShell overrides:

```powershell
$env:THAI_AI_OUTPUT_DIR = "outputs"
$env:THAI_AI_PORT = "7861"
$env:THAI_AI_OFFLINE_RUNTIME = "true"
$env:THAI_AI_LOG_LEVEL = "WARNING"
$env:THAI_AI_NLP_ENABLED = "true"
```

Other supported overrides: `THAI_AI_HOST`, `THAI_AI_CONFIG`,
`THAI_AI_NLP_BACKEND`, `THAI_AI_NLP_WORD_ENGINE`, `THAI_AI_NLP_SENTENCE_ENGINE`,
`THAI_AI_TTS_ENABLED`, `THAI_AI_TTS_BACKEND`, `THAI_AI_ASR_ENABLED`,
`THAI_AI_ASR_BACKEND`, `THAI_AI_LLM_ENABLED`, `THAI_AI_LLM_BACKEND`, and
`THAI_AI_LLM_BASE_URL`. Setting a future component to enabled does not make its
inference available; doctor reports a warning.

The supported word engines are `newmm` and `newmm-safe`. Sentence engines are
`crfcut` (default), `whitespace`, and `whitespace+newline`. CRFcut uses a small
model bundled in the PyThaiNLP wheel. Its `python-crfsuite` dependency is included
in this project's core installation. No PyThaiNLP `full` extra is needed.

## Runtime privacy

- Gradio telemetry and version-check analytics are disabled before import and
  explicitly disabled on the app. Public sharing, SSR, MCP, and monitoring are off.
- UI binding and configured LLM URLs are restricted to loopback. Cloud LLM URLs
  are rejected even when offline mode is false. No LLM server is contacted in Phase 1.
- Offline runtime installs a process-wide Python audit hook rejecting external
  socket connections, DNS lookups, datagrams, and public socket binds. Redirects
  and proxy destinations must still pass this guard. Local startup checks bypass proxies.
- PyThaiNLP offline/read-only mode and Hugging Face offline/telemetry opt-outs are
  enforced. The supported NLP engines use installed resources only.
- System fonts avoid remote font requests. Browser response headers restrict
  scripts, styles, fonts, and connections to the local application.
- Input text is not written to application logs or files. Gradio temporary
  storage is configured below the output directory; no upload/microphone UI is offered.

These safeguards apply to the application process, not pip or setup commands.
The Python socket guard is **not an OS firewall** and cannot police arbitrary
native code, child processes, browser extensions, or unrelated programs.
Keep `offline_runtime: true`; false disables installation of the socket guard in
a fresh process, but never enables cloud backends or optional inference. Once
installed, the guard cannot be removed until the process exits.
The old optional adapter prototypes are not supported Phase 1 entry points and
must be verified/hardened before a future phase exposes them.

## Validation and diagnostics

```powershell
.\.venv\Scripts\python.exe -m compileall src tests run_app.py scripts
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/doctor.py
```

Doctor uses `[OK]`, `[WARN]`, `[FAIL]`, and `[SKIP]`; exit 1 means a required
dependency, configuration, or output check failed. Optional packages are checked
through metadata/discovery only. Doctor imports core packages and runs a small
bundled-resource NLP smoke test; it never initializes TTS/ASR/LLM/GPU models.
The System tab uses lightweight metadata checks and does not claim inference readiness.

Tests block external Python networking. They exercise real NLP and briefly start
a local Gradio server, send an NLP HTTP request, then close it. No microphone,
GPU, optional inference package, internet access, or model download is required.

See [troubleshooting](docs/TROUBLESHOOTING.md),
[architecture](docs/ARCHITECTURE.md), and
[Phase 1 validation](docs/PHASE1_VALIDATION.md).

## Project layout

`src/thai_offline_ai/` contains configuration, privacy/runtime preparation,
diagnostics, CLI/UI, and the adapters. `tests/` contains automated checks;
`scripts/` contains setup and diagnostic entry points. `models/` is reserved for
later phases and `outputs/` is the default local artifact directory.
