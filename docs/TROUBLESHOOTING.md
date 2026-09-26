# Troubleshooting Notes

## Phase 1: Python launcher or broken virtual environment

Run `py -0p` and `py -3.10 --version`. If `python` reports a missing base
interpreter, check `Get-Command python -All`; it may point to an unrelated/broken
environment. Use the project `.venv\Scripts\python.exe` directly. An unregistered
Python 3.10 can be passed to `setup_windows.ps1 -PythonExe`.

A virtual environment depends on its base Python installation. If that Python
was moved/removed, restore it or rename the old `.venv` folder and rerun setup.
Do not delete a working environment just because another `python` is first on PATH.
If pip is missing, run `.\.venv\Scripts\python.exe -m ensurepip --upgrade`.

## Phase 1: PowerShell execution policy

Activation is optional: `.\.venv\Scripts\python.exe run_app.py` works directly.
If you want activation and it is blocked, change policy for this terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Organizational policy may override this. Use the executable directly in that case.
The documented setup command uses `-ExecutionPolicy Bypass` only for its child process.

## Phase 1: NLP dependency or output failure

`No module named pycrfsuite` means the sentence-tokenizer dependency is missing.
Activate the project environment and run `python -m pip install -e .`, followed
by `python -m pip check` and `python scripts/doctor.py`. The CRF model is already
bundled; no model download is required. A broken binary dependency should be
reinstalled in a clean Python 3.10 environment.

Output errors identify the configured directory. Choose a writable local folder
using `app.output_dir` or `$env:THAI_AI_OUTPUT_DIR = "outputs"`. A file, drive root,
or network share is not an appropriate output directory.

## Phase 1: configuration and local server

An explicitly selected YAML file must exist and contain valid named sections.
Use `config/settings.example.yaml` as the schema; quoted booleans/numbers and
unknown settings are rejected. `.env` is a reference format only; use YAML or
actual PowerShell environment variables.

If port 7860 is occupied, set `$env:THAI_AI_PORT = "7861"` and restart. The app
cannot bind to `0.0.0.0` or a LAN address. An offline guard error means some code
attempted external Python networking; check configuration and run doctor.
Install dependencies in a separate setup process. Do not disable privacy to
work around an unknown model download. Gradio 5 may emit deprecation warnings
about Gradio 6; this phase deliberately supports its verified 5.50 API.

TTS and ASR require their separately documented local model caches. Local LLM
requires an Ollama service and a locally installed configured model; none of these
components pull a model during normal application startup.

## PyThaiASR / Typhoon ASR compatibility
The Python package published on PyPI may lag behind the current GitHub repository. When a required API is missing, inspect the installed version first; then either install the current upstream repository in a dedicated environment or adapt to the verified API.

Do not silently swap to a cloud ASR service.

## Windows-specific ASR issue
If the chosen ASR backend depends on a stack that is not supported reliably on native Windows, use WSL2 or an isolated local ASR service. Keep communication on localhost and document the startup command.

## PyThaiTTS dependency/model issues
Use a backend/model that is verified with the currently installed PyThaiTTS release. Prefer a CPU-friendly ONNX model for the default path. If another voice/backend pulls large framework dependencies, make it optional.

## Model download vs offline runtime
First-run model downloads are setup-time network activity. After download, runtime should be tested with the internet disconnected.

## Local LLM issues
- Ollama mode accepts only `http(s)` loopback endpoints, normally
  `http://127.0.0.1:11434`; remote URLs are rejected before a request is sent.
- Run `ollama list` to check the configured model. If it is absent, deliberately
  run `ollama pull <model>` or `python scripts/setup_llm_model.py` during setup.
- If doctor reports the service unavailable, start the installed Ollama service and
  rerun doctor. Doctor lists models but never loads one or generates text.
- A connection timeout normally means the local model is too large for the machine
  or the request is long. Choose a smaller installed model or lower `llm.max_tokens`.
- `llama_cpp` is reserved for a future adapter; Phase 4 implements Ollama only.

## Speaker diarization issues
- Run `python scripts/setup_diarization_model.py` deliberately once. It places the
  PyThaiASR Nemotron int8 ONNX files under `models\pythaiasr\nemotron-3-diarization`.
  Doctor checks their presence but never loads them.
- `diarization.device=auto` uses CUDA only when `onnxruntime` exposes
  `CUDAExecutionProvider`. An NVIDIA GPU alone does not make CUDA usable. On this
  RTX 4050 Laptop GPU with 6 GB dedicated VRAM, CPU is the verified current path.
- `SPEAKER_00` and `SPEAKER_01` are anonymous local labels, not Agent/Customer.
  Single-speaker audio can test model execution but cannot validate speaker
  separation; use a consented non-sensitive two-speaker recording for that check.
- Overlapping speech and coarse ASR timestamps can leave an ambiguous whole ASR
  segment. The workbench records that condition and does not invent word timing.

## Audio format
Normalize ASR input to a supported mono PCM format and sample rate in a temporary file when needed. Delete temporary files after inference unless debug mode is explicitly enabled.
