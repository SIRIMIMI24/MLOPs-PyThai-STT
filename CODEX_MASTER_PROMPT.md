# MASTER PROMPT FOR CODEX

Copy everything below into Codex as the first task after opening this repository.

---

You are the lead Python/AI engineer for this repository. Build the **Thai Offline AI Workbench** end to end from the existing project specification and starter code.

Before editing anything, read `AGENTS.md`, `docs/PROJECT_SPEC.md`, `docs/ARCHITECTURE.md`, `docs/ACCEPTANCE_CRITERIA.md`, and `docs/TROUBLESHOOTING.md`. Then inspect the whole repository and produce a short implementation plan in your task log; do not ask me to restate requirements that are already documented.

## Primary objective
Deliver a local, CPU-first Thai AI application with one Gradio UI containing:
1. Thai NLP using PyThaiNLP;
2. Thai TTS using PyThaiTTS, preferring a verified CPU/ONNX backend;
3. Thai ASR using PyThaiASR for local audio-file transcription, then microphone input;
4. Local open-weight LLM chat using a pluggable **local-only** backend such as Ollama localhost or llama.cpp;
5. a System tab showing backend readiness and configuration.

The application must keep user data on-device during runtime. Do not add cloud inference or remote API fallbacks. Internet may be used only for explicit setup/install/model download. In offline runtime mode, reject non-localhost LLM endpoints.

## Important dependency rule
Do **not** trust stale examples. Verify the exact APIs exposed by the versions you install. In particular, inspect PyThaiASR and PyThaiTTS before finalizing their adapters. If PyThaiASR's desired modern functionality is only available in the current upstream repository rather than the PyPI release, document that fact and choose the least fragile installation strategy. If a backend is not reliable on native Windows, isolate it behind the adapter and provide a WSL2/local-service path rather than breaking the entire app.

## Implementation requirements
- Python 3.10 baseline.
- Keep heavy imports lazy.
- Cache model instances.
- No model download during unit tests.
- Make optional backends fail gracefully with useful remediation instructions.
- Bind Gradio to `127.0.0.1` by default.
- Keep generated artifacts under the configured `outputs/` directory.
- Add type hints and concise docstrings.
- Keep the UI thin; inference belongs in adapters/services.
- Add or update tests for every adapter and network guard.
- Preserve a clean modular architecture so a future mobile/desktop frontend can reuse the same services.

## Deliverables
Implement/fix all code needed for the P0 acceptance criteria. Update docs and setup scripts so a new Windows user can clone the repo, create the environment, install the appropriate extras, download/configure models explicitly, and run the UI.

Also add a `scripts/doctor.py` diagnostic command that prints:
- Python version and platform;
- whether PyThaiNLP / PyThaiTTS / PyThaiASR / Gradio are importable;
- whether configured TTS/ASR/LLM backends are ready;
- whether the LLM URL is localhost-safe when offline mode is enabled;
- actionable install/fix suggestions.

## Validation
Run at minimum:

```bash
python -m compileall src tests run_app.py scripts
python -m pytest -q
```

Run any additional smoke checks that do not require multi-GB downloads. If you can safely install lightweight dependencies, run the real NLP smoke test. Do not pretend ASR/TTS/LLM inference passed unless you actually ran it with a real model.

## Final response format
When done, report:
1. what you implemented;
2. files changed;
3. test commands and exact pass/fail status;
4. what still requires a model download or OS-specific setup;
5. exact commands I should run next on Windows.

Do not stop at a design document—make the repository runnable and tested to the maximum extent possible in the environment.
