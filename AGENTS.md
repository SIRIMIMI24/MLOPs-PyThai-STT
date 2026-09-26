# AGENTS.md — Instructions for Codex

## Mission
Build and maintain a production-quality **Thai Offline AI Workbench** that can run locally on a normal personal computer and keeps user text/audio private by default.

## Read before coding
Always read these files first:
1. `docs/PROJECT_SPEC.md`
2. `docs/ARCHITECTURE.md`
3. `docs/ACCEPTANCE_CRITERIA.md`
4. `docs/TROUBLESHOOTING.md`

## Non-negotiable requirements
- Offline runtime is the default.
- No hidden calls to cloud APIs, telemetry endpoints, analytics endpoints, or hosted inference services.
- Localhost calls to local model servers are allowed.
- Model/dependency downloads may use the internet only during explicit setup.
- CPU-only operation must work for the MVP.
- GPU support is an optional acceleration path.
- Do not require NVIDIA/CUDA for the core app.
- Prefer Windows-native operation. If a dependency cannot reliably run on Windows, isolate that dependency behind an adapter and document a WSL2/local-service fallback.
- Never hard-code secrets, API keys, user paths, or model paths.
- Do not fabricate third-party APIs. Verify installed package APIs before implementation.

## Engineering rules
- Python baseline: 3.10 unless a verified dependency forces a different isolated environment.
- Use type hints for public functions.
- Use lazy imports for heavyweight/optional ML dependencies.
- Adapter failures must be converted into helpful user-facing messages, not cryptic tracebacks.
- Keep model loading cached so models are not reloaded on every request.
- Keep the UI thin; business/inference logic belongs in adapters/services.
- Add tests for each bug fixed.
- Do not download multi-GB models automatically when tests run.
- Tests must pass without requiring a microphone, GPU, or model download.

## Validation commands
Run these after changes:

```bash
python -m compileall src tests run_app.py
python -m pytest -q
```

If Ruff is installed:

```bash
ruff check .
```

## Definition of done
Before finishing a task:
- summarize files changed;
- list commands/tests run and their results;
- state which functionality was actually tested versus only statically validated;
- mention any OS-specific limitation;
- update relevant docs when behavior changes.

## Development sequence
Prefer this order:
1. project bootstrap and config;
2. PyThaiNLP adapter;
3. PyThaiTTS adapter;
4. PyThaiASR file transcription;
5. local LLM adapter;
6. Gradio UI;
7. microphone streaming ASR;
8. packaging / startup scripts / benchmarks.

## Scope discipline
Do not rewrite the entire project when a narrow change is enough. Preserve working behavior and interfaces unless the project spec explicitly requires a breaking change.
