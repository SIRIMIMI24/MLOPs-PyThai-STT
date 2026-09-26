# Project Context for Codex

If you can upload only one context file into a new Codex project, upload this file together with the repository/ZIP.

## What I want built
I want a local Thai AI demo/workbench inspired by the modern PyThaiNLP ecosystem: Thai NLP, offline Thai text-to-speech, offline Thai speech recognition, and an open-weight local LLM. It must be usable on a normal personal computer, CPU-first, private by default, and must not require a cloud API during normal runtime.

## Preferred stack
- Python 3.10 baseline
- PyThaiNLP for Thai text processing
- PyThaiTTS for Thai TTS; prefer a verified CPU/ONNX backend
- PyThaiASR for Thai ASR; verify current upstream API before implementation
- Gradio for a local UI
- local LLM through localhost (Ollama or llama.cpp), with no remote fallback

## UI
One app with tabs: Home, NLP, TTS, ASR, Local LLM, System.

## Critical constraints
1. Offline runtime after setup/model download.
2. Bind only to localhost by default.
3. CPU works; GPU optional.
4. Lazy-load heavy models and cache them.
5. Missing optional model/backend must show a useful error instead of crashing.
6. Tests must not download models or require microphone/GPU.
7. Verify third-party APIs rather than copying stale snippets.
8. If an ASR backend is problematic on native Windows, isolate it in WSL2/local service instead of switching to cloud.

## How Codex should work
Read `AGENTS.md` and all `docs/` files, then follow `CODEX_MASTER_PROMPT.md`. Implement code, run tests, update documentation, and report exactly what was truly tested.
