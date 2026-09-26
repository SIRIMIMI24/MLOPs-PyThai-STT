# Step-by-step Codex prompts

Use these only if you prefer several smaller tasks instead of the master prompt.

## Prompt 1 — Bootstrap and guardrails
Read `AGENTS.md` and all docs. Audit the starter repo. Make configuration, package structure, startup scripts, network-locality guard, logging, and tests robust. Do not implement heavyweight inference yet. Run tests and leave the repository in a clean runnable state.

## Prompt 2 — PyThaiNLP
Implement and verify the NLP adapter with the current PyThaiNLP API. Add normalization, word tokenization, sentence tokenization, counts, and safe optional romanization. Add tests using real PyThaiNLP if installed and graceful skip/error behavior otherwise. Wire it to the UI and CLI.

## Prompt 3 — PyThaiTTS
Inspect the current PyThaiTTS release and its verified CPU/ONNX model options. Implement lazy/cached TTS synthesis, output file management, voice/model configuration, error handling, UI, CLI, and mocked unit tests. Add an opt-in real-model smoke test/documented command, but do not auto-download models in normal tests.

## Prompt 4 — PyThaiASR
Inspect both the installed/PyPI PyThaiASR and current upstream repository before coding. Implement local audio-file transcription using the most reliable verified API. Normalize audio only when required. Add microphone-file input through Gradio. If native Windows support for the selected backend is fragile, document and implement an isolated WSL2/local-service fallback without replacing it with a cloud API.

## Prompt 5 — Local LLM
Implement a local-only LLM adapter. Support at least one tested localhost backend (prefer Ollama for easy setup) and keep the interface extensible for llama.cpp. In offline runtime mode, hard reject remote hosts. Add UI/CLI, connection status, and tests that mock localhost responses without internet.

## Prompt 6 — Integration and doctor
Create `scripts/doctor.py`, complete the System tab, run compile/tests, improve setup docs, and verify all P0 acceptance criteria. Clearly distinguish tested real inference from mocked/static validation.

## Prompt 7 — Streaming ASR (after MVP)
Implement true microphone streaming ASR only after the file-transcription path is stable. Measure latency/RTF; do not label it real-time unless the measurements support that claim. Keep a non-streaming fallback.
