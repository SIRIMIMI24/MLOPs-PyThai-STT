# Acceptance Criteria

## P0 — must pass
- [x] `python run_app.py` starts the UI on localhost.
- [x] App does not load heavyweight models until the associated tab/action is used.
- [x] NLP analysis works with Thai text.
- [x] TTS adapter can synthesize a Thai WAV file when PyThaiTTS + model are installed.
- [x] ASR adapter can transcribe a local audio file when a supported PyThaiASR backend is installed.
- [x] Local LLM tab can connect to at least one verified local-only backend.
- [x] Missing optional backend does not crash the app.
- [x] Runtime network guard rejects non-localhost LLM URLs when `offline_runtime=true`.
- [x] Tests run without GPU, microphone, or model download.
- [x] README contains reproducible Windows setup steps.

## P1 — should pass
- [x] User can record audio in the browser UI and send it to ASR.
- [x] TTS model/voice choices are shown clearly (Lunarlist ONNX model and fixed 22050 Hz sample rate displayed; no multi-voice selector on this backend).
- [x] System tab reports Python version, OS, available backends, and configured model names.
- [x] Generated WAV files are stored under `outputs/` or configured directory.
- [x] Error messages include a specific remediation command when possible.

## P2 — later
- [x] True low-latency microphone streaming ASR (near-real-time CPU streaming ASR in Phase 3.5 + streaming speaker diarization in Phase 5.5).
- [x] Measured RTF/latency benchmark page (local file ASR: decode + normalization + inference, with JSON/CSV export).
- [x] One-click model manager (local cache inspector + explicit setup-mode child-process installer in UI and CLI).
- [x] Portable desktop packaging (`start_workbench.bat` one-click launcher + `build_portable_bundle` generator).
- [x] Local RAG / embeddings (100% offline SQLite + hybrid PyThaiNLP word/char n-gram TF-IDF retrieval, grounded Local LLM Q&A, and JARVIS `rag_search` tool).

## Test policy
Unit tests must monkeypatch optional model libraries. Integration tests that require real models should be opt-in and marked clearly.
