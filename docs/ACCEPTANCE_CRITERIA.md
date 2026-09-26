# Acceptance Criteria

## P0 — must pass
- [ ] `python run_app.py` starts the UI on localhost.
- [ ] App does not load heavyweight models until the associated tab/action is used.
- [ ] NLP analysis works with Thai text.
- [ ] TTS adapter can synthesize a Thai WAV file when PyThaiTTS + model are installed.
- [ ] ASR adapter can transcribe a local audio file when a supported PyThaiASR backend is installed.
- [ ] Local LLM tab can connect to at least one verified local-only backend.
- [ ] Missing optional backend does not crash the app.
- [ ] Runtime network guard rejects non-localhost LLM URLs when `offline_runtime=true`.
- [ ] Tests run without GPU, microphone, or model download.
- [ ] README contains reproducible Windows setup steps.

## P1 — should pass
- [ ] User can record audio in the browser UI and send it to ASR.
- [ ] TTS model/voice choices are shown clearly.
- [ ] System tab reports Python version, OS, available backends, and configured model names.
- [ ] Generated WAV files are stored under `outputs/` or configured directory.
- [ ] Error messages include a specific remediation command when possible.

## P2 — later
- [ ] True low-latency microphone streaming ASR.
- [ ] Measured RTF/latency benchmark page.
- [ ] One-click model manager.
- [ ] Portable desktop packaging.
- [ ] Local RAG / embeddings.

## Test policy
Unit tests must monkeypatch optional model libraries. Integration tests that require real models should be opt-in and marked clearly.
