# Phase 6 — Local Voice Assistant (100% Offline Voice-to-Voice)

## Overview
Phase 6 implements a direct, low-overhead, **100% offline** voice-to-voice conversation pipeline:

```text
Microphone / Local Audio -> Shared ASRAdapter -> OllamaLocalLLM (127.0.0.1) -> Shared TTSAdapter -> Spoken WAV Reply
```

Unlike Phase 8 (JARVIS), which adds multi-step tool planning and optional browser automation, Phase 6 communicates directly with the local Ollama chat endpoint with zero web access and zero tool-planning overhead — minimizing end-to-end latency for private Thai and Thai-English voice conversations.

## Key Design Decisions
- **Shared Adapter Reuse**: `LocalVoiceAssistantService` accepts the existing `ASRAdapter` and `TTSAdapter` instances from the Gradio UI, ensuring ONNX models are never loaded twice.
- **Measured Latency Telemetry**: Every turn records genuine wall-clock durations (`asr_seconds`, `llm_seconds`, `tts_seconds`, `total_seconds`) and reports them in both the UI latency card and CLI JSON output.
- **Graceful Degradation**: If TTS is disabled or fails, the user still receives the transcript and Local LLM response along with a clear `tts_error` notice.
- **Privacy & Session-Local History**: Conversation context is kept strictly in memory (bounded to `max_history_messages`), never written to disk or sent off-device.
- **Speech Text Normalization**: Shared `prepare_speech_text()` strips Markdown formatting and raw URLs so Kokoro Thai ONNX / Lunarlist ONNX reads natural Thai + English replies cleanly.

## Usage

### UI
Open the **Local LLM** tab and expand **Local Voice Assistant (Phase 6 · 100% Offline Voice-to-Voice)**:
1. Record from microphone or upload a local WAV/FLAC/MP3/OGG file.
2. Click **พูดคุยด้วยเสียง (Voice Chat)**.
3. Review the live transcript, conversation bubbles, measured `ASR / LLM / TTS / Total` timings, and play the synthesized Thai/English voice response.

### CLI
```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli voice-assistant sample.wav
.\.venv\Scripts\python.exe -m thai_offline_ai.cli voice-assistant sample.wav --model openthai2_asr --prompt th-mix --voice f_young_bright --speed 1.05
```
