# Project Specification

## 1. Project name
**Thai Offline AI Workbench**

## 2. Goal
Create one local application that demonstrates a practical open-source Thai-language AI stack on a personal computer. It should let a user type Thai text, speak/upload Thai audio, generate Thai speech, run Thai NLP utilities, and optionally chat with a local open-weight LLM without sending runtime data to a cloud provider.

## 3. Target users
- Thai developers / data scientists evaluating open-source Thai AI.
- Business users who need a private local prototype.
- Students learning Thai NLP, ASR, TTS, and local LLM deployment.

## 4. MVP capabilities

### 4.1 NLP tab
Input: Thai text.
Output:
- normalized text;
- word tokens;
- sentence tokens;
- token count;
- Thai/Latin character summary;
- optional romanization if the installed PyThaiNLP API supports it reliably.

### 4.2 TTS tab
Input: Thai text and optional voice/model selection.
Output: locally generated WAV file playable in the UI.

Default target backend:
- PyThaiTTS
- CPU-friendly ONNX model when available (for example `lunarlist_onnx`).

### 4.3 ASR tab
Input: uploaded or microphone-recorded audio.
Output: Thai transcription.

Primary integration target:
- PyThaiASR.

MVP priority:
1. file transcription;
2. microphone capture through Gradio;
3. true streaming transcription only after the first two are stable.

### 4.4 Local LLM tab
Input: Thai prompt.
Output: response generated only by a model running on the local machine.

Backends should be pluggable:
- `ollama` localhost server;
- `llama_cpp` / llama.cpp-compatible local model;
- future adapters may support local Transformers.

Example model families (not mandatory downloads): Qwen, DeepSeek, Gemma, Typhoon, ThaiLLM-related open models.

## 5. Offline definition
There are two modes:

### Setup mode
Internet may be used to install Python packages and download model artifacts.

### Runtime mode
No external internet call is permitted. User audio, text, prompts, outputs, and metadata remain on-device. Localhost communication is allowed.

## 6. UX requirements
- One local Gradio app with tabs: `Home`, `NLP`, `TTS`, `ASR`, `Local LLM`, `System`.
- Thai labels first; English technical terms may be shown in parentheses.
- Show backend/model status in the System tab.
- If a backend is unavailable, explain exactly how to install/enable it.
- Do not crash the entire UI when one optional backend fails.

## 7. Performance requirements
MVP is CPU-first; performance depends on the model and hardware.
- UI startup without loading all models: target under 5 seconds on a typical modern PC.
- Heavy models load only on first use.
- Reuse loaded models after first inference.
- Do not claim "real time" unless measured real-time factor or streaming latency supports the claim on the user's machine.

## 8. Security/privacy requirements
- Bind to `127.0.0.1` by default, not `0.0.0.0`.
- No cloud API key is required for MVP.
- No analytics/telemetry added by this project.
- Do not log raw microphone audio or prompts by default.
- Generated output files may be saved only under a configurable local output directory.

## 9. Compatibility strategy
- Baseline Python: 3.10.
- Main app: Windows 11 / Linux / WSL2.
- macOS is a nice-to-have.
- ASR backends with OS-specific limitations must be isolated behind `ASRAdapter`.

## 10. Non-goals for the first milestone
- Mobile application packaging.
- Training/fine-tuning models.
- Voice cloning.
- Public web hosting.
- Enterprise authentication.
- Automatic model downloads during test runs.
