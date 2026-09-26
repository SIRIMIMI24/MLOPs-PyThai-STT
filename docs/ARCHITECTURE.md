# Architecture

## Phase 2 implementation boundary

The diagram below remains the target architecture. Phase 2 exposes `NLPAdapter`
and the CPU-only `TTSAdapter` through CLI and Gradio. TTS uses the verified
PyThaiTTS 0.4.2 `lunarlist_onnx` backend and a configured local Hugging Face cache;
the adapter forces cache-only loading at runtime. ASR and LLM tabs remain placeholders.

Shared supporting modules preserve the existing boundaries:
- `config.py`: strict dataclass/YAML/environment validation, including local endpoints.
- `privacy.py`: telemetry/download opt-outs and an offline Python socket audit hook.
- `runtime.py`: validated output creation and shared runtime preparation.
- `diagnostics.py`: core import checks, bundled NLP smoke test, optional metadata checks.
- `errors.py`: actionable errors that avoid echoing private model inputs.

CLI, UI, and doctor load the same settings. Relative paths use the working
directory; explicit config paths must exist. `.env` is not automatically loaded.
NLP receives settings through `NLPAdapter.from_config`; its standalone constructor
remains usable. The adapter lazily caches imported functions and serializes use
of PyThaiNLP's shared CRF tagger. No separate service/framework was introduced.

The UI enforces loopback binding, no sharing, system fonts, and a browser
Content Security Policy. Offline Python socket protection is process-local and
does not claim to sandbox arbitrary native libraries or external programs.


## High-level flow

```text
                 ┌───────────────────────────┐
                 │       Gradio Local UI     │
                 │  127.0.0.1:7860 by default│
                 └──────────────┬────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
       ┌────────────┐     ┌────────────┐     ┌────────────┐
       │ NLPAdapter │     │ TTSAdapter │     │ ASRAdapter │
       └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
             │                  │                  │
             ▼                  ▼                  ▼
        PyThaiNLP          PyThaiTTS           PyThaiASR
                                │                  │
                                └──── local CPU ───┘

                       ┌────────────────┐
                       │   LLMAdapter   │
                       └───────┬────────┘
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
              Ollama localhost         llama.cpp
                   │                       │
                   └──── open-weight model┘
```

## Design principles

### Adapter boundary
Third-party model libraries change quickly. Keep every third-party package behind a small adapter so the UI and CLI remain stable.

### Lazy loading
Do not import/load ASR, TTS, or LLM weights during app import. Load the backend on the first request and cache it.

### Graceful degradation
The NLP tab should still work if TTS, ASR, or LLM packages are missing. The UI should report backend status instead of terminating.

### Local-only network policy
A local LLM adapter may call `http://127.0.0.1:*` or `http://localhost:*`. Any other network host is invalid in offline runtime mode.

## Proposed Python interfaces

```python
class NLPAdapter:
    def analyze(self, text: str) -> dict: ...

class TTSAdapter:
    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str: ...

class ASRAdapter:
    def transcribe_file(self, audio_path: str) -> str: ...

class LocalLLMAdapter:
    def chat(self, message: str, system_prompt: str | None = None) -> str: ...
```

## Configuration
Configuration order:
1. defaults in code;
2. optional YAML file;
3. environment variables override YAML.

Important fields:
- host / port;
- output directory;
- TTS model and voice;
- ASR model/device;
- LLM backend, base URL, model path/name;
- runtime offline flag.

## Separate-ASR-service fallback
If a desired ASR backend has Python or OS dependency conflicts, do not contaminate the main environment. Run that backend in a second local environment/service and call it through localhost. This preserves the offline/privacy requirement while keeping the main app maintainable.
