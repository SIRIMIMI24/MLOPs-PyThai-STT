# Architecture

## Phase 4 implementation boundary

Phase 3 exposes NLP, CPU-only TTS, and file-based CPU-only ASR through CLI and
Gradio. Phase 3.5 adds a separate near-real-time streaming service, without
changing `ASRResult` or the file adapter. Phase 3.6 adds a shared local audio boundary
and sequential batch processor. Phase 3.7 adds an evaluation layer that consumes
batch results and never alters inference. Phase 4 adds a loopback-only Ollama
intelligence boundary that does not consume ASR text implicitly. File ASR returns structured `ASRResult`
and `SpeechSegment` values; speaker remains `None` until a future diarization phase. See
[`ASR_FUTURE_ROADMAP.md`](ASR_FUTURE_ROADMAP.md) for the planned evaluation,
synthetic-data, and controlled adaptation governance.

Shared supporting modules preserve the existing boundaries:
- `config.py`: strict dataclass/YAML/environment validation, including local endpoints.
- `privacy.py`: telemetry/download opt-outs and an offline Python socket audit hook.
- `runtime.py`: validated output creation and shared runtime preparation.
- `diagnostics.py`: core import checks, bundled NLP smoke test, optional metadata checks.
- `errors.py`: actionable errors that avoid echoing private model inputs.
- `streaming.py`: session-specific PyThaiASR `RealtimeStreamASR` decoders and
  JSON-serializable `StreamingASREvent` updates.
- `audio.py`: local WAV/FLAC/MP3/OGG decoding, mono conversion, deterministic
  polyphase resampling, metadata preservation, and safe temporary WAV output.
- `batch.py`: sequential, failure-isolated ASR jobs with JSON/CSV exports below
  the configured output root.
- `evaluation.py`: manifest parsing, versioned normalization, CER/WER edit counts,
  error analysis, optional domain terms, grouped metrics, and local JSON/CSV reports.
- `adapters/llm.py` and `llm_types.py`: structured local chat contracts and one
  lazy Ollama `/api/chat` adapter; no cloud fallback or automatic model pull.

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
    def transcribe_file(self, audio_path: str) -> ASRResult: ...

class BatchASRProcessor:
    def transcribe_paths(self, paths) -> BatchASRResult: ...

class ASREvaluator:
    def evaluate(self, records, *, dataset_id: str, domain_terms=()) -> EvaluationReport: ...

class OllamaLocalLLM:
    def chat(self, messages: list[ChatMessage]) -> LLMResult: ...

class StreamingASRService:
    def start_session(self) -> str: ...
    def push_audio(self, session_id: str, samples, sample_rate: int) -> list[StreamingASREvent]: ...
    def finalize(self, session_id: str) -> list[StreamingASREvent]: ...
    def close_session(self, session_id: str) -> None: ...

class OllamaLocalLLM:
    def chat(self, messages: Sequence[ChatMessage]) -> LLMResult: ...
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

## Streaming boundary

PyThaiASR 2.2.0's public `stream_asr()` is an optional microphone capture helper;
it uses PyAudio and instantiates `RealtimeStreamASR`. The workbench uses the same
verified lower-level decoder directly so browser chunks and deterministic WAV
simulation share one service and do not require microphone hardware in tests.
The canonical stream input is mono 16 kHz float PCM. Other rates are actually
resampled and stereo is mixed to mono at the service boundary.

Each session owns a decoder and a bounded 256-event history. PyThaiASR maintains a
bounded rolling audio context itself; the workbench does not accumulate raw chunks.
The model is lazy-loaded at the first pushed chunk, cached for future sessions, and
serialised for inference because shared-model thread safety is undocumented.

`StreamingASREvent` has `session_id`, ordered `sequence`, `text`, `is_final`,
optional `start`, `end`, `speaker`, `confidence`, and metadata. The verified
decoder emits stable newly decoded fragments, not revisable partial hypotheses, so
Phase 3.5 records every emitted fragment as final and uses `finalize()` solely to
flush trailing context. No timestamps, confidence, or speaker identity are invented.
Future VAD can determine endpoints; Phase 5.5 can populate the existing speaker
field without changing the event schema. Future evaluation can retain raw events
for the same CER/WER/RTF workflow defined in the ASR roadmap.
