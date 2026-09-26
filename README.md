# Thai Offline AI Workbench — Phase 5

A local, CPU-first Thai NLP application using **PyThaiNLP** and **Gradio**.
The workbench supports Thai normalization, word and sentence tokenization, local
Thai TTS through PyThaiTTS Lunarlist ONNX, file ASR, and near-real-time local ASR.
Local LLM uses a configured Ollama model through a loopback-only endpoint.

No GPU, NVIDIA/CUDA, WSL, Docker, or API key is required. TTS and ASR models are
separate explicit setup downloads; normal runtime is cache-only.
Python **3.10 (64-bit)** is the primary supported version.

## Windows: fresh folder to working NLP UI

Open PowerShell in the project folder containing `run_app.py`.

### 1. Check Python

```powershell
py -0p
py -3.10 --version
```

If Python 3.10 is missing, install it during setup, then open a new PowerShell:

```powershell
winget install --exact --id Python.Python.3.10 --source winget
```

This is the official Python package identifier in the
[Microsoft WinGet repository](https://github.com/microsoft/winget-pkgs/tree/master/manifests/p/Python/Python/3/10).
If WinGet is unavailable, use the [Python Windows downloads](https://www.python.org/downloads/windows/)
and include the Python launcher. The setup script also accepts `-PythonExe` for
an existing Python 3.10 executable that is not registered with the launcher.

### 2. Set up the project

Internet access is needed **during this installation step**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

The script checks Python, creates or validates `.venv`, checks pip, installs
`.[dev]`, runs `pip check`, copies the example configuration only if the real
configuration is absent, and runs doctor. It stops on every critical failure.
It does not install optional inference extras or download large models.

For an unregistered Python executable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -PythonExe "C:\Path\To\Python310\python.exe"
```

Manual equivalent after verifying Python 3.10:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip check
if (-not (Test-Path -LiteralPath config/settings.yaml)) {
    Copy-Item -LiteralPath config/settings.example.yaml -Destination config/settings.yaml
}
python scripts/doctor.py
```

Use `python -m pip install -e .` if you do not need the test tools.

### 3. Run NLP

Activation is optional when using the virtual environment executable directly:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli nlp "ประเทศไทยมีประชากรประมาณเจ็ดสิบล้านคน"
.\.venv\Scripts\python.exe run_app.py
```

Open **http://127.0.0.1:7860**, select **NLP**, enter Thai text, and click
**วิเคราะห์ข้อความ**. Normalized text, word tokens, sentence tokens, and token
count appear separately. Stop the application with **Ctrl+C**.

Subsequent runs require only `.\.venv\Scripts\python.exe run_app.py`.

## Phase 2: local Thai TTS setup

Install the optional Phase 2 packages, then explicitly download the four Lunarlist
ONNX files once. Only the setup command needs internet; the app, doctor, and tests
never download a model. The default cache is `models\pythaitts\lunarlist_onnx` and
generated WAV files are saved under `outputs\tts`.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[tts,dev]"
.\.venv\Scripts\python.exe scripts\setup_tts_model.py
```

Set `tts.enabled: true` in `config/settings.yaml`, then run either:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli tts "สวัสดีครับ ยินดีต้อนรับสู่ระบบ AI ภาษาไทย"
.\.venv\Scripts\python.exe run_app.py
```

The TTS tab loads the CPU ONNX model only after **สร้างเสียง** is selected. A
missing local cache produces an actionable setup message and never falls back to a
remote inference service.

## Phase 3: local Thai ASR setup

Install the ASR package and explicitly retrieve the Typhoon ONNX files once:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[asr,dev]"
.\.venv\Scripts\python.exe scripts\setup_asr_model.py
```

The cache is stored under `models\pythaiasr\typhoon-asr-realtime`. File ASR accepts
WAV uploads, converts them to mono 16 kHz PCM in a temporary local folder, and
removes that temporary file after inference. Use `thai-offline-ai asr input.wav` or
the ASR tab. Results include `text`, `segments`, duration, backend, and model;
speaker and confidence remain `null` until a future diarization phase.

The browser upload workflow was manually verified with a local WAV and real
PyThaiASR transcription. Direct arbitrary local-path HTTP injection is intentionally
rejected by Gradio's upload-cache security policy; this does not affect normal browser
uploads. Phase 3 is file-based; Phase 3.5 adds a separate streaming path and does
not add diarization.

## Phase 3.5: near-real-time streaming ASR

Phase 3.5 keeps file transcription unchanged and adds a session-scoped live path
through the same local Typhoon CPU model. Gradio's **Live** controls accept microphone
chunks, normalize each chunk to mono 16 kHz PCM, and create a model only when the
first chunk arrives. A live session owns its decoder state and a bounded history of
256 events; the cached model is read-only and shared serially because PyThaiASR does
not document concurrent ONNX inference safety.

PyThaiASR 2.2.0's `RealtimeStreamASR` emits newly decoded text fragments rather than
revisable partial hypotheses. Each displayed fragment is therefore a finalized
incremental transcript segment, and **Stop** flushes only the remaining decoder
context. The UI labels its partial field accordingly and never repeatedly appends a
changing hypothesis. This is near-real-time CPU transcription, not a claim of hard
real-time latency; the simulator reports elapsed time for each actual machine.

Test the streaming core without microphone hardware by simulating a local WAV in
one-second chunks. Add `--realtime` to feed the chunks at their source duration:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli stream-asr outputs\tts\your-file.wav --realtime
.\.venv\Scripts\python.exe scripts\stream_asr_wav.py outputs\tts\your-file.wav --realtime
```

These commands and live inference use only the completed local ASR cache. They do
not fetch models or call cloud ASR services. Future VAD may replace controlled
chunking, and Phase 5.5 may add speaker attribution; Phase 3.5 events already carry
`speaker: null` and `confidence: null` unless a future verified backend supplies them.

### Phase 3.5.1 microphone validation

The local service records only process-local, monotonic timing metadata: audio-chunk
receipt, inference start/end, event emission, audio duration, inference duration, and
per-chunk RTF. Session metrics report processed audio, total inference, session RTF,
the latest chunk-processing latency, finalized event count, and session duration.
Browser capture latency is not included because the browser does not supply a reliable
server-visible capture timestamp. Timing metadata contains no microphone samples and
is discarded with the session.

To validate the actual Windows microphone: run `run_app.py`, open
`http://127.0.0.1:7860`, select **ASR**, select **เริ่ม Live transcription**, allow the
browser microphone permission for this loopback page, speak a short Thai sentence,
then select **หยุดและสรุปผล**. Confirm that the status changes from **Listening** to
**Processing complete** and then **Stopped**, that the finalized transcript updates
without duplicate text, and that a second Start/Stop session begins with no text from
the first. The Live status exposes the latest service latency and final session RTF;
it does not claim end-to-end capture latency.

## Phase 3.6: multi-format and batch ASR

Single-file ASR now accepts locally decoded **WAV, FLAC, MP3, and OGG** files. These
formats were verified with SoundFile 0.14.0 in this Windows environment. M4A is not
currently supported because an accessible local FFmpeg decoder was not available;
installing or downloading FFmpeg is never automatic and WAV support does not depend on it.

Every supported file is decoded locally, mixed to mono, and resampled with SciPy's
deterministic polyphase `resample_poly` implementation to 16 kHz PCM before it reaches
PyThaiASR. The result retains source format, source sample rate, channels, original
duration, normalized format, normalized sample rate, and normalized channels.

Use sequential batch transcription to share one lazy model safely and isolate invalid
files. It writes unique JSON and CSV exports below `outputs\batch`; JSON retains every
structured `ASRResult` and segment, while CSV provides one useful row per file.

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli asr call.mp3
.\.venv\Scripts\python.exe -m thai_offline_ai.cli asr-batch C:\Audio\Calls --recursive
```

The ASR tab retains **Single File** and **Live Streaming** and adds **Batch Files**.
The batch table exposes file name, status, duration, processing time, RTF, transcript
preview, and an error when one input cannot be processed. No model loads until a
single-file, batch, or live request actually reaches ASR.

## Phase 3.7: ASR Evaluation & Accuracy Lab

The evaluation lab measures one current local ASR backend against verified ground
truth from a UTF-8 CSV manifest. It does not train, fine-tune, select, or promote a
model. The manifest requires `sample_id`, `audio_path`, and `reference_text`; optional
columns such as `audio_condition`, `domain`, `source_format`, and `notes` are retained
for breakdowns. See [the dataset contract](datasets/evaluation/README.md) and the
tracked [example manifest](datasets/evaluation/manifest.example.csv).

Evaluation retains raw reference and ASR text separately. Version `thai-asr-eval-v1`
applies Unicode NFKC, PyThaiNLP normalization, whitespace collapse, and removal of
common ASCII/typographic punctuation only; it never corrects words. CER compares
normalized characters after removing spaces. Thai WER uses normalized text tokenized
with `pythainlp:newmm`; do not compare WER reports produced with another tokenizer.
Dataset CER/WER are calculated from the sum of substitutions, deletions, insertions,
and reference units, not by averaging individual percentages.

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli evaluate-asr C:\Evaluation\manifest.csv
.\.venv\Scripts\python.exe -m thai_offline_ai.cli evaluate-asr C:\Evaluation\manifest.csv --domain-lexicon C:\Evaluation\domain_terms.yaml
```

Reports include sample raw/normalized texts, CER/WER edit counts, RTF, optional
domain-term recall, metadata breakdowns, and common substitutions/deletions/insertions.
They are written locally to `outputs\evaluation` as JSON and CSV. Synthetic TTS audio
may validate the engineering path but must be labelled **SYNTHETIC ENGINEERING SMOKE
TEST**; it is never evidence of real-human ASR accuracy or a claim such as “95% accuracy.”

## Phase 4: Local LLM

Phase 4 supports the verified **Ollama** backend on a loopback endpoint only, normally
`http://127.0.0.1:11434`. The model is configured under `llm.model`; the workbench does
not start Ollama, pull a model, or contact the service at application startup. It sends
only structured local `system`, `user`, and `assistant` messages when the user explicitly
submits a chat request.

Install Ollama for Windows separately, then use an explicit pull command. The target
machine verified `scb10x/typhoon2.5-qwen3-4b:latest` as an already-installed 2.5 GB
local model; model suitability depends on your hardware and task.

```powershell
ollama list
ollama pull scb10x/typhoon2.5-qwen3-4b:latest
.\.venv\Scripts\python.exe scripts\setup_llm_model.py
```

Set `llm.enabled: true` and the intended model in `config/settings.yaml`, then use:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli llm "กรุณาตอบสั้น ๆ ว่า กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทยหรือไม่"
.\.venv\Scripts\python.exe run_app.py
```

The **Local LLM** tab provides a system prompt, session-local chat, send, and clear.
It reports backend/model/duration only; prompts and history are not persisted by the
workbench. Doctor checks executable, loopback service, and listed configured model
without generating text or pulling a model. A future llama.cpp adapter can implement
the same `ChatMessage -> LLMResult` boundary, but is not part of Phase 4.

## Phase 5: local speaker diarization

Phase 5 identifies anonymous speaker turns in a local audio file. It answers
**who spoke when**; ASR answers **what was said**. Labels such as `SPEAKER_00`
and `SPEAKER_01` are stable only within one diarization run and never mean
Agent or Customer. Role mapping, streaming diarization, and conversation analysis
are future work.

The verified backend is PyThaiASR 2.2.0 `nemotron-3-diarization` with the local
int8 ONNX files (`preprocessor_core.onnx`, `model.int8.onnx`, `constants.npz`),
about 104 MB total. It receives audio normalized locally to mono 16 kHz PCM, and
supports WAV, FLAC, MP3, and OGG through the existing file decoder.

```powershell
.\.venv\Scripts\python.exe scripts\setup_diarization_model.py
```

Set `diarization.enabled: true` in `config/settings.yaml`. The setup command is
the only diarization command allowed to retrieve model files; the app, doctor,
tests, ASR, batch ASR, and LLM do not download it.

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli diarize C:\Audio\meeting.wav
.\.venv\Scripts\python.exe -m thai_offline_ai.cli transcribe-speakers C:\Audio\meeting.wav
```

`transcribe-speakers` independently runs ASR and diarization, then assigns each
timestamped ASR segment to the speaker turn with greatest temporal overlap. A tie
uses earliest turn then speaker label and records ambiguity metadata; missing/no
overlap timestamps leave `speaker: null`. It never invents word timestamps or
splits text across a speaker boundary.

PyThaiASR 2.2.0 also exposes `asr_diarize(data, asr_model, diarize_model, ...)`,
which diarizes first and transcribes each returned turn. The workbench deliberately
does not make that combined upstream helper its public contract: independent
`ASRResult`, `DiarizationResult`, and composition preserve observable timestamps,
raw ASR output, and ambiguity metadata for later evaluation.

The current installed ONNX Runtime provides CPU execution only, so `device: auto`
selects CPU. The RTX 4050 Laptop GPU is detected with 6 GB dedicated VRAM, but its
presence alone is insufficient: CUDA is selected only if a compatible ONNX Runtime
CUDA provider is actually installed. Do not treat Windows shared GPU memory as
dedicated VRAM, and do not assume diarization and Ollama can share the GPU safely.

The ASR tab includes **Speaker Diarization** and **Transcribe with Speakers**
actions. Both remain lazy and do not load a model during UI startup. A real browser
upload of a consented, non-sensitive two-speaker recording was manually verified:
local diarization returned `SPEAKER_00` then `SPEAKER_01` with sensible timestamps
and boundaries, and speaker-attributed ASR segments completed without a UI crash.
This is a functional smoke result only. It does not identify Agent/Customer, claim
perfect speaker identity, or establish DER/JER accuracy.

ASR and diarization remain separate concerns: diarization answers **who spoke
when**, while ASR answers **what was said**. Recognition errors observed in that
browser sample are ASR errors and are evaluated separately by Phase 3.7; they are
not diarization failures. Future diarization evaluation needs consented recordings
with verified speaker-turn boundaries and should report DER (and optionally JER).

## CLI and configuration

The original `--text` syntax remains supported:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli nlp --text "สวัสดีครับ วันนี้อากาศดี"
.\.venv\Scripts\python.exe -m thai_offline_ai.cli --config config/settings.yaml nlp "ภาษาไทย"
.\.venv\Scripts\python.exe -m thai_offline_ai.cli doctor
```

NLP, TTS, file ASR, streaming-simulation, evaluation, and local-LLM commands print
readable JSON. The legacy `chat` command is reserved; use `llm` for local chat.

Configuration order: **defaults → YAML → environment variables**.
`config/settings.yaml` is optional; an explicitly supplied `--config` or
`THAI_AI_CONFIG` file must exist. Unknown fields, invalid types, unsupported
backends, duplicate YAML keys, unknown `THAI_AI_*` overrides, and unsafe endpoints
are errors. `.env.example` documents shell
variables; **`.env` is not loaded automatically**.

Relative configuration and output paths resolve from the current working
directory. Run the commands above from the project root. You can explicitly
configure another local output directory; network shares and drive roots are rejected.
Output creation/permissions are checked. Phase 1 does not save NLP inputs/results.

Examples of PowerShell overrides:

```powershell
$env:THAI_AI_OUTPUT_DIR = "outputs"
$env:THAI_AI_PORT = "7861"
$env:THAI_AI_OFFLINE_RUNTIME = "true"
$env:THAI_AI_LOG_LEVEL = "WARNING"
$env:THAI_AI_NLP_ENABLED = "true"
```

Other supported overrides: `THAI_AI_HOST`, `THAI_AI_CONFIG`,
`THAI_AI_NLP_BACKEND`, `THAI_AI_NLP_WORD_ENGINE`, `THAI_AI_NLP_SENTENCE_ENGINE`,
`THAI_AI_TTS_ENABLED`, `THAI_AI_TTS_BACKEND`, `THAI_AI_ASR_ENABLED`,
`THAI_AI_ASR_BACKEND`, `THAI_AI_LLM_ENABLED`, `THAI_AI_LLM_BACKEND`, and
`THAI_AI_LLM_BASE_URL`, `THAI_AI_LLM_MODEL`, and `THAI_AI_LLM_SYSTEM_PROMPT`.
Enabling LLM requires a locally installed Ollama model; doctor reports its separate
service and model-list readiness without performing generation.

The supported word engines are `newmm` and `newmm-safe`. Sentence engines are
`crfcut` (default), `whitespace`, and `whitespace+newline`. CRFcut uses a small
model bundled in the PyThaiNLP wheel. Its `python-crfsuite` dependency is included
in this project's core installation. No PyThaiNLP `full` extra is needed.

## Runtime privacy

- Gradio telemetry and version-check analytics are disabled before import and
  explicitly disabled on the app. Public sharing, SSR, MCP, and monitoring are off.
- UI binding and configured LLM URLs are restricted to loopback. Cloud LLM URLs
  are rejected even when offline mode is false. The LLM server is contacted only for
  an explicit chat request or a doctor readiness check.
- Offline runtime installs a process-wide Python audit hook rejecting external
  socket connections, DNS lookups, datagrams, and public socket binds. Redirects
  and proxy destinations must still pass this guard. Local startup checks bypass proxies.
- PyThaiNLP offline/read-only mode and Hugging Face offline/telemetry opt-outs are
  enforced. The supported NLP engines use installed resources only.
- System fonts avoid remote font requests. Browser response headers restrict
  scripts, styles, fonts, and connections to the local application.
- Input text is not written to application logs or files. Gradio temporary storage
  is configured below the output directory; audio uploads and microphone chunks are
  processed locally under the same loopback-only UI policy.

These safeguards apply to the application process, not pip or setup commands.
The Python socket guard is **not an OS firewall** and cannot police arbitrary
native code, child processes, browser extensions, or unrelated programs.
Keep `offline_runtime: true`; false disables installation of the socket guard in
a fresh process, but never enables cloud backends or optional inference. Once
installed, the guard cannot be removed until the process exits.
The workbench has no cloud fallback for local models.

## Validation and diagnostics

```powershell
.\.venv\Scripts\python.exe -m compileall src tests run_app.py scripts
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/doctor.py
```

Doctor uses `[OK]`, `[WARN]`, `[FAIL]`, and `[SKIP]`; exit 1 means a required
dependency, configuration, or output check failed. Optional packages are checked
through metadata/discovery only. Doctor imports core packages and runs a small
bundled-resource NLP smoke test; it never initializes TTS/ASR/LLM/GPU models.
The System tab uses lightweight metadata checks and does not claim inference readiness.

Tests block external Python networking. They exercise real NLP and briefly start
a local Gradio server, send an NLP HTTP request, then close it. No microphone,
GPU, optional inference package, internet access, or model download is required.

See [troubleshooting](docs/TROUBLESHOOTING.md),
[architecture](docs/ARCHITECTURE.md), and
[Phase 1 validation](docs/PHASE1_VALIDATION.md).

## Future ASR accuracy roadmap

Phase 3.7 establishes reproducible CER/WER/RTF evaluation on held-out real-human
audio. Future Phase 6.5 adds local synthetic speech and controlled pseudo-labeling
only after that baseline exists. Synthetic audio never enters the real-human
validation or final test sets, and candidates are promoted only after measurable
improvement on a fixed real holdout. See
[the future ASR roadmap](docs/ASR_FUTURE_ROADMAP.md).

## Project layout

`src/thai_offline_ai/` contains configuration, privacy/runtime preparation,
diagnostics, CLI/UI, and the adapters. `tests/` contains automated checks;
`scripts/` contains setup and diagnostic entry points. `models/` is reserved for
later phases and `outputs/` is the default local artifact directory.
