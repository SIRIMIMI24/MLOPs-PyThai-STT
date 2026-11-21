# Phase 8 Specification — JARVIS Personal AI Agent

## 1. Overview
Phase 8 extends **Thai Offline AI Workbench** into a local-first, bilingual (Thai + English) **JARVIS Personal AI Assistant** capable of natural voice conversations, multi-step task planning, visible local browser automation, local persistent memory, and permission-gated tool execution.

This phase is an **additive extension** of Phases 1–5.5. All existing adapters (`NLPAdapter`, `TTSAdapter`, `ASRAdapter`, `DiarizationAdapter`, `StreamingASRService`, `StreamingSpeakerASRService`, `OllamaLocalLLM`) and all existing Gradio tabs (`Home`, `NLP`, `TTS`, `ASR`, `Local LLM`, `System`) remain intact and are reused without duplicating loaded models in RAM/VRAM.

## 2. Operating Modes
- **Offline Mode (Default)**:
  - Uses local ASR, local TTS, local Ollama LLM, local SQLite memory, and local utility tools (`get_current_datetime`, `memory_search`, approved `memory_save`).
  - Blocks all browser web navigation and search tools with a clear user-facing message.
  - Preserves the repository's default offline privacy posture (`app.offline_runtime=true`).
- **Online Agent Mode (Explicit User Opt-In)**:
  - Enabled explicitly in the **JARVIS AI Assistant** workspace or CLI.
  - Allows the Playwright-controlled visible Chromium child process (isolated automation profile) to search public websites, open public pages, and extract readable content.
  - **Never** sends private conversation history to external cloud AI APIs; reasoning stays 100% on the local Ollama LLM.
  - If the network is unreachable or a site blocks automation / presents a CAPTCHA, the agent reports the condition transparently and never fabricates live search results.

## 3. Core Subsystems

### 3.1 Phase 8.1 — Local Browser Agent (`thai_offline_ai.agent.browser`)
- Powered by Playwright Python Async API (`playwright.async_api`), imported lazily so missing Playwright never breaks the workbench.
- Launches **headed Chromium** (`headless=false` by default) with a dedicated automation profile under `outputs/jarvis_browser_profile/` (never attaches to the user's personal browser profile).
- Tools:
  - `browser_open(url)`
  - `browser_search(query, max_results=5)`
  - `browser_read_page(max_chars=4000)`
  - `browser_click(selector_or_text)`
  - `browser_scroll(direction="down", amount=700)`
  - `browser_get_tabs()`
  - `browser_close()`
- Detects timeouts, unreachable hosts, consent overlays, and CAPTCHA / bot-verification gates gracefully; never attempts to bypass authentication or CAPTCHAs.

### 3.2 Phase 8.2 — AI Agent Orchestrator (`thai_offline_ai.agent`)
- Bounded execution loop:
  `USER REQUEST -> UNDERSTAND -> PLAN -> SELECT TOOL -> EXECUTE -> OBSERVE -> REPLAN -> FINAL ANSWER`
- Works reliably with local 4B–8B Ollama models (e.g., `scb10x/typhoon2.5-qwen3-4b:latest`) via a structured JSON protocol with strict schema validation, `<think>`/code-fence stripping, and controlled repair retries.
- Safeguards: `max_steps`, per-tool timeout, consecutive error limit, cancellation / emergency stop flag, and pause/resume for user approvals.
- Utility & Memory tools:
  - `get_current_datetime()`
  - `memory_search(query, limit=5)`
  - `memory_save(key, value, category="preference")` *(requires user approval)*

### 3.3 Phase 8.3 — Voice-Controlled JARVIS (`thai_offline_ai.agent.voice`)
- Pipeline: `MICROPHONE (Push-to-Talk) -> Shared ASRAdapter -> AgentOrchestrator -> Browser/Local Tools -> Local LLM -> Shared TTSAdapter -> AUDIO OUTPUT`.
- Supports Thai, English, and mixed Thai-English input and spoken replies.
- Reuses the already-instantiated `ASRAdapter` and `TTSAdapter` objects from the UI/workbench to protect 6 GB VRAM / RAM budgets.
- Includes an extensible `WakeWordHook` protocol reserved for future wake-word detection without running background audio capture by default.

### 3.4 Phase 8.4 — Memory & Context (`thai_offline_ai.agent.memory`)
- Local SQLite database (`outputs/jarvis_memory.db`) storing:
  1. Conversation sessions and multi-turn messages.
  2. Recent task execution logs (plan summary, tool trace, sources).
  3. User-approved durable preferences/facts (viewable, editable, deletable in UI).
- Automatically rejects or scrubs credentials/secrets (passwords, API keys, tokens, cookies) from durable memory persistence.

### 3.5 Phase 8.5 — Browser Safety & Permission System (`thai_offline_ai.agent.permissions`)
- Three-tier permission model (Level 1 Read-Only, Level 2 Interactive, Level 3 Sensitive) documented in [`JARVIS_SECURITY_MODEL.md`](JARVIS_SECURITY_MODEL.md).
- Prompt-injection isolation wrapping all external webpage observations in untrusted-content delimiters.
- Strict URL validation blocking unsafe schemes and private/loopback/metadata IP ranges by default.

## 4. Hardware Optimization (RTX 4050 6 GB VRAM / 64 GB RAM)
- Ollama LLM runs locally via localhost HTTP (`127.0.0.1:11434`).
- ONNX ASR (OpenThai 2.0 INT8) and ONNX TTS (Kokoro Thai) run on CPU so GPU VRAM remains dedicated to the local LLM.
- Single shared adapter instances prevent loading duplicate weights.
