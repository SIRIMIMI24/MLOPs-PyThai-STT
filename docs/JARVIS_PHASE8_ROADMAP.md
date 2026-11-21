# Phase 8 Roadmap — JARVIS Personal AI Agent

This roadmap extends [`ASR_FUTURE_ROADMAP.md`](ASR_FUTURE_ROADMAP.md) without altering Phases 1–7.

## Phase 8 Subphases

| Subphase | Deliverable | Scope & Verification |
| --- | --- | --- |
| **8.1** | **Local Browser Agent** | Headed Playwright Chromium with dedicated automation profile, 7 browser tools (`open`, `search`, `read_page`, `click`, `scroll`, `get_tabs`, `close`), CAPTCHA/timeout handling, URL safety validation. |
| **8.2** | **AI Agent Orchestrator** | Bounded `UNDERSTAND -> PLAN -> SELECT TOOL -> EXECUTE -> OBSERVE -> REPLAN -> FINAL ANSWER` loop, strict JSON schema validation, structured-output repair for local Ollama models, cancellation & loop guards. |
| **8.3** | **Voice-Controlled JARVIS** | Push-to-talk `ASR -> Orchestrator -> TTS` pipeline reusing shared `ASRAdapter` and `TTSAdapter` instances (Thai + English) plus `WakeWordHook` extension point. |
| **8.4** | **Memory & Context** | Local SQLite store for multi-turn conversation continuity, task history, and user-approved editable/deletable preferences with secret scrubbing. |
| **8.5** | **Safety & Permission System** | 3-tier permission classifier (Read-only / Interactive / Sensitive), action-specific approval pause/resume, prompt-injection isolation, URL/SSRF guard, full execution logging. |
| **8.6** | **JARVIS UI & Demos** | New **JARVIS AI Assistant** tab in Gradio using the existing design system (Offline/Online toggle, chat, push-to-talk, status, timeline, tool log, browser controls, sources, approval dialog, memory viewer, stop button) + automated Demos 1–5. |

## Incremental Milestones

1. **Vertical Slice 1 (Text-to-Browser)**:
   `TEXT COMMAND -> LOCAL LLM -> BROWSER SEARCH -> SOURCE EXTRACTION -> ANSWER WITH CLICKABLE LINKS`
2. **Vertical Slice 2 (Voice-to-Browser)**:
   `VOICE COMMAND (Push-to-Talk) -> SHARED ASR -> ORCHESTRATOR -> BROWSER AGENT -> SHARED TTS RESPONSE`
3. **Safety & Continuity Hardening**:
   Multi-turn itinerary modification (Demo 3), approval pause/deny on sensitive actions (Demo 5), prompt-injection resistance, and SQLite memory management.

## Dependencies
- Core (already installed): `pydantic`, `sqlite3`, `requests`, `gradio`, `pythainlp`, `onnxruntime`, `pythaiasr`, `PyThaiTTS`, Kokoro ONNX, OpenThai 2.0 ONNX, Ollama.
- Optional Browser Automation extra (`pip install -e ".[agent]"` + `python scripts/setup_playwright.py`): `playwright>=1.50` + Chromium binary.
