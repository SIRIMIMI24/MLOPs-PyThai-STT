# Phase 8 Acceptance Criteria — JARVIS Personal AI Agent

## Core Engineering Criteria
- [x] Existing Phases 1–5.5 tabs, adapters, CLI commands, and 203 existing tests remain unbroken.
- [x] Offline Mode is the default; browser/web tools require explicit Online Agent Mode opt-in.
- [x] No private conversation history is sent to cloud AI APIs.
- [x] Playwright is lazily imported; missing Playwright displays an actionable setup command (`python scripts/setup_playwright.py`) without crashing the app.
- [x] Shared `ASRAdapter` and `TTSAdapter` instances are reused by the JARVIS Voice Pipeline (zero duplicate model loading).
- [x] Strict JSON schema validation runs before every tool invocation; arbitrary Python/shell/JS execution is prohibited.

## Required Demo Scenarios (Automated + Opt-In Live)

| Demo | Scenario | Verification Requirements |
| --- | --- | --- |
| **Demo 1** | Browser Search (`"ช่วยหาสถานที่ท่องเที่ยวเชียงใหม่ให้หน่อย"`) | Opens browser, runs search, reads source page, summarizes in Thai, returns clickable source URLs. |
| **Demo 2** | Multi-Step Planning (`"วางแผนเที่ยวเชียงใหม่ 3 วัน 2 คืน เน้นธรรมชาติและคาเฟ่"`) | Produces multi-step plan, researches multiple sources, generates day-by-day itinerary distinguishing verified facts from suggestions, cites URLs. |
| **Demo 3** | Conversation Continuity (`"เปลี่ยนวันที่สองให้เน้นดอยกับวิวภูเขา"`) | Uses prior session context to update Day 2 while preserving Day 1 and Day 3. |
| **Demo 4** | Voice-to-Browser | Push-to-talk audio -> ASR transcript -> Orchestrator -> Browser tools -> Local LLM response -> TTS audio output; visible status & timeline. |
| **Demo 5** | Security & Approval | Sensitive/interactive actions pause for explicit approval showing exact proposed action; denial blocks execution; webpage prompt injection cannot override safety rules or trigger unapproved actions. |

## Test Policy
- All unit and demo tests under `tests/agent/` run deterministically offline without downloading models, requiring a microphone, or hitting the public internet.
- Real live headed Chromium smoke test is opt-in (`@pytest.mark.integration`).
