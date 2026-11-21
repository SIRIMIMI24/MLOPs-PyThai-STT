# Phase 7 — Conversation Intelligence

Analyzes a speaker-attributed transcript (Phase 5 / 5.5 output, or pasted `SPEAKER_00: ...` lines).

## What it produces
- **Always (deterministic, no LLM):** per-speaker turns, word count, talk seconds (only when real
  timestamps exist) and word share; top keywords; hits for an optional domain lexicon (YAML `terms`
  list, same format as Phase 3.7).
- **Optional (local Ollama LLM, strict JSON):** summary, topics, decisions, action items.

## Grounding and safety rules
- Every action item / decision must cite an utterance index `[n]`. Items citing a non-existent
  index are dropped and counted in `dropped_ungrounded`.
- Evidence text is copied from the transcript, never taken from the model.
- `owner_speaker` is kept only if that label actually appears in the transcript; otherwise null.
- Speaker labels stay anonymous. No business roles (Agent/Customer) are inferred.
- The transcript is wrapped in `<<<TRANSCRIPT>>>` delimiters and the prompt tells the model to
  treat it as untrusted data. The service has no tools, so injected text cannot trigger actions.
- Invalid LLM output gets up to 2 repair retries. If it still fails, the deterministic results are
  returned with `analysis_error` set (no fabricated summary).
- Prompts longer than 12,000 characters are truncated and flagged `truncated=true`
  (no chunked map-reduce yet, so the LLM summary of very long calls covers only the beginning).

## Usage
```powershell
# From transcribe-speakers output
.\.venv\Scripts\python.exe -m thai_offline_ai.cli transcribe-speakers call.wav > call.json
.\.venv\Scripts\python.exe -m thai_offline_ai.cli conversation-intel --transcript-json call.json

# Directly from audio (ASR + diarization first), or pasted text, deterministic only
.\.venv\Scripts\python.exe -m thai_offline_ai.cli conversation-intel call.wav
.\.venv\Scripts\python.exe -m thai_offline_ai.cli conversation-intel --text-file dialog.txt --no-llm --domain-lexicon terms.yaml
```
Reports are written to `outputs/conversation_intel/` as JSON and Markdown.
In the UI: **ASR** tab, accordion *Conversation Intelligence (Phase 7)*.
