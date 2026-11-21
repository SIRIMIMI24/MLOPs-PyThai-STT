# JARVIS Phase 8 Security Model

## 1. Threat Model & Trust Boundaries
1. **Trusted Local Core**: Python workbench process, validated `config/agent_config.yaml`, local Ollama endpoint (`127.0.0.1`), local SQLite memory store.
2. **Semi-Trusted LLM Output**: Local LLM tool calls are treated as untrusted proposals until parsed, schema-validated, URL-validated, and permission-checked.
3. **Untrusted External Web Content**: Every page title, body text, snippet, link, and DOM attribute extracted by the browser is treated as untrusted data that may contain **prompt injection**.

## 2. Three-Tier Permission System

| Level | Category | Examples | Policy |
| :---: | --- | --- | --- |
| **Level 1** | Read-Only | `get_current_datetime`, `memory_search`, `browser_get_tabs`, `browser_close`, `browser_scroll`, `browser_search`, `browser_open` (public HTTP/HTTPS), `browser_read_page` | Allowed automatically in the matching operating mode (Offline allows local-only tools; Online Agent Mode allows public read-only browser tools). |
| **Level 2** | Interactive / Persistent | `browser_click` on interactive/auth controls, `memory_save` (durable user preferences) | Requires explicit user approval before execution unless explicitly whitelisted for non-sensitive read navigation (`browser_click` defaults to requiring approval when targeting forms, logins, or interactive controls). |
| **Level 3** | Sensitive / Irreversible | Checkout, payment, booking/reservation, form submission, posting, sending messages, account settings, file deletion | **Always pauses** for explicit, action-specific user confirmation displaying tool name, arguments, target URL, and risk summary. Denial halts execution immediately. |

## 3. URL & Network Safety
- Only `http://` and `https://` URLs are permitted for browser navigation.
- Blocked schemes: `file://`, `javascript:`, `data:`, `chrome://`, `edge://`, `about:`, `ftp://`, `vbscript:`.
- Blocked targets (SSRF & local exposure protection):
  - Loopback (`127.0.0.0/8`, `::1`, `localhost`)
  - Private RFC1918 / ULA ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `fc00::/7`)
  - Link-local & cloud metadata (`169.254.0.0/16`, `fe80::/10`, `0.0.0.0/8`)
  - *(Note: Automated unit tests may explicitly pass `allow_local_test_urls=True` to test against a deterministic local test server; production config keeps this `False`.)*
- Dedicated Chromium automation profile (`outputs/jarvis_browser_profile/`) isolates cookies and history from the user's personal browser.

## 4. Prompt-Injection Defense
- All web text returned to the Local LLM is sanitized and wrapped in explicit untrusted delimiters:
  `<<<UNTRUSTED_WEB_CONTENT source="...">>> ... <<<END_UNTRUSTED_WEB_CONTENT>>>`
- The planner system prompt explicitly instructs the model to treat everything inside those delimiters strictly as data and never as system or user instructions.
- Suspicious instruction patterns (e.g., "ignore previous instructions", "system prompt", "execute payment/delete/click checkout") are flagged in observation metadata for audit visibility.
- Even if a local LLM is fooled by webpage text into emitting a sensitive tool call, the deterministic Python `PermissionGuard` intercepts and blocks/pauses it before execution.

## 5. Authentication & Anti-Bot Rules
- Never bypass login walls, paywalls, CAPTCHAs, Cloudflare/Turnstile challenges, or anti-bot gates.
- When a CAPTCHA or login wall is detected, the browser tool reports `blocked_reason="captcha_or_bot_gate"` or `"authentication_required"` so the user can handle it manually in the visible browser window.

## 6. Durable Memory Privacy
- `memory_save` requires explicit user approval and runs a credential/secret detector rejecting passwords, bearer tokens, API keys, credit-card-like strings, and cookies.
- Users can inspect, search, edit, and delete every saved memory record in the JARVIS UI.

## 7. Execution Controls & Auditability
- Every run enforces `max_steps`, per-tool timeout, and max parse/tool retries.
- An emergency **Stop / Cancel** flag can interrupt the orchestrator between steps and close/release browser work cleanly.
- Every tool call, decision, approval, denial, and source URL is recorded in the visible Tool Execution History and local task log.
