"""Phase 8.5 — Permission classifier, URL/SSRF guard, prompt-injection & secret protection."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from thai_offline_ai.agent.config import SafetyConfig
from thai_offline_ai.agent.schemas import ApprovalRequest, OperatingMode, PermissionLevel

_SENSITIVE_ACTION_PATTERNS = re.compile(
    r"(pay|payment|checkout|purchase|buy\s*now|order|confirm\s*booking|reserve|book\s*now|"
    r"submit|delete|remove|post\s*comment|send\s*message|transfer|withdraw|login|sign\s*in|"
    r"ชำระเงิน|จ่ายเงิน|สั่งซื้อ|ยืนยันการจอง|ลบข้อมูล|โพสต์|ส่งข้อความ|เข้าสู่ระบบ)",
    re.IGNORECASE,
)

_PROMPT_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions|"
    r"disregard\s+.*(rules|instructions|policy)|"
    r"system\s*prompt|you\s+are\s+now\s+|developer\s+mode|"
    r"override\s+safety|bypass\s+approval|call\s+tool\s+browser_click|"
    r"เพิกเฉยต่อคำสั่งก่อนหน้า|ข้ามการอนุมัติ)",
    re.IGNORECASE,
)

_SECRET_PATTERNS = re.compile(
    r"(password\s*[:=]|passwd|api[_-]?key\s*[:=]|secret[_-]?key|bearer\s+[a-z0-9._-]{12,}|"
    r"sk-[a-z0-9]{16,}|ghp_[a-z0-9]{20,}|cookie\s*[:=]|session[_-]?token|"
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)",
    re.IGNORECASE,
)

_BROWSER_ONLINE_TOOLS = {
    "browser_open",
    "browser_search",
    "browser_read_page",
    "browser_click",
    "browser_scroll",
}


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    requires_approval: bool
    level: PermissionLevel
    reason: str
    approval_request: ApprovalRequest | None = None


def validate_browser_url(url: str, safety: SafetyConfig) -> tuple[bool, str]:
    """Validate URL scheme and block private/loopback/metadata targets by default."""
    cleaned = (url or "").strip()
    if not cleaned or any(ord(ch) < 32 for ch in cleaned):
        return False, "URL is empty or contains invalid control characters."
    try:
        parsed = urlsplit(cleaned)
    except ValueError:
        return False, "URL syntax is invalid."
    scheme = (parsed.scheme or "").lower()
    if scheme in safety.blocked_schemes or scheme not in {"http", "https"}:
        return False, f"Blocked URL scheme '{scheme or 'empty'}'; only http/https are allowed."
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False, "URL must contain a valid hostname."
    if parsed.username is not None or parsed.password is not None:
        return False, "URLs containing embedded credentials are blocked."
    if safety.allow_local_test_urls and hostname in {"127.0.0.1", "localhost", "::1"}:
        return True, "Allowed local test URL (explicit test override)."
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        return False, f"Blocked local/internal hostname '{hostname}'."
    try:
        ip = ipaddress.ip_address(hostname)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"Blocked non-public IP address '{hostname}'."
    except ValueError:
        # Regular domain name; ensure it is not a bare single-label internal host without a dot.
        if "." not in hostname:
            return False, f"Blocked single-label internal host '{hostname}'."
    return True, "OK"


def contains_secret_material(text: str) -> bool:
    """Detect passwords, API keys, cookies, or card-like strings before memory persistence."""
    return bool(_SECRET_PATTERNS.search(text or ""))


def detect_prompt_injection(text: str) -> bool:
    """Flag suspicious prompt-injection directives embedded inside external webpage content."""
    return bool(_PROMPT_INJECTION_PATTERNS.search(text or ""))


def sanitize_untrusted_web_content(text: str, source_url: str, max_chars: int = 4500) -> tuple[str, bool]:
    """Wrap external webpage text in untrusted delimiters and flag prompt-injection attempts."""
    raw = (text or "").replace("\x00", "").strip()
    if len(raw) > max_chars:
        raw = raw[:max_chars].rstrip() + "\n...[truncated]"
    warning = detect_prompt_injection(raw)
    # Neutralize attempts to close our delimiter prematurely.
    safe_body = raw.replace("<<<END_UNTRUSTED_WEB_CONTENT>>>", "[filtered-marker]")
    safe_url = (source_url or "unknown").replace("\n", " ").replace(">", "")
    wrapped = (
        f'<<<UNTRUSTED_WEB_CONTENT source="{safe_url}">>>\n'
        f"{safe_body}\n"
        "<<<END_UNTRUSTED_WEB_CONTENT>>>"
    )
    return wrapped, warning


class PermissionGuard:
    """Deterministic 3-tier safety gate for every JARVIS tool call."""

    def __init__(self, safety: SafetyConfig) -> None:
        self.safety = safety

    def classify_tool(
        self, tool_name: str, arguments: dict[str, Any], current_url: str | None = None
    ) -> tuple[PermissionLevel, str]:
        if tool_name in {
            "get_current_datetime",
            "memory_search",
            "rag_search",
            "browser_get_tabs",
            "browser_close",
            "browser_scroll",
            "browser_search",
            "browser_read_page",
        }:
            return PermissionLevel.READ_ONLY, "Read-only observation or session control."

        if tool_name == "browser_open":
            url = str(arguments.get("url", ""))
            if _SENSITIVE_ACTION_PATTERNS.search(url):
                return (
                    PermissionLevel.SENSITIVE,
                    f"Target URL appears to involve a sensitive/authenticated workflow: {url}",
                )
            return PermissionLevel.READ_ONLY, "Open a public webpage for read-only research."

        if tool_name == "memory_save":
            return (
                PermissionLevel.INTERACTIVE,
                "Persisting durable personal information/preference to local SQLite memory.",
            )

        if tool_name == "browser_click":
            combined = " ".join(
                [
                    str(arguments.get("selector_or_text", "")),
                    str(arguments.get("intent_description", "")),
                    str(current_url or ""),
                ]
            )
            if _SENSITIVE_ACTION_PATTERNS.search(combined):
                return (
                    PermissionLevel.SENSITIVE,
                    "Click target matches a sensitive or irreversible action (payment/booking/submit/login/delete).",
                )
            return (
                PermissionLevel.INTERACTIVE,
                "Interactive page click may change website state or open forms.",
            )

        return PermissionLevel.SENSITIVE, f"Unknown or unclassified tool '{tool_name}'."

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        mode: OperatingMode,
        *,
        current_url: str | None = None,
        approved: bool = False,
    ) -> PermissionDecision:
        # 1. Enforce Offline vs Online mode boundary first.
        if mode == OperatingMode.OFFLINE and tool_name in _BROWSER_ONLINE_TOOLS:
            return PermissionDecision(
                allowed=False,
                requires_approval=False,
                level=PermissionLevel.READ_ONLY,
                reason=(
                    "Offline Mode is active: browser web tools are disabled. "
                    "Switch JARVIS to Online Agent Mode to allow public web research."
                ),
            )

        # 2. Enforce URL safety for browser_open.
        target_url: str | None = current_url
        if tool_name == "browser_open":
            target_url = str(arguments.get("url", "")).strip()
            ok, msg = validate_browser_url(target_url, self.safety)
            if not ok:
                return PermissionDecision(
                    allowed=False,
                    requires_approval=False,
                    level=PermissionLevel.SENSITIVE,
                    reason=f"URL blocked by safety policy: {msg}",
                )

        # 3. Block secrets in memory_save before even asking approval.
        if tool_name == "memory_save":
            combined = f"{arguments.get('key', '')} {arguments.get('value', '')}"
            if contains_secret_material(combined):
                return PermissionDecision(
                    allowed=False,
                    requires_approval=False,
                    level=PermissionLevel.SENSITIVE,
                    reason=(
                        "Blocked: JARVIS refuses to save passwords, tokens, cookies, "
                        "or payment credentials into durable memory."
                    ),
                )

        # 4. Classify permission tier and check approval requirements.
        level, reason = self.classify_tool(tool_name, arguments, current_url=current_url)
        needs_approval = (
            (level == PermissionLevel.SENSITIVE and self.safety.require_approval_for_sensitive)
            or (tool_name == "memory_save" and self.safety.require_approval_for_memory_save)
            or (
                level == PermissionLevel.INTERACTIVE
                and self.safety.require_approval_for_interactive
            )
        )

        if needs_approval and not approved:
            summary = self._format_action_summary(tool_name, arguments, target_url)
            req = ApprovalRequest(
                approval_id=f"apr-{uuid4().hex[:10]}",
                tool_name=tool_name,
                arguments=dict(arguments),
                permission_level=level,
                reason=reason,
                target_url=target_url,
                action_summary=summary,
            )
            return PermissionDecision(
                allowed=False,
                requires_approval=True,
                level=level,
                reason=reason,
                approval_request=req,
            )

        return PermissionDecision(
            allowed=True,
            requires_approval=False,
            level=level,
            reason=reason,
        )

    @staticmethod
    def _format_action_summary(
        tool_name: str, arguments: dict[str, Any], target_url: str | None
    ) -> str:
        if tool_name == "memory_save":
            return (
                f"Save personal memory [{arguments.get('category', 'preference')}] "
                f"'{arguments.get('key', '')}' = '{arguments.get('value', '')}'"
            )
        if tool_name == "browser_click":
            target = arguments.get("selector_or_text", "")
            intent = arguments.get("intent_description", "")
            url_part = f" on {target_url}" if target_url else ""
            intent_part = f" ({intent})" if intent else ""
            return f"Click '{target}'{url_part}{intent_part}"
        if tool_name == "browser_open":
            return f"Open sensitive/authenticated URL: {arguments.get('url', '')}"
        return f"Execute {tool_name} with arguments {arguments}"
