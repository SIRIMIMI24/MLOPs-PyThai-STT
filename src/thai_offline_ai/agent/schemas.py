"""Strict Pydantic schemas for Phase 8 JARVIS Agent tools, planning, and state."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OperatingMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class PermissionLevel(int, Enum):
    READ_ONLY = 1
    INTERACTIVE = 2
    SENSITIVE = 3


class BrowserOpenArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., min_length=4, max_length=2048, description="Public http(s) URL to open.")
    new_tab: bool = Field(default=False, description="Open in a new agent-controlled browser tab.")


class BrowserSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=400, description="Public web search query.")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum search results to return.")


class BrowserReadPageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(
        default=4000, ge=200, le=12000, description="Maximum readable characters to extract."
    )
    tab_index: int | None = Field(
        default=None, ge=0, le=15, description="Optional tab index to switch to before reading."
    )


class BrowserClickArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selector_or_text: str = Field(
        ..., min_length=1, max_length=300, description="Visible link/button text or CSS selector."
    )
    intent_description: str = Field(
        default="",
        max_length=300,
        description="Human-readable explanation of what clicking this element does.",
    )


class BrowserScrollArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["down", "up", "top", "bottom"] = Field(
        default="down", description="Scroll direction."
    )
    amount: int = Field(default=700, ge=100, le=4000, description="Vertical pixels to scroll.")


class BrowserGetTabsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BrowserCloseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetCurrentDatetimeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemorySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=300, description="Search keyword or phrase.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum memory items to return.")


class MemorySaveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=120, description="Short identifier for the preference.")
    value: str = Field(..., min_length=1, max_length=500, description="Preference or fact to remember.")
    category: Literal["preference", "fact", "task_context"] = Field(
        default="preference", description="Memory category."
    )


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=400)


class SourceLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=300)
    url: str = Field(..., min_length=4, max_length=2048)
    snippet: str = Field(default="", max_length=500)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[SourceLink] = Field(default_factory=list)
    untrusted_content: str = ""
    blocked_reason: str | None = None
    prompt_injection_warning: bool = False
    duration_seconds: float = 0.0


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    permission_level: PermissionLevel
    reason: str
    target_url: str | None = None
    action_summary: str


class TimelineStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int
    phase: str
    summary: str
    tool_name: str | None = None
    status: Literal["planned", "running", "completed", "awaiting_approval", "denied", "error", "cancelled"] = "completed"


class PlannerDecision(BaseModel):
    """Strict structured response expected from the local LLM planner at each step."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["tool_call", "final_answer"]
    plan_summary: str = Field(default="", max_length=500)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""
    cited_urls: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: Literal["completed", "awaiting_approval", "cancelled", "error"]
    mode: OperatingMode
    answer: str
    plan_summary: str = ""
    timeline: list[TimelineStep] = Field(default_factory=list)
    tool_history: list[ToolResult] = Field(default_factory=list)
    sources: list[SourceLink] = Field(default_factory=list)
    pending_approval: ApprovalRequest | None = None
    browser_active: bool = False
    browser_url: str | None = None
    duration_seconds: float = 0.0
