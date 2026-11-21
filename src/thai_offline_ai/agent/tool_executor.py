"""Phase 8.2/8.5 — Validated, permission-gated Tool Executor with timeout & error recovery."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from thai_offline_ai.agent.permissions import PermissionGuard
from thai_offline_ai.agent.schemas import ApprovalRequest, OperatingMode, ToolCall, ToolResult
from thai_offline_ai.agent.tool_registry import ToolRegistry
from thai_offline_ai.errors import WorkbenchError


class ToolExecutor:
    """Validates schemas, enforces 3-tier permissions, and executes authorized tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        guard: PermissionGuard,
        *,
        timeout_seconds: float = 25.0,
    ) -> None:
        self.registry = registry
        self.guard = guard
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        call: ToolCall,
        mode: OperatingMode,
        *,
        current_url: str | None = None,
        approved: bool = False,
    ) -> tuple[ToolResult | None, ApprovalRequest | None]:
        started = time.perf_counter()
        # 1. Strict schema validation first.
        try:
            validated_args = self.registry.validate_args(call.tool_name, call.arguments)
        except WorkbenchError as exc:
            return (
                ToolResult(
                    tool_name=call.tool_name,
                    ok=False,
                    summary=str(exc),
                    blocked_reason="schema_validation_error",
                    duration_seconds=time.perf_counter() - started,
                ),
                None,
            )

        clean_dict: dict[str, Any] = validated_args.model_dump()

        # 2. Deterministic permission & URL/mode safety evaluation.
        decision = self.guard.evaluate(
            call.tool_name,
            clean_dict,
            mode,
            current_url=current_url,
            approved=approved,
        )
        if decision.requires_approval and decision.approval_request is not None:
            return None, decision.approval_request
        if not decision.allowed:
            return (
                ToolResult(
                    tool_name=call.tool_name,
                    ok=False,
                    summary=decision.reason,
                    blocked_reason="permission_or_mode_blocked",
                    duration_seconds=time.perf_counter() - started,
                ),
                None,
            )

        # 3. Execute authorized handler with bounded timeout and friendly error recovery.
        tool = self.registry.get(call.tool_name)
        assert tool is not None
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool.handler, validated_args)
                result = future.result(timeout=self.timeout_seconds)
            result.duration_seconds = round(time.perf_counter() - started, 3)
            return result, None
        except FuturesTimeoutError:
            return (
                ToolResult(
                    tool_name=call.tool_name,
                    ok=False,
                    summary=(
                        f"Tool '{call.tool_name}' timed out after {self.timeout_seconds:.0f}s."
                    ),
                    blocked_reason="timeout",
                    duration_seconds=round(time.perf_counter() - started, 3),
                ),
                None,
            )
        except WorkbenchError as exc:
            return (
                ToolResult(
                    tool_name=call.tool_name,
                    ok=False,
                    summary=str(exc),
                    blocked_reason="tool_error",
                    duration_seconds=round(time.perf_counter() - started, 3),
                ),
                None,
            )
        except Exception as exc:  # noqa: BLE001
            return (
                ToolResult(
                    tool_name=call.tool_name,
                    ok=False,
                    summary=f"Tool '{call.tool_name}' failed: {type(exc).__name__}: {exc}",
                    blocked_reason="unexpected_error",
                    duration_seconds=round(time.perf_counter() - started, 3),
                ),
                None,
            )
