"""Phase 8.2 — Strict Tool Registry with Pydantic schema validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from thai_offline_ai.agent.memory import AgentMemoryStore
from thai_offline_ai.agent.schemas import (
    GetCurrentDatetimeArgs,
    MemorySaveArgs,
    MemorySearchArgs,
    OperatingMode,
    ToolResult,
)
from thai_offline_ai.errors import WorkbenchError


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], ToolResult]
    requires_online: bool = False


class ToolRegistry:
    """Authoritative catalog of authorized JARVIS tools and JSON argument schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[[Any], ToolResult],
        requires_online: bool = False,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            args_model=args_model,
            handler=handler,
            requires_online=requires_online,
        )

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self, mode: OperatingMode) -> list[RegisteredTool]:
        return [
            tool
            for tool in self._tools.values()
            if mode == OperatingMode.ONLINE or not tool.requires_online
        ]

    def validate_args(self, tool_name: str, raw_args: dict[str, Any]) -> BaseModel:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise WorkbenchError(f"Unknown or unauthorized tool: '{tool_name}'.")
        if not isinstance(raw_args, dict):
            raise WorkbenchError(f"Arguments for '{tool_name}' must be a JSON object.")
        try:
            return tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            msg = first.get("msg", "invalid value")
            raise WorkbenchError(f"Invalid arguments for '{tool_name}' ({loc}): {msg}") from exc

    def describe_for_prompt(self, mode: OperatingMode) -> str:
        """Compact JSON schema catalog injected into the local LLM planner prompt."""
        lines: list[str] = []
        for tool in self.list_tools(mode):
            schema = tool.args_model.model_json_schema()
            props = schema.get("properties", {})
            required = schema.get("required", [])
            compact_props = {
                k: {
                    "type": v.get("type", "any"),
                    **({"enum": v["enum"]} if "enum" in v else {}),
                    **({"default": v["default"]} if "default" in v else {}),
                }
                for k, v in props.items()
            }
            lines.append(
                f"- {tool.name}: {tool.description}\n"
                f"  args={json.dumps(compact_props, ensure_ascii=False)} required={required}"
            )
        return "\n".join(lines)


def register_core_local_tools(registry: ToolRegistry, memory: AgentMemoryStore) -> None:
    """Register local-only utility & SQLite memory tools."""

    def handle_datetime(_args: GetCurrentDatetimeArgs) -> ToolResult:
        now = datetime.now().astimezone()
        iso = now.isoformat(timespec="seconds")
        readable = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        return ToolResult(
            tool_name="get_current_datetime",
            ok=True,
            summary=f"Current local date/time: {readable}",
            data={"iso": iso, "readable": readable, "weekday": now.strftime("%A")},
        )

    def handle_memory_search(args: MemorySearchArgs) -> ToolResult:
        records = memory.search_memories(args.query, limit=args.limit)
        items = [
            {
                "key": r.key,
                "value": r.value,
                "category": r.category,
                "updated_at": r.updated_at,
            }
            for r in records
        ]
        summary = (
            f"Found {len(items)} saved memory item(s) matching '{args.query}'."
            if items
            else f"No saved memory matched '{args.query}'."
        )
        return ToolResult(
            tool_name="memory_search",
            ok=True,
            summary=summary,
            data={"memories": items},
        )

    def handle_memory_save(args: MemorySaveArgs) -> ToolResult:
        rec = memory.save_memory(args.key, args.value, category=args.category)
        return ToolResult(
            tool_name="memory_save",
            ok=True,
            summary=f"Saved preference '{rec.key}' ({rec.category}).",
            data={"key": rec.key, "value": rec.value, "category": rec.category},
        )

    registry.register(
        name="get_current_datetime",
        description="Get the current local date, time, and day of week on the user's computer.",
        args_model=GetCurrentDatetimeArgs,
        handler=handle_datetime,
        requires_online=False,
    )
    registry.register(
        name="memory_search",
        description="Search user-approved saved preferences and facts in local SQLite memory.",
        args_model=MemorySearchArgs,
        handler=handle_memory_search,
        requires_online=False,
    )
    registry.register(
        name="memory_save",
        description="Save a durable user preference or non-sensitive fact to local SQLite memory (requires user approval).",
        args_model=MemorySaveArgs,
        handler=handle_memory_save,
        requires_online=False,
    )
