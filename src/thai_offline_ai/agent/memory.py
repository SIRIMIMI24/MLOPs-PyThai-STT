"""Phase 8.4 — Local-first SQLite conversation, task history, and user-approved memory."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from thai_offline_ai.agent.permissions import contains_secret_material
from thai_offline_ai.errors import WorkbenchError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    key: str
    value: str
    category: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class TaskLogRecord:
    id: int
    session_id: str
    user_request: str
    mode: str
    status: str
    plan_summary: str
    tools_used: list[str]
    sources: list[dict[str, str]]
    created_at: str


class AgentMemoryStore:
    """Thread-safe local SQLite store for JARVIS conversation context and preferences."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'preference',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_summary TEXT NOT NULL,
                    tools_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_logs_session ON task_logs(session_id, id);
                """
            )

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content.strip(), _utc_now()),
            )

    def get_recent_messages(self, session_id: str, limit: int = 16) -> list[dict[str, str]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": str(r["role"]), "content": str(r["content"])} for r in rows]

    def clear_session(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    # ------------------------------------------------------------------
    # User-approved durable preferences / facts
    # ------------------------------------------------------------------
    def save_memory(self, key: str, value: str, category: str = "preference") -> MemoryRecord:
        clean_key = key.strip()
        clean_val = value.strip()
        if not clean_key or not clean_val:
            raise WorkbenchError("Memory key and value must not be empty.")
        if contains_secret_material(f"{clean_key} {clean_val}"):
            raise WorkbenchError(
                "Refused to store sensitive credentials, tokens, cookies, or passwords in memory."
            )
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (key, value, category, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = excluded.updated_at
                """,
                (clean_key, clean_val, category, now),
            )
            row = conn.execute(
                "SELECT id, key, value, category, updated_at FROM memories WHERE key = ?",
                (clean_key,),
            ).fetchone()
        return MemoryRecord(
            id=int(row["id"]),
            key=str(row["key"]),
            value=str(row["value"]),
            category=str(row["category"]),
            updated_at=str(row["updated_at"]),
        )

    def search_memories(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        term = f"%{query.strip()}%"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, key, value, category, updated_at
                FROM memories
                WHERE key LIKE ? OR value LIKE ? OR category LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (term, term, term, limit),
            ).fetchall()
        return [
            MemoryRecord(
                id=int(r["id"]),
                key=str(r["key"]),
                value=str(r["value"]),
                category=str(r["category"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]

    def list_memories(self, limit: int = 50) -> list[MemoryRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, key, value, category, updated_at
                FROM memories
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            MemoryRecord(
                id=int(r["id"]),
                key=str(r["key"]),
                value=str(r["value"]),
                category=str(r["category"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]

    def delete_memory(self, key: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key.strip(),))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Task execution logs
    # ------------------------------------------------------------------
    def record_task(
        self,
        *,
        session_id: str,
        user_request: str,
        mode: str,
        status: str,
        plan_summary: str,
        tools_used: list[str],
        sources: list[dict[str, Any]],
        max_logs: int = 50,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_logs (
                    session_id, user_request, mode, status, plan_summary,
                    tools_json, sources_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_request.strip(),
                    mode,
                    status,
                    plan_summary.strip(),
                    json.dumps(tools_used, ensure_ascii=False),
                    json.dumps(sources, ensure_ascii=False),
                    _utc_now(),
                ),
            )
            conn.execute(
                """
                DELETE FROM task_logs
                WHERE id NOT IN (
                    SELECT id FROM task_logs ORDER BY id DESC LIMIT ?
                )
                """,
                (max_logs,),
            )

    def list_task_logs(self, session_id: str | None = None, limit: int = 15) -> list[TaskLogRecord]:
        with self._lock, self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT * FROM task_logs
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM task_logs ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        results: list[TaskLogRecord] = []
        for r in rows:
            results.append(
                TaskLogRecord(
                    id=int(r["id"]),
                    session_id=str(r["session_id"]),
                    user_request=str(r["user_request"]),
                    mode=str(r["mode"]),
                    status=str(r["status"]),
                    plan_summary=str(r["plan_summary"]),
                    tools_used=json.loads(r["tools_json"] or "[]"),
                    sources=json.loads(r["sources_json"] or "[]"),
                    created_at=str(r["created_at"]),
                )
            )
        return results
