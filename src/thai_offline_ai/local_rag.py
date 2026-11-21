"""100% Offline Local RAG & Hybrid Thai-English Document Retrieval.

Stores ingested documents and chunks in a local SQLite database and ranks matches
using a deterministic hybrid of PyThaiNLP word-token BM25 and character n-gram
TF-IDF cosine similarity (requires zero external embedding model download and
zero network access). Optionally synthesizes grounded answers with citations via
the local Ollama LLM.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.evaluation import normalize_text, thai_tokens
from thai_offline_ai.llm_types import ChatMessage, LLMResult

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".csv", ".json", ".html", ".htm"})
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RAGLLMClient(Protocol):
    def chat(self, messages: Sequence[ChatMessage]) -> LLMResult: ...


@dataclass(frozen=True, slots=True)
class RAGDocumentSummary:
    doc_id: str
    title: str
    source_path: str
    chunk_count: int
    char_count: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RAGChunkMatch:
    doc_id: str
    title: str
    source_path: str
    chunk_index: int
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RAGAnswer:
    question: str
    answer: str
    matches: list[RAGChunkMatch] = field(default_factory=list)
    used_llm: bool = False
    llm_model: str | None = None
    llm_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "used_llm": self.used_llm,
            "llm_model": self.llm_model,
            "llm_error": self.llm_error,
            "matches": [m.to_dict() for m in self.matches],
        }


def chunk_text(text: str, *, chunk_chars: int = 700, overlap_chars: int = 120) -> list[str]:
    """Split Thai/English text into overlapping chunks preserving paragraph/line boundaries."""
    cleaned = (text or "").replace("\r\n", "\n").replace("\x00", "").strip()
    if not cleaned:
        return []
    chunk_chars = max(chunk_chars, 100)
    overlap_chars = max(0, min(overlap_chars, chunk_chars // 2))

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_chars:
            current = f"{current}\n{para}".strip() if current else para
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap_chars:].strip() if overlap_chars else ""
            current = f"{tail}\n{para}".strip() if tail else para
        else:
            current = para
        while len(current) > chunk_chars:
            piece = current[:chunk_chars].strip()
            if piece:
                chunks.append(piece)
            step = max(1, chunk_chars - overlap_chars)
            current = current[step:].strip()
    if current:
        chunks.append(current)
    return chunks


def _extract_features(text: str) -> Counter[str]:
    """Build hybrid word-token + character 3-gram features for Thai + English retrieval."""
    norm = normalize_text(text).lower()
    counts: Counter[str] = Counter()
    for tok in thai_tokens(norm):
        clean_tok = tok.strip()
        if len(clean_tok) >= 2:
            counts[f"w:{clean_tok}"] += 2
    compact = norm.replace(" ", "")
    for i in range(max(0, len(compact) - 2)):
        counts[f"c:{compact[i : i + 3]}"] += 1
    return counts


def _read_local_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise WorkbenchError(
            f"ไม่รองรับไฟล์นามสกุล '{ext}' (รองรับเฉพาะ {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise WorkbenchError(f"ไม่สามารถอ่านไฟล์ UTF-8: {path.name}") from exc
    if ext in {".html", ".htm"}:
        raw = _HTML_TAG_RE.sub(" ", raw)
    elif ext == ".json":
        try:
            parsed = json.loads(raw)
            raw = json.dumps(parsed, ensure_ascii=False, indent=2)
        except ValueError:
            pass
    return raw.strip()


class LocalRAGStore:
    """Thread-safe local SQLite knowledge base with hybrid BM25 + TF-IDF retrieval."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        settings: Settings | None = None,
        llm_client: RAGLLMClient | None = None,
        chunk_chars: int = 700,
        overlap_chars: int = 120,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self._llm_client = llm_client
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
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
                CREATE TABLE IF NOT EXISTS rag_documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES rag_documents(doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(doc_id, chunk_index);
                """
            )

    def ingest_text(
        self,
        title: str,
        content: str,
        *,
        source_path: str = "inline",
        doc_id: str | None = None,
    ) -> RAGDocumentSummary:
        clean_title = (title or "").strip() or "Untitled"
        clean_content = (content or "").strip()
        if not clean_content:
            raise WorkbenchError("เนื้อหาเอกสารว่างเปล่า ไม่สามารถนำเข้า Local RAG ได้")
        sha = hashlib.sha256(clean_content.encode()).hexdigest()
        resolved_id = doc_id or f"doc-{sha[:12]}"
        chunks = chunk_text(
            clean_content, chunk_chars=self.chunk_chars, overlap_chars=self.overlap_chars
        )
        if not chunks:
            raise WorkbenchError("ไม่สามารถแบ่งส่วนข้อความ (chunks) จากเอกสารได้")

        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (resolved_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO rag_documents (
                    doc_id, title, source_path, content_sha256, chunk_count, char_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    clean_title,
                    source_path,
                    sha,
                    len(chunks),
                    len(clean_content),
                    now,
                ),
            )
            for idx, chunk in enumerate(chunks):
                feats = dict(_extract_features(chunk))
                conn.execute(
                    """
                    INSERT INTO rag_chunks (doc_id, chunk_index, text, features_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (resolved_id, idx, chunk, json.dumps(feats, ensure_ascii=False)),
                )
        return RAGDocumentSummary(
            doc_id=resolved_id,
            title=clean_title,
            source_path=source_path,
            chunk_count=len(chunks),
            char_count=len(clean_content),
            created_at=now,
        )

    def ingest_files(self, paths: Sequence[str | Path]) -> list[RAGDocumentSummary]:
        if not paths:
            raise WorkbenchError("เลือกไฟล์เอกสารอย่างน้อย 1 ไฟล์เพื่อนำเข้า Local RAG")
        summaries: list[RAGDocumentSummary] = []
        for raw in paths:
            p = Path(raw).expanduser().resolve()
            if not p.is_file():
                raise WorkbenchError(f"ไม่พบไฟล์เอกสาร: {p}")
            text = _read_local_document(p)
            summaries.append(
                self.ingest_text(p.name, text, source_path=str(p))
            )
        return summaries

    def list_documents(self) -> list[RAGDocumentSummary]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM rag_documents ORDER BY created_at DESC"
            ).fetchall()
        return [
            RAGDocumentSummary(
                doc_id=str(r["doc_id"]),
                title=str(r["title"]),
                source_path=str(r["source_path"]),
                chunk_count=int(r["chunk_count"]),
                char_count=int(r["char_count"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        target = (doc_id or "").strip()
        if not target:
            return False
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (target,))
            cur = conn.execute("DELETE FROM rag_documents WHERE doc_id = ?", (target,))
            return cur.rowcount > 0

    def clear_all(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks")
            conn.execute("DELETE FROM rag_documents")

    def search(self, query: str, *, top_k: int = 4) -> list[RAGChunkMatch]:
        clean_q = (query or "").strip()
        if not clean_q:
            raise WorkbenchError("กรุณาระบุคำค้นหาสำหรับ Local RAG")
        q_feats = _extract_features(clean_q)
        if not q_feats:
            return []

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.doc_id, d.title, d.source_path, c.chunk_index, c.text, c.features_json
                FROM rag_chunks c
                JOIN rag_documents d ON d.doc_id = c.doc_id
                """
            ).fetchall()

        if not rows:
            return []

        n_docs = len(rows)
        parsed_chunks: list[tuple[sqlite3.Row, dict[str, int]]] = []
        df: Counter[str] = Counter()
        for row in rows:
            feats: dict[str, int] = json.loads(row["features_json"])
            parsed_chunks.append((row, feats))
            for term in q_feats:
                if term in feats:
                    df[term] += 1

        # Compute IDF-weighted cosine similarity for each chunk
        idf = {
            term: math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5)) + 1.0
            for term in q_feats
        }
        q_weights = {term: count * idf[term] for term, count in q_feats.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

        scored: list[RAGChunkMatch] = []
        for row, feats in parsed_chunks:
            dot = 0.0
            for term, q_w in q_weights.items():
                tf = feats.get(term, 0)
                if tf:
                    dot += q_w * (1.0 + math.log(tf)) * idf[term]
            if dot <= 0.0:
                continue
            c_norm = math.sqrt(sum((1.0 + math.log(v)) ** 2 for v in feats.values() if v > 0)) or 1.0
            score = round(dot / (q_norm * c_norm), 4)
            scored.append(
                RAGChunkMatch(
                    doc_id=str(row["doc_id"]),
                    title=str(row["title"]),
                    source_path=str(row["source_path"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    score=score,
                )
            )

        scored.sort(key=lambda m: (-m.score, m.doc_id, m.chunk_index))
        return scored[: max(1, top_k)]

    def _get_llm(self) -> RAGLLMClient:
        if self._llm_client is not None:
            return self._llm_client
        if self.settings is None:
            raise WorkbenchError("Local LLM settings are not configured for LocalRAGStore.")
        from thai_offline_ai.adapters.llm import OllamaLocalLLM

        return OllamaLocalLLM.from_config(
            self.settings.llm, self.settings.app.offline_runtime
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int = 4,
        use_llm: bool = True,
    ) -> RAGAnswer:
        matches = self.search(question, top_k=top_k)
        if not matches:
            return RAGAnswer(
                question=question,
                answer="ไม่พบเอกสารใน Local RAG ที่ตรงกับคำถามนี้ กรุณานำเข้าเอกสารก่อนครับ",
                matches=[],
                used_llm=False,
            )

        context_blocks = [
            f"[Doc #{idx} · {m.title} (chunk {m.chunk_index}, score={m.score:.3f})]\n{m.text}"
            for idx, m in enumerate(matches, start=1)
        ]
        extractive_fallback = (
            "ผลลัพธ์จากเอกสารในเครื่อง (Extractive Local RAG):\n\n"
            + "\n\n".join(context_blocks)
        )
        if not use_llm:
            return RAGAnswer(
                question=question,
                answer=extractive_fallback,
                matches=matches,
                used_llm=False,
            )

        messages = [
            ChatMessage(
                "system",
                "You are a private offline Thai-English RAG assistant. Answer the user's question "
                "using ONLY the provided local document chunks. Cite sources as [Doc #1], [Doc #2], etc. "
                "If the chunks do not contain the answer, state that clearly.",
            ),
            ChatMessage(
                "user",
                "Local Document Chunks:\n"
                + "\n\n".join(context_blocks)
                + f"\n\nQuestion: {question.strip()}",
            ),
        ]
        try:
            llm_res = self._get_llm().chat(messages)
            return RAGAnswer(
                question=question,
                answer=llm_res.text.strip(),
                matches=matches,
                used_llm=True,
                llm_model=llm_res.model,
            )
        except WorkbenchError as exc:
            return RAGAnswer(
                question=question,
                answer=extractive_fallback,
                matches=matches,
                used_llm=False,
                llm_error=str(exc),
            )
