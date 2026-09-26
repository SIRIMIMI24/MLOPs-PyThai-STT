"""Sequential, failure-isolated local batch ASR and safe exports."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.asr_types import ASRResult
from thai_offline_ai.audio import is_supported_audio
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_output_dir


@dataclass(frozen=True, slots=True)
class BatchASRItemResult:
    file_name: str
    source_path: str
    status: str
    asr_result: ASRResult | None = None
    duration: float | None = None
    processing_time: float | None = None
    rtf: float | None = None
    original_audio_metadata: dict[str, object] = field(default_factory=dict)
    normalized_audio_metadata: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchASRResult:
    batch_id: str
    total_files: int
    succeeded: int
    failed: int
    total_audio_duration: float
    total_processing_time: float
    items: list[BatchASRItemResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BatchASRProcessor:
    """Sequential processor: one cached adapter, isolated failures per input file."""

    def __init__(self, settings: Settings, adapter: ASRAdapter | None = None) -> None:
        self.settings = settings
        self.adapter = adapter or ASRAdapter(settings)

    def paths_in_directory(self, directory: str | Path, *, recursive: bool = False) -> list[Path]:
        root = Path(directory)
        if not root.is_dir():
            raise WorkbenchError("ASR batch input must be an existing local folder.")
        candidates = root.rglob("*") if recursive else root.iterdir()
        return sorted((item for item in candidates if item.is_file() and is_supported_audio(item)), key=lambda item: item.name.lower())

    def transcribe_paths(self, paths: list[str | Path]) -> BatchASRResult:
        batch_id = uuid4().hex
        started = time.perf_counter()
        items: list[BatchASRItemResult] = []
        for raw_path in paths:
            path = Path(raw_path)
            item_started = time.perf_counter()
            try:
                result = self.adapter.transcribe_file(path)
                processing_time = time.perf_counter() - item_started
                metadata = result.metadata
                duration = result.duration
                items.append(
                    BatchASRItemResult(
                        file_name=path.name,
                        source_path=str(path),
                        status="SUCCESS",
                        asr_result=result,
                        duration=duration,
                        processing_time=processing_time,
                        rtf=(processing_time / duration) if duration else None,
                        original_audio_metadata=dict(metadata.get("original_audio", {})),
                        normalized_audio_metadata=dict(metadata.get("normalized_audio", {})),
                    )
                )
            except WorkbenchError as exc:
                items.append(
                    BatchASRItemResult(
                        file_name=path.name,
                        source_path=str(path),
                        status="FAILED",
                        processing_time=time.perf_counter() - item_started,
                        error=str(exc),
                    )
                )
        succeeded = sum(item.status == "SUCCESS" for item in items)
        total_duration = sum(item.duration or 0.0 for item in items)
        return BatchASRResult(
            batch_id=batch_id,
            total_files=len(items),
            succeeded=succeeded,
            failed=len(items) - succeeded,
            total_audio_duration=total_duration,
            total_processing_time=time.perf_counter() - started,
            items=items,
        )


def export_batch_result(result: BatchASRResult, settings: Settings) -> tuple[Path, Path]:
    """Create non-colliding structured JSON and tabular CSV below ``outputs/batch``."""
    directory = prepare_output_dir(settings) / "batch"
    directory.mkdir(exist_ok=True)
    json_path = directory / f"asr_batch_{result.batch_id}.json"
    csv_path = directory / f"asr_batch_{result.batch_id}.csv"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "batch_id", "file_name", "status", "duration", "processing_time", "rtf", "transcript",
        "backend", "model", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result.items:
            asr = item.asr_result
            writer.writerow(
                {
                    "batch_id": result.batch_id,
                    "file_name": item.file_name,
                    "status": item.status,
                    "duration": item.duration,
                    "processing_time": item.processing_time,
                    "rtf": item.rtf,
                    "transcript": asr.text if asr else "",
                    "backend": asr.backend if asr else "",
                    "model": asr.model if asr else "",
                    "error": item.error or "",
                }
            )
    return json_path, csv_path
