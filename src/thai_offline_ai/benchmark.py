"""Measured, local-only ASR performance benchmarks.

The benchmark deliberately times the public adapter boundary.  Its numbers therefore
include decoding, local audio normalization and ASR inference exactly as an end user
experiences them.  It does not claim browser capture or end-to-end microphone latency.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_output_dir


@dataclass(frozen=True, slots=True)
class ASRBenchmarkItem:
    """One real adapter invocation and its measured timing."""

    file_name: str
    status: str
    audio_seconds: float | None = None
    elapsed_seconds: float | None = None
    rtf: float | None = None
    backend: str | None = None
    model: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ASRBenchmarkResult:
    """A reproducible local benchmark report, without retaining transcripts."""

    benchmark_id: str
    created_at_epoch: float
    selected_model: str | None
    prompt_mode: str | None
    total_files: int
    succeeded: int
    failed: int
    total_audio_seconds: float
    total_elapsed_seconds: float
    aggregate_rtf: float | None
    items: list[ASRBenchmarkItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ASRBenchmark:
    """Sequential benchmark runner sharing one lazy ASR adapter cache."""

    def __init__(self, settings: Settings, adapter: ASRAdapter | None = None) -> None:
        self.settings = settings
        self.adapter = adapter or ASRAdapter(settings)

    def run(
        self,
        paths: list[str | Path],
        *,
        model: str | None = None,
        prompt_mode: str | None = None,
    ) -> ASRBenchmarkResult:
        if not paths:
            raise WorkbenchError("Choose at least one local audio file to benchmark.")
        items: list[ASRBenchmarkItem] = []
        total_elapsed = 0.0
        total_audio = 0.0
        for raw_path in paths:
            path = Path(raw_path)
            started = time.perf_counter()
            try:
                result = self.adapter.transcribe_file(path, model=model, prompt_mode=prompt_mode)
                elapsed = time.perf_counter() - started
                duration = result.duration
                total_elapsed += elapsed
                total_audio += duration or 0.0
                items.append(
                    ASRBenchmarkItem(
                        file_name=path.name,
                        status="SUCCESS",
                        audio_seconds=duration,
                        elapsed_seconds=elapsed,
                        rtf=(elapsed / duration) if duration and duration > 0 else None,
                        backend=result.backend,
                        model=result.model,
                    )
                )
            except WorkbenchError as exc:
                elapsed = time.perf_counter() - started
                total_elapsed += elapsed
                items.append(
                    ASRBenchmarkItem(file_name=path.name, status="FAILED", elapsed_seconds=elapsed, error=str(exc))
                )
        successful = sum(item.status == "SUCCESS" for item in items)
        return ASRBenchmarkResult(
            benchmark_id=uuid4().hex,
            created_at_epoch=time.time(),
            selected_model=model,
            prompt_mode=prompt_mode,
            total_files=len(items),
            succeeded=successful,
            failed=len(items) - successful,
            total_audio_seconds=total_audio,
            total_elapsed_seconds=total_elapsed,
            aggregate_rtf=(total_elapsed / total_audio) if total_audio > 0 else None,
            items=items,
        )


def export_benchmark_result(result: ASRBenchmarkResult, settings: Settings) -> tuple[Path, Path]:
    """Write compact JSON/CSV reports below the configured local output root."""
    directory = prepare_output_dir(settings) / "benchmarks"
    directory.mkdir(exist_ok=True)
    json_path = directory / f"asr_benchmark_{result.benchmark_id}.json"
    csv_path = directory / f"asr_benchmark_{result.benchmark_id}.csv"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("benchmark_id", "file_name", "status", "audio_seconds", "elapsed_seconds", "rtf", "backend", "model", "error"),
        )
        writer.writeheader()
        for item in result.items:
            writer.writerow({"benchmark_id": result.benchmark_id, **item.to_dict()})
    return json_path, csv_path
