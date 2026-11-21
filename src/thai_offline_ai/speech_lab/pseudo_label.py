"""Controlled pseudo-labeling pipeline with deterministic quality filters and provenance."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.evaluation import NORMALIZATION_VERSION, normalize_text
from thai_offline_ai.speech_lab.datasets import (
    DatasetManifest,
    DatasetSample,
    compute_file_sha256,
)

# Detect degenerate ASR hallucination loops (e.g. same 2-8 char chunk repeated 5+ times in a row).
_REPETITION_LOOP_RE = re.compile(r"(.{2,10}?)\1{4,}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class PseudoLabelFilterConfig:
    """Transparent deterministic acceptance thresholds for unlabeled real audio."""

    min_chars: int = 4
    max_chars: int = 800
    min_duration_seconds: float = 0.4
    max_duration_seconds: float = 45.0
    min_chars_per_second: float = 1.5
    max_chars_per_second: float = 32.0
    min_confidence: float | None = None
    reject_repetition_loops: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PseudoLabelRecord:
    sample_id: str
    audio_path: str
    audio_sha256: str
    teacher_backend: str
    teacher_model: str
    raw_transcript: str
    normalized_transcript: str
    normalization_version: str
    duration_seconds: float
    confidence: float | None
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rejection_reasons"] = list(self.rejection_reasons)
        return data


@dataclass(slots=True)
class PseudoLabelReport:
    dataset_id: str
    dataset_version: str
    teacher_model: str
    filter_config: PseudoLabelFilterConfig
    records: list[PseudoLabelRecord]
    accepted_manifest: DatasetManifest
    report_path: Path | None = None
    accepted_manifest_path: Path | None = None

    @property
    def accepted_count(self) -> int:
        return sum(1 for r in self.records if r.accepted)

    @property
    def rejected_count(self) -> int:
        return sum(1 for r in self.records if not r.accepted)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "teacher_model": self.teacher_model,
            "total": len(self.records),
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "filter_config": self.filter_config.to_dict(),
            "report_path": str(self.report_path) if self.report_path else None,
            "accepted_manifest_path": (
                str(self.accepted_manifest_path) if self.accepted_manifest_path else None
            ),
        }


def evaluate_pseudo_label_quality(
    *,
    raw_transcript: str,
    normalized_transcript: str,
    duration_seconds: float,
    confidence: float | None,
    filters: PseudoLabelFilterConfig,
) -> tuple[bool, list[str]]:
    """Apply deterministic quality gates; never fabricates confidence when missing."""
    reasons: list[str] = []
    char_len = len(normalized_transcript.replace(" ", ""))
    if not raw_transcript.strip() or char_len < filters.min_chars:
        reasons.append(f"transcript_too_short(<{filters.min_chars} chars)")
    if char_len > filters.max_chars:
        reasons.append(f"transcript_too_long(>{filters.max_chars} chars)")
    if duration_seconds < filters.min_duration_seconds:
        reasons.append(f"audio_too_short(<{filters.min_duration_seconds:.2f}s)")
    if duration_seconds > filters.max_duration_seconds:
        reasons.append(f"audio_too_long(>{filters.max_duration_seconds:.2f}s)")
    if duration_seconds > 0 and char_len > 0:
        cps = char_len / duration_seconds
        if cps < filters.min_chars_per_second:
            reasons.append(f"speaking_rate_too_low({cps:.2f} chars/s)")
        if cps > filters.max_chars_per_second:
            reasons.append(f"speaking_rate_too_high({cps:.2f} chars/s)")
    if (
        filters.min_confidence is not None
        and confidence is not None
        and confidence < filters.min_confidence
    ):
        reasons.append(f"confidence_below_threshold({confidence:.3f}<{filters.min_confidence:.3f})")
    if filters.reject_repetition_loops and _REPETITION_LOOP_RE.search(normalized_transcript):
        reasons.append("repetition_loop_detected")
    return len(reasons) == 0, reasons


class PseudoLabelingPipeline:
    """Runs one fixed teacher ASR model on unlabeled_real audio with reviewable filtering."""

    def __init__(self, settings: Settings, teacher_asr: ASRAdapter | None = None) -> None:
        self.settings = settings
        self.teacher_asr = teacher_asr or ASRAdapter(settings)

    def run(
        self,
        audio_paths: Sequence[str | Path],
        *,
        dataset_id: str,
        dataset_version: str = "v1",
        output_dir: str | Path,
        filters: PseudoLabelFilterConfig | None = None,
        teacher_model: str | None = None,
        prompt_mode: str | None = None,
        consent_status: str = "consented_unlabeled_real",
    ) -> PseudoLabelReport:
        if not audio_paths:
            raise WorkbenchError("Provide at least one local unlabeled audio file for pseudo-labeling.")
        cfg = filters or PseudoLabelFilterConfig()
        out_root = Path(output_dir).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        records: list[PseudoLabelRecord] = []
        accepted_samples: list[DatasetSample] = []
        resolved_teacher_model = teacher_model or (self.settings.asr.model or "default")

        for idx, raw_path in enumerate(audio_paths, start=1):
            path = Path(raw_path).expanduser().resolve()
            sample_id = f"{dataset_id}_pl_{idx:04d}_{path.stem}"
            sha = compute_file_sha256(path)
            asr_res = self.teacher_asr.transcribe_file(
                path, model=teacher_model, prompt_mode=prompt_mode
            )
            resolved_teacher_model = asr_res.model
            raw_text = (asr_res.text or "").strip()
            norm_text = normalize_text(raw_text) if raw_text else ""

            # Preserve genuinely supplied confidence only; never fabricate one.
            raw_conf = asr_res.metadata.get("confidence")
            confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else None

            accepted, reasons = evaluate_pseudo_label_quality(
                raw_transcript=raw_text,
                normalized_transcript=norm_text,
                duration_seconds=float(asr_res.duration or 0.0),
                confidence=confidence,
                filters=cfg,
            )
            rec = PseudoLabelRecord(
                sample_id=sample_id,
                audio_path=str(path),
                audio_sha256=sha,
                teacher_backend=asr_res.backend,
                teacher_model=asr_res.model,
                raw_transcript=raw_text,
                normalized_transcript=norm_text,
                normalization_version=NORMALIZATION_VERSION,
                duration_seconds=round(float(asr_res.duration or 0.0), 3),
                confidence=confidence,
                accepted=accepted,
                rejection_reasons=tuple(reasons),
            )
            records.append(rec)
            if accepted:
                accepted_samples.append(
                    DatasetSample(
                        sample_id=sample_id,
                        split="unlabeled_real",
                        audio_path=str(path),
                        source_text=raw_text,
                        synthetic=False,
                        provenance=f"pseudo_label:{asr_res.backend}:{asr_res.model}",
                        consent_status=consent_status,
                        transcript_state="pseudo_accepted",
                        audio_sha256=sha,
                        dataset_version=dataset_version,
                        metadata={
                            "raw_transcript": raw_text,
                            "normalized_transcript": norm_text,
                            "normalization_version": NORMALIZATION_VERSION,
                            "confidence": confidence,
                            "duration_seconds": rec.duration_seconds,
                        },
                    )
                )

        accepted_manifest = DatasetManifest(
            dataset_id=f"{dataset_id}_pseudo_accepted",
            split="unlabeled_real",
            version=dataset_version,
            samples=accepted_samples,
        )
        manifest_path = accepted_manifest.save_jsonl(
            out_root / f"{dataset_id}_{dataset_version}_accepted.jsonl"
        )
        report_path = out_root / f"{dataset_id}_{dataset_version}_pseudo_report.json"
        report_payload = {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "teacher_model": resolved_teacher_model,
            "filter_config": cfg.to_dict(),
            "accepted": sum(1 for r in records if r.accepted),
            "rejected": sum(1 for r in records if not r.accepted),
            "records": [r.to_dict() for r in records],
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return PseudoLabelReport(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            teacher_model=resolved_teacher_model,
            filter_config=cfg,
            records=records,
            accepted_manifest=accepted_manifest,
            report_path=report_path,
            accepted_manifest_path=manifest_path,
        )
