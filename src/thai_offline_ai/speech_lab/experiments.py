"""Local SQLite experiment tracking, training bundle export, and anti-degradation promotion gate."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.evaluation import EvaluationReport
from thai_offline_ai.speech_lab.datasets import (
    HOLDOUT_SPLITS,
    DatasetManifest,
    DatasetSample,
    validate_no_leakage,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Anti-degradation thresholds required before a candidate replaces the baseline."""

    min_cer_improvement: float = 0.005
    max_wer_regression: float = 0.005
    max_rtf_ratio: float = 1.15
    max_domain_recall_regression: float = 0.01


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    approved: bool
    status: str
    baseline_id: str
    candidate_id: str
    cer_delta: float
    wer_delta: float | None
    rtf_ratio: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    created_at: str
    base_model: str
    candidate_model: str
    real_sample_count: int
    synthetic_sample_count: int
    pseudo_sample_count: int
    train_dataset_version: str
    pseudo_dataset_version: str
    eval_dataset_id: str
    eval_dataset_version: str
    eval_split: str
    preprocessing_config: dict[str, Any]
    augmentation_config: dict[str, Any]
    training_config: dict[str, Any]
    cer: float | None
    wer: float | None
    rtf: float | None
    domain_term_recall: float | None
    artifact_path: str
    promotion_status: str = "candidate"
    promotion_reason: str = ""

    @property
    def real_synthetic_ratio(self) -> str:
        return f"{self.real_sample_count}:{self.synthetic_sample_count}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "real_synthetic_ratio": self.real_synthetic_ratio}


def evaluate_promotion(
    baseline: ExperimentRecord,
    candidate: ExperimentRecord,
    criteria: PromotionCriteria | None = None,
) -> PromotionDecision:
    """Deterministic anti-degradation gate comparing candidate vs baseline on identical holdout."""
    rules = criteria or PromotionCriteria()
    reasons: list[str] = []

    if baseline.eval_split not in HOLDOUT_SPLITS or candidate.eval_split not in HOLDOUT_SPLITS:
        reasons.append("Evaluation split must be an immutable real holdout (validation_real or test_real).")
    if baseline.eval_dataset_id != candidate.eval_dataset_id:
        reasons.append(
            f"Eval dataset mismatch: baseline='{baseline.eval_dataset_id}' vs candidate='{candidate.eval_dataset_id}'."
        )
    if baseline.eval_dataset_version != candidate.eval_dataset_version:
        reasons.append(
            f"Eval dataset version/fingerprint mismatch: baseline='{baseline.eval_dataset_version}' "
            f"vs candidate='{candidate.eval_dataset_version}'."
        )
    if baseline.cer is None or candidate.cer is None:
        reasons.append("Both baseline and candidate must have measured aggregate CER.")
        return PromotionDecision(
            approved=False,
            status="rejected",
            baseline_id=baseline.experiment_id,
            candidate_id=candidate.experiment_id,
            cer_delta=0.0,
            wer_delta=None,
            rtf_ratio=None,
            reasons=tuple(reasons),
        )

    cer_delta = candidate.cer - baseline.cer  # negative is better
    if -cer_delta < rules.min_cer_improvement:
        reasons.append(
            f"CER improvement {-cer_delta:.4f} is below required minimum {rules.min_cer_improvement:.4f} "
            f"(baseline={baseline.cer:.4f}, candidate={candidate.cer:.4f})."
        )

    wer_delta: float | None = None
    if baseline.wer is not None and candidate.wer is not None:
        wer_delta = candidate.wer - baseline.wer
        if wer_delta > rules.max_wer_regression:
            reasons.append(
                f"WER regressed by +{wer_delta:.4f} (max allowed +{rules.max_wer_regression:.4f})."
            )

    rtf_ratio: float | None = None
    if baseline.rtf and baseline.rtf > 0 and candidate.rtf is not None:
        rtf_ratio = candidate.rtf / baseline.rtf
        if rtf_ratio > rules.max_rtf_ratio:
            reasons.append(
                f"RTF slowdown ratio {rtf_ratio:.2f}x exceeds limit {rules.max_rtf_ratio:.2f}x."
            )

    if (
        baseline.domain_term_recall is not None
        and candidate.domain_term_recall is not None
        and (baseline.domain_term_recall - candidate.domain_term_recall)
        > rules.max_domain_recall_regression
    ):
        reasons.append(
            f"Domain term recall regressed from {baseline.domain_term_recall:.3f} "
            f"to {candidate.domain_term_recall:.3f}."
        )

    approved = len(reasons) == 0
    if approved:
        reasons.append(
            f"Passed anti-degradation gate on unchanged {candidate.eval_split} "
            f"({candidate.eval_dataset_id}@{candidate.eval_dataset_version}): "
            f"CER {baseline.cer:.4f} -> {candidate.cer:.4f}."
        )
    return PromotionDecision(
        approved=approved,
        status="promoted" if approved else "rejected",
        baseline_id=baseline.experiment_id,
        candidate_id=candidate.experiment_id,
        cer_delta=round(cer_delta, 6),
        wer_delta=round(wer_delta, 6) if wer_delta is not None else None,
        rtf_ratio=round(rtf_ratio, 4) if rtf_ratio is not None else None,
        reasons=tuple(reasons),
    )


class ExperimentRegistry:
    """Thread-safe local SQLite store tracking ASR experiments and model promotion decisions."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
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
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    candidate_model TEXT NOT NULL,
                    real_sample_count INTEGER NOT NULL,
                    synthetic_sample_count INTEGER NOT NULL,
                    pseudo_sample_count INTEGER NOT NULL,
                    train_dataset_version TEXT NOT NULL,
                    pseudo_dataset_version TEXT NOT NULL,
                    eval_dataset_id TEXT NOT NULL,
                    eval_dataset_version TEXT NOT NULL,
                    eval_split TEXT NOT NULL,
                    preprocessing_json TEXT NOT NULL,
                    augmentation_json TEXT NOT NULL,
                    training_json TEXT NOT NULL,
                    cer REAL,
                    wer REAL,
                    rtf REAL,
                    domain_term_recall REAL,
                    artifact_path TEXT NOT NULL,
                    promotion_status TEXT NOT NULL,
                    promotion_reason TEXT NOT NULL
                );
                """
            )

    def record_from_evaluation(
        self,
        report: EvaluationReport,
        *,
        base_model: str,
        candidate_model: str,
        eval_dataset_version: str,
        eval_split: str = "test_real",
        real_sample_count: int = 0,
        synthetic_sample_count: int = 0,
        pseudo_sample_count: int = 0,
        train_dataset_version: str = "none",
        pseudo_dataset_version: str = "none",
        preprocessing_config: dict[str, Any] | None = None,
        augmentation_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        artifact_path: str = "",
        is_baseline: bool = False,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        if eval_split not in HOLDOUT_SPLITS:
            raise WorkbenchError(
                f"Experiments must be evaluated on real holdouts ({', '.join(sorted(HOLDOUT_SPLITS))}), "
                f"never '{eval_split}'."
            )
        domain_recall: float | None = None
        if report.domain_term_metrics and report.domain_term_metrics.get("reference_occurrences", 0):
            domain_recall = float(report.domain_term_metrics["recall"])
        record = ExperimentRecord(
            experiment_id=experiment_id or f"exp-{uuid4().hex[:10]}",
            created_at=_utc_now(),
            base_model=base_model,
            candidate_model=candidate_model,
            real_sample_count=real_sample_count,
            synthetic_sample_count=synthetic_sample_count,
            pseudo_sample_count=pseudo_sample_count,
            train_dataset_version=train_dataset_version,
            pseudo_dataset_version=pseudo_dataset_version,
            eval_dataset_id=report.dataset_id,
            eval_dataset_version=eval_dataset_version,
            eval_split=eval_split,
            preprocessing_config=dict(preprocessing_config or {}),
            augmentation_config=dict(augmentation_config or {}),
            training_config=dict(training_config or {}),
            cer=report.aggregate_cer,
            wer=report.aggregate_wer,
            rtf=report.aggregate_rtf,
            domain_term_recall=domain_recall,
            artifact_path=artifact_path,
            promotion_status="baseline" if is_baseline else "candidate",
            promotion_reason="Registered as active baseline." if is_baseline else "",
        )
        self.save_record(record)
        return record

    def save_record(self, rec: ExperimentRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, created_at, base_model, candidate_model,
                    real_sample_count, synthetic_sample_count, pseudo_sample_count,
                    train_dataset_version, pseudo_dataset_version,
                    eval_dataset_id, eval_dataset_version, eval_split,
                    preprocessing_json, augmentation_json, training_json,
                    cer, wer, rtf, domain_term_recall, artifact_path,
                    promotion_status, promotion_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.experiment_id,
                    rec.created_at,
                    rec.base_model,
                    rec.candidate_model,
                    rec.real_sample_count,
                    rec.synthetic_sample_count,
                    rec.pseudo_sample_count,
                    rec.train_dataset_version,
                    rec.pseudo_dataset_version,
                    rec.eval_dataset_id,
                    rec.eval_dataset_version,
                    rec.eval_split,
                    json.dumps(rec.preprocessing_config, ensure_ascii=False),
                    json.dumps(rec.augmentation_config, ensure_ascii=False),
                    json.dumps(rec.training_config, ensure_ascii=False),
                    rec.cer,
                    rec.wer,
                    rec.rtf,
                    rec.domain_term_recall,
                    rec.artifact_path,
                    rec.promotion_status,
                    rec.promotion_reason,
                ),
            )

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_experiments(self, limit: int = 30) -> list[ExperimentRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def promote_or_reject(
        self,
        baseline_id: str,
        candidate_id: str,
        criteria: PromotionCriteria | None = None,
    ) -> PromotionDecision:
        baseline = self.get(baseline_id)
        candidate = self.get(candidate_id)
        if baseline is None or candidate is None:
            raise WorkbenchError("Both baseline_id and candidate_id must exist in ExperimentRegistry.")
        decision = evaluate_promotion(baseline, candidate, criteria=criteria)
        reason_text = "; ".join(decision.reasons)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET promotion_status = ?, promotion_reason = ? WHERE experiment_id = ?",
                (decision.status, reason_text, candidate.experiment_id),
            )
        return decision

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=str(row["experiment_id"]),
            created_at=str(row["created_at"]),
            base_model=str(row["base_model"]),
            candidate_model=str(row["candidate_model"]),
            real_sample_count=int(row["real_sample_count"]),
            synthetic_sample_count=int(row["synthetic_sample_count"]),
            pseudo_sample_count=int(row["pseudo_sample_count"]),
            train_dataset_version=str(row["train_dataset_version"]),
            pseudo_dataset_version=str(row["pseudo_dataset_version"]),
            eval_dataset_id=str(row["eval_dataset_id"]),
            eval_dataset_version=str(row["eval_dataset_version"]),
            eval_split=str(row["eval_split"]),
            preprocessing_config=json.loads(row["preprocessing_json"] or "{}"),
            augmentation_config=json.loads(row["augmentation_json"] or "{}"),
            training_config=json.loads(row["training_json"] or "{}"),
            cer=float(row["cer"]) if row["cer"] is not None else None,
            wer=float(row["wer"]) if row["wer"] is not None else None,
            rtf=float(row["rtf"]) if row["rtf"] is not None else None,
            domain_term_recall=(
                float(row["domain_term_recall"])
                if row["domain_term_recall"] is not None
                else None
            ),
            artifact_path=str(row["artifact_path"]),
            promotion_status=str(row["promotion_status"]),
            promotion_reason=str(row["promotion_reason"]),
        )


@dataclass(frozen=True, slots=True)
class TrainingBundleExport:
    manifest_path: Path
    recipe_path: Path
    real_count: int
    synthetic_count: int
    pseudo_count: int
    total_count: int
    capped_synthetic: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


def export_training_bundle(
    *,
    output_dir: str | Path,
    bundle_name: str,
    base_model: str,
    train_real: DatasetManifest | None = None,
    train_synthetic: DatasetManifest | None = None,
    pseudo_accepted: DatasetManifest | None = None,
    holdout_manifests: Sequence[DatasetManifest] = (),
    max_synthetic_to_real_ratio: float = 2.0,
) -> TrainingBundleExport:
    """Export leakage-checked NeMo/HF-compatible JSONL training manifest & fine-tuning recipe."""
    train_sources = [m for m in (train_real, train_synthetic, pseudo_accepted) if m is not None]
    if not train_sources:
        raise WorkbenchError("Provide at least one training manifest to build a training bundle.")
    if holdout_manifests:
        validate_no_leakage(train_sources, holdout_manifests)

    real_samples: list[DatasetSample] = list(train_real.samples) if train_real else []
    pseudo_samples: list[DatasetSample] = (
        [s for s in pseudo_accepted.samples if s.transcript_state == "pseudo_accepted"]
        if pseudo_accepted
        else []
    )
    syn_samples: list[DatasetSample] = list(train_synthetic.samples) if train_synthetic else []

    warnings: list[str] = []
    capped = False
    if real_samples and syn_samples:
        max_syn = max(1, round(len(real_samples) * max_synthetic_to_real_ratio))
        if len(syn_samples) > max_syn:
            syn_samples = syn_samples[:max_syn]
            capped = True
            warnings.append(
                f"Capped synthetic samples to {max_syn} (max ratio {max_synthetic_to_real_ratio}:1 vs real)."
            )
    elif not real_samples and syn_samples:
        warnings.append(
            "No train_real samples included; synthetic-only training should be used for smoke experiments only."
        )

    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / f"{bundle_name}_train.jsonl"
    combined = [*real_samples, *pseudo_samples, *syn_samples]
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in combined:
            entry = {
                "sample_id": item.sample_id,
                "audio_filepath": item.audio_path,
                "text": item.source_text,
                "split": item.split,
                "synthetic": item.synthetic,
                "provenance": item.provenance,
                "audio_sha256": item.audio_sha256,
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    recipe_path = out_root / f"{bundle_name}_recipe.json"
    recipe = {
        "bundle_name": bundle_name,
        "created_at": _utc_now(),
        "base_model": base_model,
        "train_manifest": str(manifest_path),
        "counts": {
            "real": len(real_samples),
            "pseudo_accepted": len(pseudo_samples),
            "synthetic": len(syn_samples),
            "total": len(combined),
        },
        "dataset_versions": {
            "train_real": train_real.fingerprint if train_real else None,
            "train_synthetic": train_synthetic.fingerprint if train_synthetic else None,
            "pseudo_accepted": pseudo_accepted.fingerprint if pseudo_accepted else None,
        },
        "hardware_guidance": {
            "target_gpu": "RTX 4050 6 GB VRAM / CPU fallback",
            "recommended_strategy": "LoRA / adapter fine-tuning with fp16/bf16 and gradient accumulation",
            "promotion_rule": "Re-evaluate on unchanged test_real via ASREvaluator and pass evaluate_promotion()",
        },
        "warnings": warnings,
    }
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainingBundleExport(
        manifest_path=manifest_path,
        recipe_path=recipe_path,
        real_count=len(real_samples),
        synthetic_count=len(syn_samples),
        pseudo_count=len(pseudo_samples),
        total_count=len(combined),
        capped_synthetic=capped,
        warnings=tuple(warnings),
    )
