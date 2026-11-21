"""Dataset split separation, provenance manifests, and leakage/contamination guards."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thai_offline_ai.errors import WorkbenchError

VALID_SPLITS = (
    "train_real",
    "train_synthetic",
    "unlabeled_real",
    "validation_real",
    "test_real",
)
HOLDOUT_SPLITS = frozenset({"validation_real", "test_real"})
TRAIN_SPLITS = frozenset({"train_real", "train_synthetic", "unlabeled_real"})
VALID_TRANSCRIPT_STATES = frozenset(
    {"verified", "synthetic_source", "pseudo_accepted", "pseudo_rejected", "unlabeled"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_file_sha256(path: str | Path) -> str:
    """Compute deterministic SHA-256 hex digest of an audio file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise WorkbenchError(f"Audio file not found for checksum: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One immutable dataset record carrying full provenance and consent metadata."""

    sample_id: str
    split: str
    audio_path: str
    source_text: str
    synthetic: bool
    provenance: str
    consent_status: str
    transcript_state: str
    audio_sha256: str
    dataset_version: str
    domain_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise WorkbenchError(
                f"Invalid dataset split '{self.split}'. Must be one of: {', '.join(VALID_SPLITS)}."
            )
        if self.transcript_state not in VALID_TRANSCRIPT_STATES:
            raise WorkbenchError(
                f"Invalid transcript_state '{self.transcript_state}'."
            )
        # Hard anti-contamination guard: synthetic audio must NEVER enter real holdouts.
        if self.synthetic and self.split in HOLDOUT_SPLITS:
            raise WorkbenchError(
                f"Governance violation: synthetic sample '{self.sample_id}' cannot be placed in "
                f"real holdout split '{self.split}'."
            )
        if not self.synthetic and self.split == "train_synthetic":
            raise WorkbenchError(
                f"Sample '{self.sample_id}' in train_synthetic must have synthetic=True."
            )
        if not self.sample_id.strip() or not self.audio_path.strip():
            raise WorkbenchError("DatasetSample requires non-empty sample_id and audio_path.")
        if not self.audio_sha256.strip():
            raise WorkbenchError("DatasetSample requires a computed audio_sha256 checksum.")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["domain_tags"] = list(self.domain_tags)
        return raw

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSample:
        return cls(
            sample_id=str(data["sample_id"]),
            split=str(data["split"]),
            audio_path=str(data["audio_path"]),
            source_text=str(data.get("source_text", "")),
            synthetic=bool(data.get("synthetic", False)),
            provenance=str(data.get("provenance", "unknown")),
            consent_status=str(data.get("consent_status", "unspecified")),
            transcript_state=str(data.get("transcript_state", "unlabeled")),
            audio_sha256=str(data["audio_sha256"]),
            dataset_version=str(data.get("dataset_version", "v1")),
            domain_tags=tuple(str(t) for t in data.get("domain_tags", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class DatasetManifest:
    """Versioned, checksum-fingerprinted dataset manifest for one split."""

    dataset_id: str
    split: str
    version: str
    samples: list[DatasetSample] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise WorkbenchError(f"Invalid manifest split '{self.split}'.")
        seen_ids: set[str] = set()
        for sample in self.samples:
            if sample.split != self.split:
                raise WorkbenchError(
                    f"Sample '{sample.sample_id}' has split '{sample.split}', expected '{self.split}'."
                )
            if sample.sample_id in seen_ids:
                raise WorkbenchError(f"Duplicate sample_id '{sample.sample_id}' in manifest.")
            seen_ids.add(sample.sample_id)

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint over sorted (sample_id, audio_sha256, source_text)."""
        digest = hashlib.sha256()
        digest.update(f"{self.dataset_id}|{self.split}|{self.version}\n".encode())
        for item in sorted(self.samples, key=lambda s: s.sample_id):
            line = f"{item.sample_id}|{item.audio_sha256}|{item.source_text}|{int(item.synthetic)}\n"
            digest.update(line.encode())
        return digest.hexdigest()[:16]

    def save_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "_manifest": {
                "dataset_id": self.dataset_id,
                "split": self.split,
                "version": self.version,
                "fingerprint": self.fingerprint,
                "sample_count": len(self.samples),
                "created_at": self.created_at,
            }
        }
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for sample in self.samples:
                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        return target

    def export_evaluation_csv(self, path: str | Path) -> Path:
        """Export a real holdout manifest as Phase 3.7 ASREvaluator CSV."""
        if self.split not in HOLDOUT_SPLITS:
            raise WorkbenchError(
                f"Only real holdout splits ({', '.join(sorted(HOLDOUT_SPLITS))}) may be exported for evaluation."
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "audio_path",
                    "reference_text",
                    "split",
                    "audio_sha256",
                    "dataset_version",
                    "domain_tags",
                ],
            )
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "audio_path": sample.audio_path,
                        "reference_text": sample.source_text,
                        "split": sample.split,
                        "audio_sha256": sample.audio_sha256,
                        "dataset_version": sample.dataset_version,
                        "domain_tags": "|".join(sample.domain_tags),
                    }
                )
        return target

    @classmethod
    def load_jsonl(cls, path: str | Path) -> DatasetManifest:
        source = Path(path)
        if not source.is_file():
            raise WorkbenchError(f"Dataset manifest not found: {source}")
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise WorkbenchError(f"Dataset manifest is empty: {source}")
        first = json.loads(lines[0])
        if "_manifest" not in first:
            raise WorkbenchError("Dataset JSONL manifest is missing the _manifest header record.")
        meta = first["_manifest"]
        samples = [DatasetSample.from_dict(json.loads(line)) for line in lines[1:]]
        return cls(
            dataset_id=str(meta["dataset_id"]),
            split=str(meta["split"]),
            version=str(meta["version"]),
            samples=samples,
            created_at=str(meta.get("created_at", _utc_now())),
        )


def validate_no_leakage(
    train_manifests: Sequence[DatasetManifest],
    holdout_manifests: Sequence[DatasetManifest],
) -> None:
    """Verify zero audio-checksum overlap between any training split and real holdout splits."""
    holdout_hashes: dict[str, tuple[str, str]] = {}
    for manifest in holdout_manifests:
        if manifest.split not in HOLDOUT_SPLITS:
            raise WorkbenchError(f"Expected holdout split, got '{manifest.split}'.")
        for sample in manifest.samples:
            if sample.synthetic:
                raise WorkbenchError(
                    f"Synthetic sample '{sample.sample_id}' detected in holdout '{manifest.split}'."
                )
            holdout_hashes[sample.audio_sha256] = (manifest.split, sample.sample_id)

    for manifest in train_manifests:
        for sample in manifest.samples:
            if sample.audio_sha256 in holdout_hashes:
                holdout_split, holdout_id = holdout_hashes[sample.audio_sha256]
                raise WorkbenchError(
                    f"Data leakage detected: training sample '{sample.sample_id}' ({manifest.split}) "
                    f"matches audio checksum of holdout sample '{holdout_id}' ({holdout_split})."
                )
