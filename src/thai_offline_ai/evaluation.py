"""Deterministic local ASR evaluation, reporting, and error analysis."""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from thai_offline_ai.batch import BatchASRProcessor
from thai_offline_ai.config import Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_output_dir

NORMALIZATION_VERSION = "thai-asr-eval-v1"
TOKENIZER = "pythainlp:newmm"
_PUNCTUATION = re.compile(r"[,\.!?;:\(\)\[\]\{\}\"'“”‘’]")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class EvaluationManifestRecord:
    sample_id: str
    audio_path: Path
    reference_text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_units: int = 0

    @property
    def rate(self) -> float | None:
        if self.reference_units == 0:
            return 0.0 if self.insertions == 0 else None
        return (self.substitutions + self.deletions + self.insertions) / self.reference_units

    def to_dict(self) -> dict[str, int | float | None]:
        return {**asdict(self), "rate": self.rate}


@dataclass(frozen=True, slots=True)
class EvaluationSampleResult:
    sample_id: str
    status: str
    reference_raw: str
    prediction_raw: str
    reference_normalized: str
    prediction_normalized: str
    cer: float | None = None
    wer: float | None = None
    char_substitutions: int = 0
    char_deletions: int = 0
    char_insertions: int = 0
    char_reference_units: int = 0
    word_substitutions: int = 0
    word_deletions: int = 0
    word_insertions: int = 0
    word_reference_units: int = 0
    duration: float | None = None
    processing_time: float | None = None
    rtf: float | None = None
    domain_metrics: dict[str, int | float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_id: str
    normalization_version: str
    tokenizer: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    total_audio_duration: float
    total_processing_time: float
    aggregate_cer: float | None
    aggregate_wer: float | None
    aggregate_rtf: float | None
    char_counts: EditCounts
    word_counts: EditCounts
    domain_term_metrics: dict[str, int | float]
    breakdowns: dict[str, dict[str, dict[str, float | int | None]]]
    error_analysis: dict[str, list[dict[str, Any]]]
    samples: list[EvaluationSampleResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Versioned formatting normalization; it deliberately does not correct words."""
    from pythainlp.util import normalize as thai_normalize

    text = thai_normalize(unicodedata.normalize("NFKC", text))
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub("", text)).strip()


def thai_tokens(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return word_tokenize(normalize_text(text), engine="newmm", keep_whitespace=False)


def edit_counts(reference: Sequence[str], prediction: Sequence[str]) -> tuple[EditCounts, list[tuple[str, str, str]]]:
    """Return Levenshtein counts and deterministic aligned edit operations."""
    rows, columns = len(reference), len(prediction)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        table[row][0] = row
    for column in range(columns + 1):
        table[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            cost = int(reference[row - 1] != prediction[column - 1])
            table[row][column] = min(
                table[row - 1][column] + 1,
                table[row][column - 1] + 1,
                table[row - 1][column - 1] + cost,
            )
    row, column = rows, columns
    operations: list[tuple[str, str, str]] = []
    substitutions = deletions = insertions = 0
    while row or column:
        if row and column and reference[row - 1] == prediction[column - 1] and table[row][column] == table[row - 1][column - 1]:
            row, column = row - 1, column - 1
        elif row and column and table[row][column] == table[row - 1][column - 1] + 1:
            substitutions += 1
            operations.append(("substitution", reference[row - 1], prediction[column - 1]))
            row, column = row - 1, column - 1
        elif row and table[row][column] == table[row - 1][column] + 1:
            deletions += 1
            operations.append(("deletion", reference[row - 1], ""))
            row -= 1
        else:
            insertions += 1
            operations.append(("insertion", "", prediction[column - 1]))
            column -= 1
    operations.reverse()
    return EditCounts(substitutions, deletions, insertions, rows), operations


def character_counts(reference: str, prediction: str) -> tuple[EditCounts, list[tuple[str, str, str]]]:
    return edit_counts(list(normalize_text(reference).replace(" ", "")), list(normalize_text(prediction).replace(" ", "")))


def word_counts(reference: str, prediction: str) -> tuple[EditCounts, list[tuple[str, str, str]]]:
    return edit_counts(thai_tokens(reference), thai_tokens(prediction))


def load_manifest(path: str | Path, *, limit: int | None = None) -> list[EvaluationManifestRecord]:
    manifest = Path(path)
    if manifest.suffix.lower() != ".csv" or not manifest.is_file():
        raise WorkbenchError("Evaluation manifest must be an existing UTF-8 CSV file.")
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "audio_path", "reference_text"}
            if not reader.fieldnames or not required <= set(reader.fieldnames):
                raise WorkbenchError("Manifest requires sample_id, audio_path, and reference_text columns.")
            records: list[EvaluationManifestRecord] = []
            seen: set[str] = set()
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                reference = row.get("reference_text") or ""
                raw_audio = (row.get("audio_path") or "").strip()
                if not sample_id or not raw_audio:
                    raise WorkbenchError("Each manifest row needs non-empty sample_id and audio_path values.")
                if sample_id in seen:
                    raise WorkbenchError(f"Duplicate manifest sample_id: {sample_id}.")
                seen.add(sample_id)
                audio_path = Path(raw_audio)
                if not audio_path.is_absolute():
                    audio_path = (manifest.parent / audio_path).resolve()
                metadata = {key: value for key, value in row.items() if key not in required and value}
                records.append(EvaluationManifestRecord(sample_id, audio_path, reference, metadata))
                if limit is not None and len(records) >= limit:
                    break
            return records
    except WorkbenchError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise WorkbenchError("Cannot read the evaluation manifest as UTF-8 CSV.") from exc


def load_domain_terms(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        terms = data.get("terms", []) if isinstance(data, dict) else data
        if not isinstance(terms, list) or not all(isinstance(term, str) and term.strip() for term in terms):
            raise ValueError("terms")
        return [normalize_text(term) for term in terms]
    except Exception as exc:
        raise WorkbenchError("Domain lexicon must be YAML with a non-empty string terms list.") from exc


def domain_term_metrics(reference: str, prediction: str, terms: Sequence[str]) -> dict[str, int | float]:
    reference, prediction = normalize_text(reference), normalize_text(prediction)
    occurrences = correct = 0
    for term in terms:
        count = reference.count(term)
        occurrences += count
        correct += min(count, prediction.count(term))
    missed = occurrences - correct
    return {
        "reference_occurrences": occurrences,
        "correctly_recognized": correct,
        "missed": missed,
        "recall": correct / occurrences if occurrences else 0.0,
    }


def duration_bucket(duration: float | None) -> str:
    if duration is None:
        return "unknown"
    if duration < 5:
        return "<5s"
    if duration < 15:
        return "5-15s"
    if duration < 30:
        return "15-30s"
    return ">=30s"


def _aggregate(samples: Sequence[EvaluationSampleResult]) -> tuple[EditCounts, EditCounts, float | None, float | None, float | None]:
    successful = [item for item in samples if item.status == "SUCCESS"]
    char = EditCounts(
        sum(item.char_substitutions for item in successful),
        sum(item.char_deletions for item in successful),
        sum(item.char_insertions for item in successful),
        sum(item.char_reference_units for item in successful),
    )
    word = EditCounts(
        sum(item.word_substitutions for item in successful),
        sum(item.word_deletions for item in successful),
        sum(item.word_insertions for item in successful),
        sum(item.word_reference_units for item in successful),
    )
    duration = sum(item.duration or 0.0 for item in successful)
    processing = sum(item.processing_time or 0.0 for item in successful)
    return char, word, duration, processing, processing / duration if duration else None


def _breakdowns(samples: Sequence[EvaluationSampleResult]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    groups: dict[str, dict[str, list[EvaluationSampleResult]]] = {}
    for sample in samples:
        if sample.status != "SUCCESS":
            continue
        labels = {"duration_bucket": duration_bucket(sample.duration), **{str(k): str(v) for k, v in sample.metadata.items()}}
        for key, value in labels.items():
            groups.setdefault(key, {}).setdefault(value, []).append(sample)
    return {
        key: {
            value: {
                "samples": len(items),
                "cer": _aggregate(items)[0].rate,
                "wer": _aggregate(items)[1].rate,
                "rtf": _aggregate(items)[4],
            }
            for value, items in values.items()
        }
        for key, values in groups.items()
    }


def _error_analysis(char_ops: Sequence[tuple[str, str, str]], word_ops: Sequence[tuple[str, str, str]]) -> dict[str, list[dict[str, Any]]]:
    def top(operations: Sequence[tuple[str, str, str]], kind: str) -> list[dict[str, Any]]:
        counts = Counter((reference, prediction) for operation, reference, prediction in operations if operation == kind)
        return [{"reference": ref, "prediction": pred, "count": count} for (ref, pred), count in counts.most_common(20)]

    return {
        "top_character_substitutions": top(char_ops, "substitution"),
        "top_word_substitutions": top(word_ops, "substitution"),
        "common_deleted_words": top(word_ops, "deletion"),
        "common_inserted_words": top(word_ops, "insertion"),
    }


class ASREvaluator:
    """Evaluate one current adapter/model; future candidates reuse this unchanged evaluator."""

    def __init__(self, settings: Settings, processor: BatchASRProcessor | None = None) -> None:
        self.settings = settings
        self.processor = processor or BatchASRProcessor(settings)

    def evaluate(
        self,
        records: Sequence[EvaluationManifestRecord],
        *,
        dataset_id: str,
        domain_terms: Sequence[str] = (),
    ) -> EvaluationReport:
        batch = self.processor.transcribe_paths([record.audio_path for record in records])
        samples: list[EvaluationSampleResult] = []
        all_char_ops: list[tuple[str, str, str]] = []
        all_word_ops: list[tuple[str, str, str]] = []
        for record, item in zip(records, batch.items, strict=True):
            if item.status != "SUCCESS" or item.asr_result is None:
                samples.append(
                    EvaluationSampleResult(
                        sample_id=record.sample_id,
                        status="FAILED",
                        reference_raw=record.reference_text,
                        prediction_raw="",
                        reference_normalized=normalize_text(record.reference_text),
                        prediction_normalized="",
                        processing_time=item.processing_time,
                        metadata=record.metadata,
                        error=item.error,
                    )
                )
                continue
            prediction = item.asr_result.text
            char, char_ops = character_counts(record.reference_text, prediction)
            word, word_ops = word_counts(record.reference_text, prediction)
            all_char_ops.extend(char_ops)
            all_word_ops.extend(word_ops)
            samples.append(
                EvaluationSampleResult(
                    sample_id=record.sample_id,
                    status="SUCCESS",
                    reference_raw=record.reference_text,
                    prediction_raw=prediction,
                    reference_normalized=normalize_text(record.reference_text),
                    prediction_normalized=normalize_text(prediction),
                    cer=char.rate,
                    wer=word.rate,
                    char_substitutions=char.substitutions,
                    char_deletions=char.deletions,
                    char_insertions=char.insertions,
                    char_reference_units=char.reference_units,
                    word_substitutions=word.substitutions,
                    word_deletions=word.deletions,
                    word_insertions=word.insertions,
                    word_reference_units=word.reference_units,
                    duration=item.duration,
                    processing_time=item.processing_time,
                    rtf=item.rtf,
                    domain_metrics=domain_term_metrics(record.reference_text, prediction, domain_terms),
                    metadata=record.metadata,
                )
            )
        char, word, duration, processing, rtf = _aggregate(samples)
        domain = {
            key: sum(item.domain_metrics.get(key, 0) for item in samples)
            for key in ("reference_occurrences", "correctly_recognized", "missed")
        }
        domain["recall"] = domain["correctly_recognized"] / domain["reference_occurrences"] if domain["reference_occurrences"] else 0.0
        return EvaluationReport(
            dataset_id=dataset_id,
            normalization_version=NORMALIZATION_VERSION,
            tokenizer=TOKENIZER,
            total_samples=len(samples),
            successful_samples=sum(item.status == "SUCCESS" for item in samples),
            failed_samples=sum(item.status != "SUCCESS" for item in samples),
            total_audio_duration=duration,
            total_processing_time=processing,
            aggregate_cer=char.rate,
            aggregate_wer=word.rate,
            aggregate_rtf=rtf,
            char_counts=char,
            word_counts=word,
            domain_term_metrics=domain,
            breakdowns=_breakdowns(samples),
            error_analysis=_error_analysis(all_char_ops, all_word_ops),
            samples=samples,
        )


def export_evaluation_report(report: EvaluationReport, settings: Settings) -> tuple[Path, Path]:
    directory = prepare_output_dir(settings) / "evaluation"
    directory.mkdir(exist_ok=True)
    report_id = uuid4().hex
    json_path = directory / f"asr_evaluation_{report_id}.json"
    csv_path = directory / f"asr_evaluation_{report_id}.csv"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "sample_id", "status", "cer", "wer", "duration", "processing_time", "rtf",
        "reference_raw", "prediction_raw", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in report.samples:
            writer.writerow({field: getattr(sample, field) for field in fields})
    return json_path, csv_path
