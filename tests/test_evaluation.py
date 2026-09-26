import csv
from pathlib import Path

import pytest

from thai_offline_ai.asr_types import ASRResult
from thai_offline_ai.batch import BatchASRItemResult, BatchASRProcessor, BatchASRResult
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.evaluation import (
    NORMALIZATION_VERSION,
    ASREvaluator,
    character_counts,
    domain_term_metrics,
    duration_bucket,
    edit_counts,
    export_evaluation_report,
    load_manifest,
    normalize_text,
    thai_tokens,
    word_counts,
)


def test_normalization_is_deterministic_for_thai_unicode_whitespace_and_punctuation():
    assert normalize_text("สวัสดี  ครับ,  โลก!") == "สวัสดี ครับ โลก"
    assert normalize_text("ＡＢＣ") == "ABC"
    assert NORMALIZATION_VERSION == "thai-asr-eval-v1"


def test_character_edit_counts_exact_substitution_insertion_deletion_and_empty_cases():
    counts, _ = edit_counts(list("abc"), list("adc"))
    assert (counts.substitutions, counts.deletions, counts.insertions, counts.reference_units) == (1, 0, 0, 3)
    insertion, _ = edit_counts([], list("a"))
    assert insertion.rate is None
    deletion, _ = edit_counts(list("a"), [])
    assert deletion.rate == 1
    thai, _ = character_counts("สวัสดี ครับ", "สวัสดีครับ")
    assert thai.rate == 0


def test_thai_word_tokenizer_and_word_counts_are_consistent():
    assert thai_tokens("ประเทศไทย") == thai_tokens("ประเทศไทย")
    counts, _ = word_counts("สวัสดีครับ", "สวัสดีค่ะ")
    assert counts.reference_units > 0 and counts.rate is not None


def test_domain_terms_and_duration_buckets():
    metrics = domain_term_metrics("ต่ออายุกรมธรรม์", "ต่ออายุ", ["ต่ออายุ", "กรมธรรม์"])
    assert metrics == {"reference_occurrences": 2, "correctly_recognized": 1, "missed": 1, "recall": 0.5}
    assert [duration_bucket(value) for value in (4, 5, 15, 30, None)] == ["<5s", "5-15s", "15-30s", ">=30s", "unknown"]


class FakeProcessor:
    def transcribe_paths(self, paths):
        success = BatchASRItemResult(
            file_name=Path(paths[0]).name,
            source_path=str(paths[0]),
            status="SUCCESS",
            asr_result=ASRResult(text="สวัสดี", duration=2.0, model="typhoon_asr"),
            duration=2.0,
            processing_time=1.0,
            rtf=0.5,
        )
        failed = BatchASRItemResult(
            file_name=Path(paths[1]).name,
            source_path=str(paths[1]),
            status="FAILED",
            processing_time=0.1,
            error="bad audio",
        )
        return BatchASRResult("batch", 2, 1, 1, 2.0, 1.1, [success, failed])


def make_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "audio_path", "reference_text", "audio_condition"])
        writer.writeheader()
        writer.writerow({"sample_id": "one", "audio_path": "one.wav", "reference_text": "สวัสดี", "audio_condition": "clean"})
        writer.writerow({"sample_id": "two", "audio_path": "two.wav", "reference_text": "โลก", "audio_condition": "noisy"})
    return manifest


def test_manifest_evaluator_aggregate_breakdowns_and_exports(tmp_path):
    records = load_manifest(make_manifest(tmp_path))
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    report = ASREvaluator(settings, processor=FakeProcessor()).evaluate(records, dataset_id="fixture", domain_terms=["สวัสดี"])
    assert report.successful_samples == 1 and report.failed_samples == 1
    assert report.aggregate_cer == 0 and report.aggregate_wer == 0
    assert report.aggregate_rtf == 0.5
    assert report.breakdowns["audio_condition"]["clean"]["samples"] == 1
    assert report.samples[1].error == "bad audio"
    json_path, csv_path = export_evaluation_report(report, settings)
    assert json_path.is_file() and csv_path.is_file()
    assert "sample_id" in csv_path.read_text(encoding="utf-8")


def test_manifest_validation_rejects_missing_required_duplicate_or_missing_file(tmp_path):
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("sample_id,audio_path\nx,a.wav\n", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="requires"):
        load_manifest(invalid)
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("sample_id,audio_path,reference_text\nx,a.wav,a\nx,b.wav,b\n", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="Duplicate"):
        load_manifest(duplicate)
    with pytest.raises(WorkbenchError, match="existing"):
        load_manifest(tmp_path / "missing.csv")


def test_missing_manifest_audio_becomes_a_failed_evaluation_sample(tmp_path):
    manifest = tmp_path / "missing-audio.csv"
    manifest.write_text(
        "sample_id,audio_path,reference_text\nmissing,not-present.wav,สวัสดี\n",
        encoding="utf-8",
    )
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    report = ASREvaluator(settings, processor=BatchASRProcessor(settings)).evaluate(
        load_manifest(manifest), dataset_id="missing-audio"
    )
    assert report.failed_samples == 1
    assert report.samples[0].error == "ASR input must be a non-empty readable audio file."
