"""Phase 6.5 — Automated tests for Speech AI Continuous Improvement Lab governance & pipelines."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from thai_offline_ai.adapters.tts import TTSResult
from thai_offline_ai.asr_types import ASRResult
from thai_offline_ai.cli import main as cli_main
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.evaluation import EditCounts, EvaluationReport
from thai_offline_ai.speech_lab.datasets import (
    DatasetManifest,
    DatasetSample,
    compute_file_sha256,
    validate_no_leakage,
)
from thai_offline_ai.speech_lab.experiments import (
    ExperimentRegistry,
    PromotionCriteria,
    export_training_bundle,
)
from thai_offline_ai.speech_lab.pseudo_label import (
    PseudoLabelFilterConfig,
    PseudoLabelingPipeline,
)
from thai_offline_ai.speech_lab.synthetic import (
    AUGMENTATION_PRESETS,
    AugmentationConfig,
    SyntheticCorpusItem,
    SyntheticSpeechGenerator,
    augment_waveform,
)


def _write_sine_wav(path: Path, *, sample_rate: int = 16000, duration_s: float = 1.0, freq: float = 440.0) -> Path:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    samples = 0.25 * np.sin(2.0 * np.pi * freq * t)
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def _fake_eval_report(
    dataset_id: str, *, cer: float, wer: float, rtf: float, domain_recall: float = 0.9
) -> EvaluationReport:
    return EvaluationReport(
        dataset_id=dataset_id,
        normalization_version="thai-asr-eval-v1",
        tokenizer="pythainlp:newmm",
        total_samples=10,
        successful_samples=10,
        failed_samples=0,
        total_audio_duration=25.0,
        total_processing_time=25.0 * rtf,
        aggregate_cer=cer,
        aggregate_wer=wer,
        aggregate_rtf=rtf,
        char_counts=EditCounts(0, 0, 0, 100),
        word_counts=EditCounts(0, 0, 0, 40),
        domain_term_metrics={
            "reference_occurrences": 10,
            "correctly_recognized": round(10 * domain_recall),
            "missed": 10 - round(10 * domain_recall),
            "recall": domain_recall,
        },
        breakdowns={},
        error_analysis={},
        samples=[],
    )


def test_dataset_governance_blocks_synthetic_holdout_and_audio_leakage(tmp_path):
    wav_real = _write_sine_wav(tmp_path / "real1.wav", freq=440.0)
    wav_other = _write_sine_wav(tmp_path / "real2.wav", freq=660.0)
    sha1 = compute_file_sha256(wav_real)
    sha2 = compute_file_sha256(wav_other)

    # Synthetic sample MUST NOT enter validation_real or test_real
    with pytest.raises(WorkbenchError, match="Governance violation"):
        DatasetSample(
            sample_id="bad_syn",
            split="test_real",
            audio_path=str(wav_real),
            source_text="ทดสอบ",
            synthetic=True,
            provenance="local_tts",
            consent_status="synthetic",
            transcript_state="synthetic_source",
            audio_sha256=sha1,
            dataset_version="v1",
        )

    holdout = DatasetManifest(
        dataset_id="thai_holdout",
        split="test_real",
        version="v1",
        samples=[
            DatasetSample(
                sample_id="hold_1",
                split="test_real",
                audio_path=str(wav_real),
                source_text="สวัสดีครับ",
                synthetic=False,
                provenance="real_consented",
                consent_status="consented",
                transcript_state="verified",
                audio_sha256=sha1,
                dataset_version="v1",
            )
        ],
    )
    leaking_train = DatasetManifest(
        dataset_id="thai_train",
        split="train_real",
        version="v1",
        samples=[
            DatasetSample(
                sample_id="train_dup",
                split="train_real",
                audio_path=str(wav_real),
                source_text="สวัสดีครับ",
                synthetic=False,
                provenance="real_consented",
                consent_status="consented",
                transcript_state="verified",
                audio_sha256=sha1,
                dataset_version="v1",
            )
        ],
    )
    clean_train = DatasetManifest(
        dataset_id="thai_train",
        split="train_real",
        version="v1",
        samples=[
            DatasetSample(
                sample_id="train_ok",
                split="train_real",
                audio_path=str(wav_other),
                source_text="ขอบคุณครับ",
                synthetic=False,
                provenance="real_consented",
                consent_status="consented",
                transcript_state="verified",
                audio_sha256=sha2,
                dataset_version="v1",
            )
        ],
    )

    with pytest.raises(WorkbenchError, match="Data leakage detected"):
        validate_no_leakage([leaking_train], [holdout])
    validate_no_leakage([clean_train], [holdout])

    # Round-trip JSONL & Phase 3.7 evaluation CSV export
    jsonl_path = holdout.save_jsonl(tmp_path / "holdout.jsonl")
    reloaded = DatasetManifest.load_jsonl(jsonl_path)
    assert reloaded.fingerprint == holdout.fingerprint
    csv_path = holdout.export_evaluation_csv(tmp_path / "holdout.csv")
    assert "reference_text" in csv_path.read_text(encoding="utf-8")


def test_synthetic_speech_generator_and_audio_augmentations(tmp_path):
    base_wav = _write_sine_wav(tmp_path / "tts_out.wav", sample_rate=24000, duration_s=0.8)

    class StubTTS:
        def synthesize(self, text: str, *, model=None, voice=None, speed=None) -> TTSResult:
            return TTSResult(
                audio_path=str(base_wav),
                sample_rate=24000,
                backend="kokoro_onnx",
                model=model or "kokoro_thai_onnx",
                voice=voice or "f_young_bright",
                speed=speed or 1.05,
            )

    # Verify waveform augmentation profiles preserve finite [-1, 1] bounds
    raw = np.sin(np.linspace(0, 40, 16000, dtype=np.float32)) * 0.3
    for preset in AUGMENTATION_PRESETS.values():
        aug = augment_waveform(raw, 16000, preset)
        assert aug.size > 0
        assert float(np.max(np.abs(aug))) <= 1.0

    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    gen = SyntheticSpeechGenerator(settings, tts_adapter=StubTTS())  # type: ignore[arg-type]
    summary = gen.generate_dataset(
        [
            SyntheticCorpusItem(
                sample_id="ins_01",
                text="ขอสอบถามเบี้ยประกันรถยนต์ชั้นหนึ่งครับ",
                domain_tags=("insurance", "premium"),
            )
        ],
        dataset_id="insurance_synth",
        dataset_version="v1",
        output_dir=tmp_path / "synth_out",
        augmentations=[
            AugmentationConfig(profile_name="clean"),
            AUGMENTATION_PRESETS["telephone_bandpass"],
        ],
    )
    assert summary.generated_count == 2
    assert summary.manifest_path.is_file()
    for sample in summary.manifest.samples:
        assert sample.synthetic is True
        assert sample.split == "train_synthetic"
        assert sample.metadata["tts_model"] == "kokoro_thai_onnx"
        assert Path(sample.audio_path).is_file()


def test_pseudo_labeling_quality_filters_and_no_fabricated_confidence(tmp_path):
    good_wav = _write_sine_wav(tmp_path / "good.wav", duration_s=2.0)
    loop_wav = _write_sine_wav(tmp_path / "loop.wav", duration_s=2.0)
    short_wav = _write_sine_wav(tmp_path / "short.wav", duration_s=0.1)

    outputs = {
        good_wav.resolve(): ASRResult(
            text="สวัสดีครับ ขอสอบถามเรื่องต่ออายุกรมธรรม์",
            segments=[],
            duration=2.0,
            backend="openthai_onnx",
            model="openthai2_asr",
            metadata={},  # No confidence supplied -> must remain None!
        ),
        loop_wav.resolve(): ASRResult(
            text="ครับครับครับครับครับครับครับครับ",
            segments=[],
            duration=2.0,
            backend="openthai_onnx",
            model="openthai2_asr",
        ),
        short_wav.resolve(): ASRResult(
            text="อืม",
            segments=[],
            duration=0.1,
            backend="openthai_onnx",
            model="openthai2_asr",
        ),
    }

    class StubTeacherASR:
        def transcribe_file(self, audio_path, *, model=None, prompt_mode=None) -> ASRResult:
            return outputs[Path(audio_path).resolve()]

    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    pipe = PseudoLabelingPipeline(settings, teacher_asr=StubTeacherASR())  # type: ignore[arg-type]
    report = pipe.run(
        [good_wav, loop_wav, short_wav],
        dataset_id="calls_batch1",
        output_dir=tmp_path / "pl_out",
        filters=PseudoLabelFilterConfig(),
    )
    assert report.accepted_count == 1
    assert report.rejected_count == 2
    accepted_rec = next(r for r in report.records if r.accepted)
    assert accepted_rec.confidence is None
    assert accepted_rec.raw_transcript == "สวัสดีครับ ขอสอบถามเรื่องต่ออายุกรมธรรม์"
    rejected_reasons = [reason for r in report.records if not r.accepted for reason in r.rejection_reasons]
    assert "repetition_loop_detected" in rejected_reasons
    assert any("too_short" in r for r in rejected_reasons)


def test_experiment_registry_promotion_gate_and_training_bundle(tmp_path):
    reg = ExperimentRegistry(tmp_path / "experiments.db")
    base_rep = _fake_eval_report("holdout_thai", cer=0.12, wer=0.22, rtf=0.25, domain_recall=0.80)
    better_rep = _fake_eval_report("holdout_thai", cer=0.09, wer=0.19, rtf=0.27, domain_recall=0.85)
    slow_rep = _fake_eval_report("holdout_thai", cer=0.08, wer=0.18, rtf=0.60, domain_recall=0.88)

    base = reg.record_from_evaluation(
        base_rep,
        base_model="openthai2_asr",
        candidate_model="openthai2_asr",
        eval_dataset_version="fp-v1",
        is_baseline=True,
    )
    good_cand = reg.record_from_evaluation(
        better_rep,
        base_model="openthai2_asr",
        candidate_model="openthai2_finetuned_v1",
        eval_dataset_version="fp-v1",
        real_sample_count=100,
        synthetic_sample_count=150,
    )
    slow_cand = reg.record_from_evaluation(
        slow_rep,
        base_model="openthai2_asr",
        candidate_model="openthai2_heavy_v2",
        eval_dataset_version="fp-v1",
    )
    mismatch_cand = reg.record_from_evaluation(
        better_rep,
        base_model="openthai2_asr",
        candidate_model="openthai2_wrong_eval",
        eval_dataset_version="fp-v2-DIFFERENT",
    )

    # 1. Genuine improvement on identical holdout -> PROMOTED
    d_ok = reg.promote_or_reject(base.experiment_id, good_cand.experiment_id, PromotionCriteria())
    assert d_ok.approved is True and d_ok.status == "promoted"
    assert reg.get(good_cand.experiment_id).promotion_status == "promoted"  # type: ignore[union-attr]

    # 2. RTF slowdown beyond 1.15x -> REJECTED
    d_slow = reg.promote_or_reject(base.experiment_id, slow_cand.experiment_id, PromotionCriteria())
    assert d_slow.approved is False and any("RTF slowdown" in r for r in d_slow.reasons)

    # 3. Mismatched holdout version -> REJECTED
    d_mis = reg.promote_or_reject(base.experiment_id, mismatch_cand.experiment_id, PromotionCriteria())
    assert d_mis.approved is False and any("version/fingerprint mismatch" in r for r in d_mis.reasons)

    # 4. Training bundle caps synthetic-to-real ratio
    real_wav = _write_sine_wav(tmp_path / "r.wav", freq=400.0)
    syn_wavs = [_write_sine_wav(tmp_path / f"s{i}.wav", freq=500.0 + i * 20) for i in range(5)]
    train_real = DatasetManifest(
        dataset_id="r",
        split="train_real",
        version="v1",
        samples=[
            DatasetSample(
                sample_id="r1",
                split="train_real",
                audio_path=str(real_wav),
                source_text="จริง",
                synthetic=False,
                provenance="real",
                consent_status="consented",
                transcript_state="verified",
                audio_sha256=compute_file_sha256(real_wav),
                dataset_version="v1",
            )
        ],
    )
    train_syn = DatasetManifest(
        dataset_id="s",
        split="train_synthetic",
        version="v1",
        samples=[
            DatasetSample(
                sample_id=f"s{i}",
                split="train_synthetic",
                audio_path=str(w),
                source_text=f"สังเคราะห์ {i}",
                synthetic=True,
                provenance="tts",
                consent_status="synthetic",
                transcript_state="synthetic_source",
                audio_sha256=compute_file_sha256(w),
                dataset_version="v1",
            )
            for i, w in enumerate(syn_wavs)
        ],
    )
    bundle = export_training_bundle(
        output_dir=tmp_path / "bundle",
        bundle_name="exp1",
        base_model="openthai2_asr",
        train_real=train_real,
        train_synthetic=train_syn,
        max_synthetic_to_real_ratio=2.0,
    )
    assert bundle.capped_synthetic is True
    assert bundle.real_count == 1 and bundle.synthetic_count == 2 and bundle.total_count == 3
    assert bundle.manifest_path.is_file() and bundle.recipe_path.is_file()


def test_phase65_cli_generate_synthetic_and_evaluate_promote(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    tts_wav = _write_sine_wav(tmp_path / "mock_tts.wav", sample_rate=24000, duration_s=0.5)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("สวัสดีครับ ทดสอบระบบ\n", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("app:\n  output_dir: outputs\ntts:\n  enabled: true\nasr:\n  enabled: true\n", encoding="utf-8")

    monkeypatch.setattr(
        "thai_offline_ai.adapters.tts.TTSAdapter.synthesize",
        lambda self, text, *, model=None, voice=None, speed=None: TTSResult(
            audio_path=str(tts_wav),
            sample_rate=24000,
            backend="kokoro_onnx",
            model="kokoro_thai_onnx",
            voice="f_young_bright",
            speed=1.05,
        ),
    )
    rc = cli_main(
        [
            "--config",
            str(cfg),
            "lab-generate-synthetic",
            str(corpus),
            "--dataset-id",
            "cli_syn",
            "--profiles",
            "clean,gaussian_noise",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["generated_count"] == 2
    assert Path(out["manifest_path"]).is_file()
