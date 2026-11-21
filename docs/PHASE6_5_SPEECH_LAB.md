# Phase 6.5 — Speech AI Continuous Improvement Lab

## Overview
Phase 6.5 implements the governed local ASR improvement lifecycle specified in [`ASR_FUTURE_ROADMAP.md`](ASR_FUTURE_ROADMAP.md):

1. **Dataset Separation & Leakage Guards** (`thai_offline_ai.speech_lab.datasets`)
   - Enforces the 5 canonical splits (`train_real`, `train_synthetic`, `unlabeled_real`, `validation_real`, `test_real`).
   - Hard-blocks synthetic samples (`synthetic=True`) from entering `validation_real` or `test_real`.
   - Computes deterministic SHA-256 audio checksums and manifest fingerprints; `validate_no_leakage()` blocks any training sample whose audio hash overlaps a holdout split.
2. **Local Synthetic Speech & Augmentation Pipeline** (`thai_offline_ai.speech_lab.synthetic`)
   - Uses the shared local `TTSAdapter` (Kokoro Thai ONNX / PyThaiTTS) and deterministic CPU NumPy/SciPy augmentations (`clean`, `gaussian_noise`, `telephone_bandpass` 300–3400 Hz, `fast_quiet`, silence padding, speed perturbation).
   - Stores complete provenance (`tts_backend`, `tts_model`, `voice`, `speed`, `augmentation`, `generator_version`, `synthetic=True`, `audio_sha256`).
3. **Controlled Pseudo-Labeling** (`thai_offline_ai.speech_lab.pseudo_label`)
   - Runs one fixed teacher `ASRAdapter` on `unlabeled_real` audio, preserving both `raw_transcript` and `normalized_transcript` (never overwrites raw output, never fabricates missing confidence).
   - Applies transparent deterministic quality filters (character length, duration bounds, speaking-rate chars/sec, optional confidence threshold, repetition-loop detection) and writes reviewable accepted/rejected JSON reports.
4. **Experiment Registry, Training Bundle & Promotion Gate** (`thai_offline_ai.speech_lab.experiments`)
   - Local SQLite registry (`outputs/speech_lab/experiments.db`) recording model metadata, real/synthetic/pseudo counts, dataset versions, CER/WER/RTF, domain term recall, and promotion status.
   - `evaluate_promotion()` enforces the **anti-degradation guard**: requires matching `eval_dataset_id` + `eval_dataset_version` on a real holdout (`validation_real` or `test_real`), measurable CER improvement, bounded WER/domain-recall, and bounded RTF ratio.
   - `export_training_bundle()` exports a leakage-checked JSONL training manifest and reproducible RTX 4050 (6 GB VRAM) / CPU training recipe while capping the synthetic-to-real ratio.

## CLI Usage

```powershell
# 1. Generate synthetic speech + telephone augmentation from an approved text corpus
.\.venv\Scripts\python.exe -m thai_offline_ai.cli lab-generate-synthetic corpus.txt --dataset-id insurance_v1 --profiles clean,telephone_bandpass

# 2. Run controlled pseudo-labeling over an unlabeled_real folder
.\.venv\Scripts\python.exe -m thai_offline_ai.cli lab-pseudo-label datasets\unlabeled_real --dataset-id call_batch1 --model openthai2_asr --prompt th-mix

# 3. Register a baseline on immutable test_real, then evaluate and gate a candidate
.\.venv\Scripts\python.exe -m thai_offline_ai.cli lab-evaluate-promote datasets\test_real\manifest.csv --eval-version v1 --register-baseline
.\.venv\Scripts\python.exe -m thai_offline_ai.cli lab-evaluate-promote datasets\test_real\manifest.csv --eval-version v1 --candidate-model candidate_v2 --baseline-id <baseline_exp_id>
```
