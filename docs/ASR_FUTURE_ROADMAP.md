# Future ASR evaluation, synthetic data, and controlled adaptation

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | PyThaiNLP | Complete |
| 2 | Thai TTS | Complete |
| 3 | File-based Thai ASR | Complete |
| 3.5 | Near real-time streaming ASR | Complete |
| 3.6 | Multi-format and batch audio | Complete |
| 3.7 | ASR evaluation and accuracy lab | Complete |
| 4 | Local LLM | Current |
| 5 | Speaker diarization | Not started |
| 5.5 | Streaming ASR plus streaming diarization | Not started |
| 6 | Local voice assistant | Not started |
| 6.5 | Speech AI Continuous Improvement Lab: synthetic speech, pseudo-labeling, self-training, fine-tuning, experiment tracking, model promotion | Not started |
| 7 | Conversation intelligence | Not started |

## Phase 3.7: real-human ASR evaluation

Candidate ASR models are measured against verified transcripts from real human
audio. Character Error Rate (CER) is the primary Thai metric. Word Error Rate
(WER) uses one pinned PyThaiNLP tokenizer and a documented deterministic
normalization policy. Reports also include insertion, deletion, substitution,
domain-term accuracy, audio duration, inference time, and Real-Time Factor
(RTF = inference seconds / audio seconds).

Each evaluation preserves `raw_text` from ASR and stores `normalized_text`
separately. Normalization may use PyThaiNLP and a deterministic approved domain
dictionary; it must never overwrite raw model output. Synthetic TTS results may
be used for engineering smoke checks only and are never reported as real-human
accuracy.

## Dataset separation and provenance

```text
datasets/
  train_real/       # real speech with verified transcripts
  train_synthetic/  # generated speech with known source text
  unlabeled_real/   # real speech eligible for controlled pseudo-labeling
  validation_real/  # real speech for tuning and model selection
  test_real/        # fixed, held-out real speech for final comparison only
```

`test_real` is immutable with respect to training and model selection. Synthetic
audio must never enter `validation_real` or `test_real`. Dataset manifests carry
sample ID, split, source/provenance, consent status where applicable, transcript
state, audio checksum, and dataset version.

## Future Phase 6.5: local synthetic speech pipeline

```text
approved text corpus -> local TTS adapter -> synthetic audio -> augmentation -> training manifest
```

Every synthetic record stores `source_text`, `audio_path`, `tts_backend`, model,
voice where supported, generation parameters, augmentation parameters, generator
version/timestamp, and `synthetic=true`. The generator is adapter-based so future
local TTS backends can add available voices, rate/pitch styles, clean/telephone
conditions, noise, reverberation, codec effects, volume, silence, and sentence
length variation. It does not assume PyThaiTTS alone supplies all diversity.

Only supported synthetic voices or appropriately consented sources are allowed.
The system must not impersonate real people without authorization. Future domain
corpora may use curated vocabulary, approved templates, anonymized non-sensitive
text, and locally generated synthetic sentences. Confidential customer text is not
stored in the repository. Insurance-domain manifests may tag terms such as policy,
renewal, premium, coverage, payment appointment, product names, and Thai/English
mixed terminology for targeted evaluation.

## Controlled pseudo-labeling and training governance

```text
unlabeled_real -> fixed teacher -> pseudo transcript -> quality filter -> accepted data
                -> candidate student -> fixed real validation/test -> promote or reject
```

Pseudo-label records retain teacher model/version, raw transcript, normalized
transcript if produced, genuinely supplied quality/confidence data, acceptance or
rejection state, and provenance. Missing confidence remains missing; it is never
fabricated. Pseudo-labeling is filtered and reviewable, never uncontrolled recursive
training on every model output.

Each experiment creates a durable record containing `experiment_id`, base model,
real/synthetic data ratio, train and pseudo-label dataset versions, preprocessing
and augmentation configuration, training parameters, evaluation dataset version,
CER, WER, RTF, and model artifact/version. A candidate never replaces the baseline
automatically. Promotion requires measured improvement against the same unchanged
held-out real-human test criteria.

## Optimization order and anti-degradation guard

1. Establish a real-human baseline.
2. Improve audio preprocessing, VAD, and segmentation.
3. Add deterministic domain vocabulary handling.
4. Add synthetic training data and compare compatible models.
5. Add filtered pseudo-labeling.
6. Fine-tune only when evaluation justifies it.
7. Re-evaluate on the unchanged real-human holdout.
8. Promote or reject the candidate.

This prevents the harmful loop of synthetic data, model output, and blind retraining.
Every candidate must be compared with its baseline on fixed real-human evidence.
