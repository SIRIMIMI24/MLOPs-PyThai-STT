# ASR evaluation dataset contract

Keep real-human audio and evaluation manifests out of Git. This directory tracks
only this guidance and `manifest.example.csv`; actual datasets should live in a
separate local directory or storage governed by the appropriate consent and privacy
policy. Use UTF-8 CSV with `sample_id`, `audio_path`, and `reference_text`; relative
audio paths resolve from the manifest directory.

Optional columns include `audio_condition`, `domain`, `speaker_group`, `duration`,
`source_format`, and `notes`. Do not require or collect sensitive demographic fields
solely for this evaluation. Ground truth must be reviewed human transcription, and a
held-out real-human test manifest must never be used for training or model selection.

Use currently verified WAV, FLAC, MP3, or OGG audio. Before evaluation, run:

```powershell
.\.venv\Scripts\python.exe -m thai_offline_ai.cli evaluate-asr manifest.csv
```

Use `--domain-lexicon domain_terms.yaml` only for an approved versioned vocabulary
list. It accepts `terms:` followed by a YAML list of non-sensitive strings. The
resulting JSON and CSV reports are written under `outputs\evaluation` and are ignored
by Git. Synthetic TTS manifests are allowed only for engineering checks and must be
labelled synthetic; they are not a real-human accuracy benchmark.
