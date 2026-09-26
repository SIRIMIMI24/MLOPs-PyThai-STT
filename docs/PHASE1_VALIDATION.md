# Phase 1 validation record

Before Phase 2 began, the project virtual environment used Python 3.10.20 on
Windows with PyThaiNLP 5.3.7, python-crfsuite 0.9.12, and Gradio 5.50.0.

The resume audit ran compilation, `pytest -q`, Ruff, and doctor. The suite passed
with `133 passed, 4 warnings`; the warnings were Gradio 6 deprecations. Doctor
confirmed the NLP smoke test, output-directory write probe, offline socket guard,
and loopback UI settings. Optional TTS, ASR, and ONNX packages were absent then.
