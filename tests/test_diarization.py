import json

import numpy as np
import pytest
import soundfile as sf

from thai_offline_ai.adapters.diarization import (
    DiarizationAdapter,
    normalize_turns,
    resolve_device,
)
from thai_offline_ai.asr_types import ASRResult, SpeechSegment
from thai_offline_ai.cli import main
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.diarization_types import DiarizationResult, SpeakerTurn
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.speaker_composition import compose_speaker_transcript


def test_diarization_contracts_are_serializable_and_validate_turns():
    turn = SpeakerTurn(0.0, 1.0, "SPEAKER_00")
    result = DiarizationResult(turns=[turn], speakers=[turn.speaker])
    assert result.to_dict()["turns"][0]["speaker"] == "SPEAKER_00"
    assert DiarizationResult().turns is not DiarizationResult().turns
    with pytest.raises(ValueError, match="end"):
        SpeakerTurn(1.0, 1.0, "SPEAKER_00")


def test_normalize_turns_orders_and_preserves_backend_labels():
    turns = normalize_turns(
        [
            {"start": 2.0, "end": 3.0, "speaker": "beta"},
            {"start": 0.0, "end": 1.0, "speaker": "alpha"},
            {"start": 1.0, "end": 2.0, "speaker": "beta"},
        ]
    )
    assert [(turn.start, turn.speaker) for turn in turns] == [
        (0.0, "SPEAKER_00"),
        (1.0, "SPEAKER_01"),
        (2.0, "SPEAKER_01"),
    ]
    assert turns[0].metadata["backend_speaker"] == "alpha"


def test_device_selection_cpu_auto_and_unavailable_cuda(monkeypatch):
    monkeypatch.setattr("thai_offline_ai.adapters.diarization.cuda_is_usable", lambda: False)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") == "cpu"
    with pytest.raises(WorkbenchError, match="CUDA diarization"):
        resolve_device("cuda")
    monkeypatch.setattr("thai_offline_ai.adapters.diarization.cuda_is_usable", lambda: True)
    assert resolve_device("auto") == "cuda" and resolve_device("cuda") == "cuda"


def test_diarization_adapter_is_lazy_and_normalizes_real_audio(tmp_path):
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, np.zeros(1600), 16000)
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    settings.diarization.enabled = True
    adapter = DiarizationAdapter(settings)
    adapter._backend = lambda *args, **kwargs: [
        {"start": 0.0, "end": 0.05, "speaker": "speaker_a"},
        {"start": 0.05, "end": 0.1, "speaker": "speaker_b"},
    ]
    result = adapter.diarize_file(audio_path)
    assert result.device == "cpu"
    assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert result.metadata["normalized_audio"]["sample_rate"] == 16000
    assert result.metadata["processing_seconds"] >= 0


def test_diarization_disabled_and_missing_cache_are_actionable(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    with pytest.raises(WorkbenchError, match="disabled"):
        DiarizationAdapter(settings)._load()
    settings.diarization.enabled = True
    settings.diarization.model_dir = str(tmp_path / "missing-model")
    with pytest.raises(WorkbenchError, match="setup_diarization"):
        DiarizationAdapter(settings)._load()


def test_timestamp_alignment_preserves_asr_and_marks_ambiguity():
    asr = ASRResult(
        text="ข้อความ",
        segments=[SpeechSegment("ข้อความ", 0.0, 10.0), SpeechSegment("ไม่มีเวลา")],
    )
    diarization = DiarizationResult(
        turns=[SpeakerTurn(0.0, 5.0, "SPEAKER_00"), SpeakerTurn(5.0, 10.0, "SPEAKER_01")],
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    combined = compose_speaker_transcript(asr, diarization)
    assert combined.segments[0].speaker == "SPEAKER_00"
    assert combined.segments[0].metadata["speaker_alignment_tie"] is True
    assert combined.segments[0].metadata["speaker_alignment_ambiguous"] is True
    assert combined.segments[1].speaker is None
    assert combined.segments[1].metadata["speaker_alignment"] == "missing_asr_timestamps"
    assert asr.segments[0].speaker is None


def test_cli_diarize_and_transcribe_speakers_with_fakes(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("THAI_AI_ASR_ENABLED", "true")
    monkeypatch.setenv("THAI_AI_DIARIZATION_ENABLED", "true")
    diarization = DiarizationResult([SpeakerTurn(0.0, 1.0, "SPEAKER_00")], ["SPEAKER_00"])
    asr = ASRResult("ทดสอบ", [SpeechSegment("ทดสอบ", 0.0, 1.0)])
    monkeypatch.setattr(DiarizationAdapter, "diarize_file", lambda self, path: diarization)
    monkeypatch.setattr("thai_offline_ai.cli.ASRAdapter.transcribe_file", lambda self, path: asr)
    assert main(["diarize", "input.wav"]) == 0
    assert json.loads(capsys.readouterr().out)["speakers"] == ["SPEAKER_00"]
    assert main(["transcribe-speakers", "input.wav"]) == 0
    assert json.loads(capsys.readouterr().out)["segments"][0]["speaker"] == "SPEAKER_00"
