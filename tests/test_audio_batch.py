from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from thai_offline_ai.asr_types import ASRResult
from thai_offline_ai.audio import SUPPORTED_AUDIO_EXTENSIONS, prepare_audio
from thai_offline_ai.batch import BatchASRProcessor, export_batch_result
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError


def write_audio(path: Path, rate: int = 22050, channels: int = 1) -> None:
    samples = np.zeros((rate // 10, channels), dtype=np.float32)
    sf.write(path, samples, rate)


def test_audio_normalizes_stereo_22050_to_mono_16000(tmp_path):
    path = tmp_path / "stereo.wav"
    write_audio(path, channels=2)
    audio = prepare_audio(path)
    assert audio.original.sample_rate == 22050
    assert audio.original.channels == 2
    assert audio.normalized.sample_rate == 16000
    assert audio.normalized.channels == 1
    assert len(audio.samples) == 1600


def test_audio_preserves_supported_format_metadata(tmp_path):
    path = tmp_path / "audio.flac"
    write_audio(path)
    audio = prepare_audio(path)
    assert audio.original.format == "flac"
    assert audio.original.duration == pytest.approx(0.1)


def test_audio_rejects_unsupported_or_corrupt_files(tmp_path):
    unsupported = tmp_path / "audio.m4a"
    unsupported.write_bytes(b"not audio")
    with pytest.raises(WorkbenchError, match="Unsupported"):
        prepare_audio(unsupported)
    corrupt = tmp_path / "audio.wav"
    corrupt.write_bytes(b"broken")
    with pytest.raises(WorkbenchError, match="readable"):
        prepare_audio(corrupt)


class FakeAdapter:
    def __init__(self):
        self.paths = []

    def transcribe_file(self, path):
        self.paths.append(Path(path).name)
        if Path(path).name.startswith("bad"):
            raise WorkbenchError("bad audio")
        return ASRResult(
            text="ทดสอบ",
            duration=1.0,
            model="typhoon_asr",
            metadata={
                "original_audio": {"format": "wav", "sample_rate": 16000, "channels": 1, "duration": 1.0},
                "normalized_audio": {"format": "wav-pcm16", "sample_rate": 16000, "channels": 1, "duration": 1.0},
            },
        )


def test_batch_isolates_failure_reuses_adapter_and_exports(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    adapter = FakeAdapter()
    processor = BatchASRProcessor(settings, adapter=adapter)
    result = processor.transcribe_paths([tmp_path / "good.wav", tmp_path / "bad.wav", tmp_path / "good2.wav"])
    assert result.total_files == 3 and result.succeeded == 2 and result.failed == 1
    assert adapter.paths == ["good.wav", "bad.wav", "good2.wav"]
    assert result.items[1].error == "bad audio"
    json_path, csv_path = export_batch_result(result, settings)
    assert json_path.is_file() and csv_path.is_file()
    assert json_path.parent == Path(settings.app.output_dir) / "batch"
    assert "transcript" in csv_path.read_text(encoding="utf-8")


def test_repeated_batch_exports_do_not_collide(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    processor = BatchASRProcessor(settings, adapter=FakeAdapter())
    first = processor.transcribe_paths([])
    second = processor.transcribe_paths([])
    first_paths = export_batch_result(first, settings)
    second_paths = export_batch_result(second, settings)
    assert first_paths != second_paths


def test_batch_directory_filters_and_handles_empty_batch(tmp_path):
    write_audio(tmp_path / "first.wav")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    processor = BatchASRProcessor(Settings(app=AppConfig(output_dir=str(tmp_path / "out"))), adapter=FakeAdapter())
    assert processor.paths_in_directory(tmp_path) == [tmp_path / "first.wav"]
    empty = processor.transcribe_paths([])
    assert empty.total_files == 0 and empty.items == []
    assert {".wav", ".flac", ".mp3", ".ogg"} <= SUPPORTED_AUDIO_EXTENSIONS
