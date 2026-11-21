from pathlib import Path

from thai_offline_ai.asr_types import ASRResult
from thai_offline_ai.benchmark import ASRBenchmark, export_benchmark_result
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError


class FakeAdapter:
    def transcribe_file(self, path, *, model=None, prompt_mode=None):
        if Path(path).name == "bad.wav":
            raise WorkbenchError("Unreadable local file.")
        return ASRResult(text="ทดสอบ", duration=2.0, backend="fake", model=model or "default")


def test_benchmark_measures_items_and_exports_local_reports(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "out")))
    result = ASRBenchmark(settings, adapter=FakeAdapter()).run(
        [tmp_path / "good.wav", tmp_path / "bad.wav"], model="openthai2_asr", prompt_mode="th-mix"
    )
    assert result.total_files == 2
    assert (result.succeeded, result.failed) == (1, 1)
    assert result.aggregate_rtf is not None
    assert result.items[0].rtf is not None
    assert result.items[1].error == "Unreadable local file."
    json_path, csv_path = export_benchmark_result(result, settings)
    assert json_path.parent == csv_path.parent == tmp_path / "out" / "benchmarks"
    assert '"total_files": 2' in json_path.read_text(encoding="utf-8")
    assert "good.wav" in csv_path.read_text(encoding="utf-8")


def test_benchmark_requires_input_files(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    try:
        ASRBenchmark(settings, adapter=FakeAdapter()).run([])
    except WorkbenchError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Expected a helpful empty-input error")
