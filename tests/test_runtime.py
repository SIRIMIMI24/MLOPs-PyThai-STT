from pathlib import Path

import pytest

from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import ConfigurationError, WorkbenchError
from thai_offline_ai.runtime import output_path, prepare_output_dir, prepare_runtime


def settings_for(path):
    return Settings(app=AppConfig(output_dir=str(path)))


def test_output_created_without_probe_remnants(tmp_path):
    directory = tmp_path / "new" / "outputs"
    assert prepare_output_dir(settings_for(directory)) == directory
    assert list(directory.iterdir()) == []


def test_existing_file_is_not_an_output_directory(tmp_path):
    path = tmp_path / "file"
    path.write_text("keep")
    with pytest.raises(WorkbenchError, match="THAI_AI_OUTPUT_DIR"):
        prepare_output_dir(settings_for(path))
    assert path.read_text() == "keep"


def test_permission_failure_is_actionable(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("thai_offline_ai.runtime.tempfile.TemporaryFile", denied)
    with pytest.raises(WorkbenchError, match="writable local folder"):
        prepare_output_dir(settings_for(tmp_path / "outputs"))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "sub/file", "..", "C:\\file", "a:b"])
def test_output_cannot_escape(tmp_path, name):
    with pytest.raises(WorkbenchError):
        output_path(settings_for(tmp_path / "outputs"), name)


def test_output_symlink_escape(tmp_path, monkeypatch):
    directory = tmp_path / "outputs"
    settings = settings_for(directory)
    prepare_output_dir(settings)
    original = Path.resolve

    def redirected(path, *args, **kwargs):
        if path.name == "linked.txt":
            return tmp_path / "elsewhere.txt"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(WorkbenchError, match="escapes"):
        output_path(settings, "linked.txt")


def test_drive_root_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="drive root"):
        prepare_output_dir(settings_for(tmp_path.anchor))


def test_runtime_places_gradio_cache_under_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "outside"))
    directory = prepare_runtime(settings_for(tmp_path / "outputs"))
    import os

    assert os.environ["GRADIO_TEMP_DIR"] == str(directory / ".gradio")
