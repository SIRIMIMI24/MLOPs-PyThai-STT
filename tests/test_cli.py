import json
import subprocess
import sys

import pytest

from thai_offline_ai.adapters.tts import TTSResult
from thai_offline_ai.cli import main


@pytest.mark.parametrize("args", [["nlp", "ประเทศไทย"], ["nlp", "--text", "ประเทศไทย"]])
def test_real_nlp_cli(args, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "outputs"))
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["token_count"] > 0
    assert result["normalized_text"] == "ประเทศไทย"


def test_module_cli_in_fresh_process(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text(f"app:\n  output_dir: '{tmp_path.as_posix()}/outputs'\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "thai_offline_ai.cli",
            "--config",
            str(config),
            "nlp",
            "ประเทศไทยมีประชากรประมาณเจ็ดสิบล้านคน",
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["token_count"] == 8


@pytest.mark.parametrize(
    "args",
    [
        ["tts", "--text", "PRIVATE_INPUT"],
        ["asr", "--audio", "PRIVATE_INPUT"],
        ["chat", "--message", "PRIVATE_INPUT"],
        ["nlp"],
        ["nlp", "a", "--text", "b"],
    ],
)
def test_unavailable_or_invalid_cli_is_actionable(args, capsys):
    assert main(args) == 2
    error = capsys.readouterr().err
    assert "[FAIL]" in error and "Traceback" not in error
    assert "PRIVATE_INPUT" not in error


def test_disabled_nlp_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("THAI_AI_NLP_ENABLED", "false")
    assert main(["nlp", "test"]) == 2
    assert "nlp.enabled=true" in capsys.readouterr().err


def test_invalid_config_cli(monkeypatch, capsys):
    monkeypatch.setenv("THAI_AI_PORT", "invalid")
    assert main(["nlp", "test"]) == 2
    assert "THAI_AI_PORT" in capsys.readouterr().err


def test_mocked_tts_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("THAI_AI_TTS_ENABLED", "true")
    monkeypatch.setattr(
        "thai_offline_ai.cli.TTSAdapter.synthesize",
        lambda self, text: TTSResult(str(tmp_path / "speech.wav"), 22050, "pythaitts", "lunarlist_onnx"),
    )
    assert main(["tts", "สวัสดีครับ"]) == 0
    assert json.loads(capsys.readouterr().out)["sample_rate"] == 22050
