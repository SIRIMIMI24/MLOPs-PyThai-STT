from importlib import metadata

from thai_offline_ai import diagnostics
from thai_offline_ai.config import AppConfig, Settings


def test_doctor_succeeds_without_optional_packages(tmp_path):
    checks = diagnostics.collect_diagnostics(Settings(app=AppConfig(output_dir=str(tmp_path))))
    assert not any(check.status == "FAIL" for check in checks)
    report = diagnostics.format_report(checks)
    assert "pythainlp" in report and "LLM URL" in report
    assert "local=True" in report
    assert "Optional; future phase" in report


def test_missing_required_dependency_fails_doctor(monkeypatch, tmp_path, capsys):
    original = metadata.version

    def missing(name):
        if name == "pythainlp":
            raise metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(metadata, "version", missing)
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path))
    assert diagnostics.main([]) == 1
    assert "[FAIL] pythainlp" in capsys.readouterr().out


def test_broken_import_is_reported(monkeypatch):
    def broken(name):
        raise OSError("DLL load failed")

    monkeypatch.setattr(diagnostics.importlib, "import_module", broken)
    check = diagnostics._package_check("gradio", "gradio", required=True, check_imports=True)
    assert check.status == "FAIL" and "OSError" in check.detail


def test_optional_packages_are_never_imported(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda name: "test")
    monkeypatch.setattr(diagnostics.importlib.util, "find_spec", lambda name: object())

    def forbidden(name):
        raise AssertionError("Optional library imported")

    monkeypatch.setattr(diagnostics.importlib, "import_module", forbidden)
    check = diagnostics._package_check("PyThaiTTS", "pythaitts", required=False, check_imports=True)
    assert check.status == "OK"
    assert "import not tested" in check.detail


def test_invalid_config_returns_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THAI_AI_HOST", "0.0.0.0")
    assert diagnostics.main([]) == 1
    assert "[FAIL] Configuration" in capsys.readouterr().out


def test_enabling_tts_without_model_reports_warning(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.tts.enabled = True
    settings.tts.model_dir = str(tmp_path / "missing-model")
    checks = diagnostics.collect_diagnostics(settings, check_imports=False)
    assert any(c.status == "WARN" and c.name == "Component tts" for c in checks)
