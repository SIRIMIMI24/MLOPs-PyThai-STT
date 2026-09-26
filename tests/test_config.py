import pytest

from thai_offline_ai.config import ConfigurationError, load_settings


def test_defaults_without_default_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.app.host == "127.0.0.1"
    assert settings.app.offline_runtime is True
    assert settings.app.output_dir == str(tmp_path / "outputs")
    assert settings.nlp.enabled and settings.nlp.word_engine == "newmm"
    assert not any((settings.tts.enabled, settings.asr.enabled, settings.llm.enabled))


def test_yaml_and_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / "settings.yaml"
    path.write_text("app:\n  port: 8000\n  log_level: INFO\nnlp:\n  enabled: false\n")
    monkeypatch.setenv("THAI_AI_CONFIG", str(path))
    monkeypatch.setenv("THAI_AI_PORT", "9000")
    monkeypatch.setenv("THAI_AI_NLP_ENABLED", "true")
    monkeypatch.setenv("THAI_AI_OUTPUT_DIR", str(tmp_path / "results"))
    settings = load_settings()
    assert settings.app.port == 9000
    assert settings.app.log_level == "INFO"
    assert settings.nlp.enabled is True
    assert settings.app.output_dir == str(tmp_path / "results")


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "false",
        "app: 5",
        "app: null",
        "app: [",
        "typo: {}",
        "app: {porrt: 8000}",
        'app: {port: "8000"}',
        "app: {port: true}",
        'app: {offline_runtime: "false"}',
        "app: {port: 0}",
        "app: {port: 65536}",
        "app: {host: 0.0.0.0}",
        "app: {log_level: VERBOSE}",
        'app: {output_dir: ""}',
        'app: {output_dir: "//server/share"}',
        "nlp: {word_engine: deepcut}",
        "nlp: {sentence_engine: wtp}",
        "nlp: {backend: cloud}",
        "tts: {backend: cloud}",
        "asr: {backend: cloud}",
        "llm: {backend: openai}",
        "llm: {base_url: 'https://api.example.com'}",
        "app: {offline_runtime: false}\nllm: {base_url: 'https://api.example.com'}",
        "asr: {sampling_rate: -1}",
        "llm: {temperature: .nan}",
        "llm: {max_tokens: 0}",
        "tts: {voice: 12}",
    ],
)
def test_invalid_yaml_config(tmp_path, contents):
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(path)


@pytest.mark.parametrize(
    "name,value",
    [
        ("THAI_AI_PORT", "oops"),
        ("THAI_AI_OFFLINE_RUNTIME", "maybe"),
        ("THAI_AI_HOST", "192.168.1.2"),
        ("THAI_AI_LLM_BASE_URL", "https://cloud.test"),
        ("THAI_AI_NLP_ENABLED", ""),
        ("THAI_AI_CONFIG", ""),
    ],
)
def test_invalid_environment(name, value, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        load_settings()


def test_explicit_missing_config_fails(tmp_path):
    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        load_settings(tmp_path / "missing.yaml")


def test_example_is_valid():
    settings = load_settings("config/settings.example.yaml")
    assert settings.nlp.enabled
    assert not settings.tts.enabled


def test_env_file_is_not_implicitly_executed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("THAI_AI_OFFLINE_RUNTIME=false")
    assert load_settings().app.offline_runtime is True


@pytest.mark.parametrize(
    "contents",
    [
        "app: {port: 8000, port: 9000}",
        "nlp: {enabled: true}\nnlp: {enabled: false}",
        "app: {123: 8000}",
    ],
)
def test_duplicate_and_non_string_keys_are_rejected(tmp_path, contents):
    path = tmp_path / "bad.yaml"
    path.write_text(contents)
    with pytest.raises(ConfigurationError):
        load_settings(path)


def test_unknown_environment_override_is_not_ignored(monkeypatch):
    monkeypatch.setenv("THAI_AI_OFFLINE_RUNTME", "false")
    with pytest.raises(ConfigurationError, match="Unknown environment"):
        load_settings()
