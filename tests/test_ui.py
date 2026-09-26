import asyncio
import socket

import httpx
import pytest

from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import ConfigurationError
from thai_offline_ai.ui import CONTENT_SECURITY_POLICY, build_app, launch_app


def test_real_ui_construction_and_callback(tmp_path):
    demo, _ = build_app(Settings(app=AppConfig(output_dir=str(tmp_path))))
    try:
        assert demo.analytics_enabled is False
        assert not demo.config.get("stylesheets")
        tabs = [c["props"]["label"] for c in demo.config["components"] if c["type"] == "tabitem"]
        assert tabs == ["Home", "NLP", "TTS", "ASR", "Local LLM", "System"]
        fn = next(key for key, value in demo.fns.items() if value.api_name == "analyze")
        result = asyncio.run(demo.process_api(fn, inputs=["ประเทศไทย"]))
        assert result["data"][0] == "ประเทศไทย"
        assert result["data"][3] > 0
    finally:
        demo.close()


def test_disabled_nlp_ui(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.nlp.enabled = False
    demo, _ = build_app(settings)
    try:
        source = next(
            c for c in demo.config["components"] if c["props"].get("label") == "ข้อความภาษาไทย"
        )
        assert source["props"]["interactive"] is False
    finally:
        demo.close()


def test_tts_enabled_ui_does_not_load_model(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.tts.enabled = True

    def unexpected_model_load(self):
        raise AssertionError("TTS must not load during UI construction")

    monkeypatch.setattr(TTSAdapter, "_get_model", unexpected_model_load)
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "เสียงที่สร้าง" for c in demo.config["components"])
    finally:
        demo.close()


def test_asr_live_ui_does_not_load_model(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.asr.enabled = True

    def unexpected_model_load(self):
        raise AssertionError("ASR must not load during UI construction")

    from thai_offline_ai.streaming import StreamingASRService

    monkeypatch.setattr(StreamingASRService, "_load_model", unexpected_model_load)
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "ไมโครโฟน (Live)" for c in demo.config["components"])
    finally:
        demo.close()


def test_batch_asr_ui_constructs_without_loading_model(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.asr.enabled = True

    def unexpected_model_load(self):
        raise AssertionError("ASR must not load during UI construction")

    monkeypatch.setattr(TTSAdapter, "_get_model", unexpected_model_load)
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "ผลลัพธ์แบบชุด" for c in demo.config["components"])
    finally:
        demo.close()


def test_public_bind_rejected_before_gradio_construction(tmp_path):
    with pytest.raises(ConfigurationError, match="app.host"):
        build_app(Settings(app=AppConfig(host="0.0.0.0", output_dir=str(tmp_path))))


def test_real_local_server_and_nlp_request(tmp_path, monkeypatch):
    # Reserve an ephemeral loopback port, then release it for Gradio.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("GRADIO_SHARE", "True")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("GRADIO_ROOT_PATH", "https://example.com")
    monkeypatch.setenv("GRADIO_SSR_MODE", "True")
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:8080")
    monkeypatch.setenv("http_proxy", "http://192.0.2.1:8080")
    settings = Settings(app=AppConfig(port=port, output_dir=str(tmp_path)))
    demo, settings = build_app(settings)
    try:
        _, url, share_url = launch_app(demo, settings, prevent_thread_lock=True)
        assert url.startswith("http://127.0.0.1:") and not share_url
        assert demo.share is False and demo.ssr_mode is False
        with httpx.Client(base_url=url, trust_env=False, timeout=15) as client:
            page = client.get("/")
            assert page.status_code == 200
            assert page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
            result = client.post("/gradio_api/run/analyze", json={"data": ["ประเทศไทย"]})
            assert result.status_code == 200, result.text
            assert result.json()["data"][0] == "ประเทศไทย"
            assert result.json()["data"][3] > 0
    finally:
        demo.close()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0, (
            "Server must be closed after verification"
        )
