import asyncio
import socket
from types import SimpleNamespace

import httpx
import pytest

from thai_offline_ai.adapters.tts import TTSAdapter, TTSResult
from thai_offline_ai.agent.schemas import (
    ApprovalRequest,
    PermissionLevel,
    SourceLink,
    TimelineStep,
)
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.diagnostics import Check
from thai_offline_ai.errors import ConfigurationError
from thai_offline_ai.llm_types import LLMResult
from thai_offline_ai.ui import CONTENT_SECURITY_POLICY, build_app, launch_app
from thai_offline_ai.ui_theme import (
    CUSTOM_CSS,
    HEAD_SCRIPT,
    build_theme,
    collect_ui_readiness,
    render_footer,
    render_home_html,
    render_jarvis_approval_card,
    render_jarvis_sources,
    render_jarvis_status_card,
    render_jarvis_timeline,
    render_navbar,
    render_page_header,
    render_system_cards,
    render_tts_audio_summary,
)


def test_real_ui_construction_and_callback(tmp_path):
    demo, _ = build_app(Settings(app=AppConfig(output_dir=str(tmp_path))))
    try:
        assert demo.analytics_enabled is False
        assert not demo.config.get("stylesheets")
        tabs = [c["props"]["label"] for c in demo.config["components"] if c["type"] == "tabitem"]
        assert tabs == ["Home", "NLP", "TTS", "ASR", "Local LLM", "JARVIS AI Assistant", "System"]
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
    monkeypatch.setattr(TTSAdapter, "_get_kokoro_model", unexpected_model_load)
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
    monkeypatch.setattr(StreamingASRService, "_load_openthai_model", unexpected_model_load)
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "ไมโครโฟน (Live)" for c in demo.config["components"])
        assert any(c["props"].get("label") == "โมเดล ASR" for c in demo.config["components"])
    finally:
        demo.close()


def test_live_start_callback_uses_a_lightweight_ready_session_contract(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.asr.enabled = True

    from thai_offline_ai.streaming import StreamingASRService

    monkeypatch.setattr(StreamingASRService, "start_session", lambda self: "live-session-1")
    monkeypatch.setattr(
        StreamingASRService,
        "get_metrics",
        lambda self, session_id: SimpleNamespace(model_initialization_seconds=0.25),
    )
    demo, _ = build_app(settings)
    try:
        fn = next(key for key, value in demo.fns.items() if value.api_name == "start_live_asr")
        result = asyncio.run(demo.process_api(fn, inputs=[]))
        assert result["is_generating"] is False
        assert len(result["data"]) == 7
        # State is deliberately not serialized to the browser; the service owns
        # the decoder and receives only the lightweight session identifier.
        assert result["data"][0] is None
        assert result["data"][3].startswith("Ready / Listening")
        assert result["data"][4]["interactive"] is True

        dependency = next(item for item in demo.config["dependencies"] if item["api_name"] == "start_live_asr")
        assert dependency["types"]["generator"] is False
        assert len(dependency["outputs"]) == 7
        output_components = {item["id"]: item for item in demo.config["components"]}
        assert output_components[dependency["outputs"][0]]["type"] == "state"
    finally:
        demo.close()


def test_batch_asr_ui_constructs_without_loading_model(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.asr.enabled = True

    def unexpected_model_load(self):
        raise AssertionError("ASR must not load during UI construction")

    monkeypatch.setattr(TTSAdapter, "_get_model", unexpected_model_load)
    monkeypatch.setattr(TTSAdapter, "_get_kokoro_model", unexpected_model_load)
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "ผลลัพธ์แบบชุด" for c in demo.config["components"])
    finally:
        demo.close()


def test_llm_ui_is_lazy_and_keeps_history_session_local(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.llm.enabled = True

    def unexpected_connection(self, messages):
        raise AssertionError("Local LLM must not connect during UI construction")

    monkeypatch.setattr("thai_offline_ai.adapters.llm.OllamaLocalLLM.chat", unexpected_connection)
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "Local Chat" for c in demo.config["components"])
        fn = next(key for key, value in demo.fns.items() if value.api_name == "local_llm_chat")
        monkeypatch.setattr(
            "thai_offline_ai.adapters.llm.OllamaLocalLLM.chat",
            lambda self, messages: LLMResult("คำตอบ", "ollama", self.model),
        )
        result = asyncio.run(demo.process_api(fn, inputs=["คำถาม", "system", []]))
        assert result["data"][0][-1]["content"] == "คำตอบ"
        clear = next(key for key, value in demo.fns.items() if value.api_name == "clear_local_llm")
        cleared = asyncio.run(demo.process_api(clear, inputs=[]))
        assert cleared["data"] == [[], "", "Cleared local chat history."]
    finally:
        demo.close()


def test_diarization_ui_constructs_without_loading_model(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.diarization.enabled = True

    def unexpected_model_load(self):
        raise AssertionError("Diarization must not load during UI construction")

    monkeypatch.setattr(
        "thai_offline_ai.adapters.diarization.DiarizationAdapter._load", unexpected_model_load
    )
    monkeypatch.setattr(
        "thai_offline_ai.streaming_diarization.StreamingDiarizationService._load_engine",
        unexpected_model_load,
    )
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "Speaker Turns" for c in demo.config["components"])
        assert any(
            c["props"].get("label") == "ไมโครโฟน (Live + Speakers)"
            for c in demo.config["components"]
        )
    finally:
        demo.close()


def test_live_speaker_start_callback_uses_lightweight_session_contract(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.asr.enabled = True
    settings.diarization.enabled = True

    from thai_offline_ai.streaming_diarization import StreamingSpeakerASRService

    monkeypatch.setattr(
        StreamingSpeakerASRService, "start_session", lambda self: "live-spk-session-1"
    )
    monkeypatch.setattr(
        StreamingSpeakerASRService,
        "get_metrics",
        lambda self, session_id: SimpleNamespace(
            asr=SimpleNamespace(model_initialization_seconds=0.2),
            diarization=SimpleNamespace(model_initialization_seconds=0.1),
        ),
    )
    demo, _ = build_app(settings)
    try:
        fn = next(
            key for key, value in demo.fns.items() if value.api_name == "start_live_speaker_asr"
        )
        result = asyncio.run(demo.process_api(fn, inputs=[]))
        assert result["is_generating"] is False
        assert len(result["data"]) == 7
        assert result["data"][0] is None
        assert result["data"][3].startswith("Ready / Listening")
        assert result["data"][4]["interactive"] is True
    finally:
        demo.close()


def test_public_bind_rejected_before_gradio_construction(tmp_path):
    with pytest.raises(ConfigurationError, match="app.host"):
        build_app(Settings(app=AppConfig(host="0.0.0.0", output_dir=str(tmp_path))))


def test_custom_ui_theme_hides_gradio_footer_and_uses_no_external_urls():
    import gradio as gr

    settings = Settings()
    theme = build_theme(gr)
    assert not theme._stylesheets
    assert theme.body_text_color_dark == "#0F172A"
    assert theme.body_background_fill_dark == "#F8FAFC"
    assert theme.input_background_fill_dark == "#FFFFFF"
    assert "footer," in CUSTOM_CSS and "display: none !important;" in CUSTOM_CSS
    assert "http://" not in CUSTOM_CSS and "https://" not in CUSTOM_CSS
    assert "http://" not in HEAD_SCRIPT and "https://" not in HEAD_SCRIPT
    assert "data-target-tab" in HEAD_SCRIPT and 'remove("dark")' in HEAD_SCRIPT
    readiness = collect_ui_readiness(settings)
    navbar = render_navbar(settings, readiness)
    home = render_home_html(settings, readiness)
    footer = render_footer(settings)
    header = render_page_header(
        "Text-to-Speech Studio",
        "Thai & English Speech Synthesis",
        badges=[("Kokoro 24 kHz Ready", "success")],
    )
    summary_idle = render_tts_audio_summary()
    summary_ready = render_tts_audio_summary(
        filename="kokoro_tts_sample.wav",
        duration_seconds=3.45,
        sample_rate=24000,
        model="kokoro_thai_onnx",
        voice="f_young_bright",
        speed=1.05,
        elapsed_seconds=1.12,
    )
    system_html = render_system_cards(
        [
            Check("OK", "Operating system", "Windows-11"),
            Check("OK", "TTS model cache", "models/pythaitts; ready"),
            Check("OK", "UI bind", "127.0.0.1:7860; sharing disabled."),
        ],
        settings,
    )
    assert "Thai Offline" in navbar and "Offline 100%" in navbar
    assert "OpenThai 2.0" in home
    for target in ("NLP", "TTS", "ASR", "Local LLM", "JARVIS AI Assistant", "System"):
        assert f'data-target-tab="{target}"' in home
    assert "Thai Offline AI Workbench" in footer and "gradio.app" not in footer
    jarvis_status = render_jarvis_status_card(mode="online", status="completed", browser_active=True, browser_url="https://example.com")
    jarvis_tl = render_jarvis_timeline([TimelineStep(step_number=1, phase="Search", summary="ค้นหาแล้ว", tool_name="browser_search")])
    jarvis_src = render_jarvis_sources([SourceLink(title="Example", url="https://example.com")])
    jarvis_apr = render_jarvis_approval_card(
        ApprovalRequest(
            approval_id="apr-1",
            tool_name="browser_click",
            arguments={"selector_or_text": "Pay"},
            permission_level=PermissionLevel.SENSITIVE,
            reason="Payment action",
            action_summary="Click Pay",
        )
    )
    assert "Online Agent Mode" in jarvis_status and "https://example.com" in jarvis_status
    assert "browser_search" in jarvis_tl
    assert 'href="https://example.com"' in jarvis_src
    assert "Level 3" in jarvis_apr and "Click Pay" in jarvis_apr
    assert "Text-to-Speech Studio" in header and "Kokoro 24 kHz Ready" in header
    assert "ยังไม่มีเสียงที่สร้าง" in summary_idle
    assert "kokoro_tts_sample.wav" in summary_ready and "3.45s" in summary_ready
    assert "24.00 kHz" in summary_ready and "f_young_bright" in summary_ready
    assert "C:\\" not in summary_ready
    assert "Hardware &amp; Runtime" in system_html
    assert "Model Readiness &amp; Caches" in system_html
    assert "Local Endpoints &amp; Privacy" in system_html
    assert "127.0.0.1:7860" in system_html


def test_tts_studio_and_system_dashboard_callbacks(tmp_path, monkeypatch):
    import wave

    wav_path = tmp_path / "kokoro_demo.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 24000)

    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.tts.enabled = True
    monkeypatch.setattr(
        TTSAdapter,
        "synthesize",
        lambda self, text, *, model=None, voice=None, speed=None: TTSResult(
            audio_path=str(wav_path),
            sample_rate=24000,
            backend="kokoro_onnx",
            model=model or "kokoro_thai_onnx",
            voice=voice or "f_young_bright",
            speed=speed or 1.05,
        ),
    )
    demo, _ = build_app(settings)
    try:
        synth_fn = next(key for key, value in demo.fns.items() if value.api_name == "synthesize")
        synth_res = asyncio.run(
            demo.process_api(
                synth_fn,
                inputs=["สวัสดี AI", "kokoro_thai_onnx", "f_young_bright", 1.05],
            )
        )
        assert len(synth_res["data"]) == 4
        # Output 1: clean filename-only summary card (must not expose full directory path)
        assert "kokoro_demo.wav" in synth_res["data"][1]
        assert str(tmp_path) not in synth_res["data"][1]
        assert "1.00s" in synth_res["data"][1]
        # Output 3: Technical Metadata accordion holds full local path
        assert str(wav_path) in synth_res["data"][3]

        sys_fn = next(key for key, value in demo.fns.items() if value.api_name == "system_status")
        sys_res = asyncio.run(demo.process_api(sys_fn, inputs=[]))
        assert len(sys_res["data"]) == 2
        assert "Hardware &amp; Runtime" in sys_res["data"][0]
        assert "Operating system" in sys_res["data"][1]
    finally:
        demo.close()


def test_jarvis_ui_tab_is_lazy_and_handles_text_turn(tmp_path, monkeypatch):
    import json

    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.llm.enabled = True
    monkeypatch.setattr(
        "thai_offline_ai.adapters.llm.OllamaLocalLLM.chat",
        lambda self, messages: LLMResult(
            text=json.dumps(
                {
                    "action": "final_answer",
                    "plan_summary": "ตอบกลับแบบออฟไลน์",
                    "final_answer": "สวัสดีครับ JARVIS พร้อมช่วยงานแล้ว",
                }
            ),
            backend="ollama",
            model=self.model,
        ),
    )
    demo, _ = build_app(settings)
    try:
        assert any(c["props"].get("label") == "JARVIS Chat" for c in demo.config["components"])
        fn = next(key for key, value in demo.fns.items() if value.api_name == "jarvis_send")
        res = asyncio.run(
            demo.process_api(fn, inputs=["สวัสดี JARVIS", "offline", False, "ui-s1", []])
        )
        assert res["data"][0][-1]["content"] == "สวัสดีครับ JARVIS พร้อมช่วยงานแล้ว"
        assert "Offline Mode" in res["data"][5]
    finally:
        demo.close()


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
