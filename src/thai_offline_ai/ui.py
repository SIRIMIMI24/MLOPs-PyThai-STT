from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from thai_offline_ai.adapters.nlp import NLPAdapter
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.config import Settings, load_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_runtime

if TYPE_CHECKING:
    import gradio as gr


CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data: blob:; "
    "media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; "
    "frame-src 'none'; object-src 'none'; base-uri 'self'; form-action 'self'; "
    "frame-ancestors 'none'"
)


class LocalOnlyHeaders:
    """ASGI middleware restricting browser resources to the local application."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        async def send_headers(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"content-security-policy", CONTENT_SECURITY_POLICY.encode("ascii")),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"cache-control", b"no-store"),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_headers)


def build_app(settings: Settings | None = None) -> tuple[gr.Blocks, Settings]:
    """Build six local tabs, with NLP as the only Phase 1 inference action."""
    settings = settings or load_settings()
    prepare_runtime(settings)
    try:
        import gradio as gr
    except (ImportError, OSError) as exc:
        raise WorkbenchError(
            "Gradio cannot load. Activate .venv; run python -m pip install -e ."
        ) from exc
    from thai_offline_ai.diagnostics import collect_diagnostics, format_report

    adapter = NLPAdapter.from_config(settings.nlp, settings.app.offline_runtime)
    tts_adapter = TTSAdapter.from_settings(settings)

    def analyze(text: str) -> tuple[str, list[str], list[str], int]:
        try:
            result = adapter.analyze(text)
            return (
                result["normalized_text"],
                result["word_tokens"],
                result["sentence_tokens"],
                result["token_count"],
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def system_status() -> str:
        # Metadata/discovery only in UI; the doctor command also checks required imports.
        return format_report(collect_diagnostics(settings, check_imports=False))

    def synthesize(text: str) -> tuple[str, str]:
        try:
            result = tts_adapter.synthesize(text)
            return result.audio_path, (
                f"สร้างไฟล์แล้ว: {result.audio_path}\n"
                f"Backend: {result.backend}; Model: {result.model}; "
                f"Sample rate: {result.sample_rate} Hz"
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    theme = gr.themes.Default(
        font=["Tahoma", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"]
    )
    with gr.Blocks(
        title="Thai Offline AI Workbench",
        analytics_enabled=False,
        theme=theme,
        delete_cache=(3600, 3600),
    ) as demo:
        gr.Markdown("# Thai Offline AI Workbench\nเครื่องมือภาษาไทยบนคอมพิวเตอร์ของคุณ")
        with gr.Tab("Home"):
            gr.Markdown(
                "## Phase 1: ภาษาไทย (NLP)\n"
                "วิเคราะห์ข้อความด้วย CPU และข้อมูลที่ติดตั้งในเครื่อง "
                "ข้อความและผลลัพธ์ไม่ถูกบันทึกเป็นไฟล์โดยแอป\n\n"
                "TTS, ASR และ Local LLM จะเปิดใช้งานในระยะถัดไป"
            )
        with gr.Tab("NLP"):
            if not settings.nlp.enabled:
                gr.Markdown("NLP ถูกปิดอยู่ — ตั้งค่า nlp.enabled=true เพื่อเปิดใช้งาน")
            source = gr.Textbox(label="ข้อความภาษาไทย", lines=5, interactive=settings.nlp.enabled)
            button = gr.Button("วิเคราะห์ข้อความ", interactive=settings.nlp.enabled)
            normalized = gr.Textbox(label="ข้อความหลังปรับรูปแบบ (Normalized text)", interactive=False)
            words = gr.JSON(label="คำ (Word tokens)")
            sentences = gr.JSON(label="ประโยค (Sentence tokens)")
            count = gr.Number(label="จำนวนคำ (Token count)", interactive=False, precision=0)
            button.click(
                analyze,
                source,
                [normalized, words, sentences, count],
                api_name="analyze",
                concurrency_limit=1,
            )
        with gr.Tab("TTS"):
            gr.Markdown(
                "## สังเคราะห์เสียงภาษาไทย (Thai TTS)\n"
                "PyThaiTTS Lunarlist ONNX ใช้ CPU และโหลดโมเดลเมื่อกดสร้างเสียงเท่านั้น"
            )
            gr.Markdown(
                f"Backend: `{settings.tts.backend}` · Model: `{settings.tts.model}` · "
                f"Sample rate: {settings.tts.sample_rate} Hz"
            )
            tts_source = gr.Textbox(
                label="ข้อความภาษาไทย", lines=4, interactive=settings.tts.enabled
            )
            tts_button = gr.Button("สร้างเสียง", interactive=settings.tts.enabled)
            tts_audio = gr.Audio(label="เสียงที่สร้าง", type="filepath", interactive=False)
            tts_info = gr.Textbox(label="ข้อมูลไฟล์", lines=3, interactive=False)
            tts_button.click(synthesize, tts_source, [tts_audio, tts_info], api_name="synthesize")
            if not settings.tts.enabled:
                gr.Markdown("TTS ถูกปิดอยู่ — ตั้งค่า `tts.enabled=true` หลังติดตั้งโมเดลในเครื่อง")
        for title in ("ASR", "Local LLM"):
            with gr.Tab(title):
                gr.Markdown(f"## {title}\nยังไม่พร้อมใช้งาน — สงวนไว้สำหรับระยะถัดไป (Future phase)")
        with gr.Tab("System"):
            refresh = gr.Button("ตรวจสอบระบบ")
            status = gr.Textbox(label="สถานะระบบ", lines=20, interactive=False)
            refresh.click(system_status, outputs=status, api_name="system_status")
    return demo, settings


def launch_app(demo: gr.Blocks, settings: Settings, *, prevent_thread_lock: bool = False) -> Any:
    """Launch only on loopback; sharing, SSR, monitoring and MCP are explicitly off."""
    prepare_runtime(settings)
    from starlette.middleware import Middleware

    return demo.launch(
        server_name=settings.app.host,
        server_port=settings.app.port,
        share=False,
        inbrowser=False,
        inline=False,
        strict_cors=True,
        ssr_mode=False,
        mcp_server=False,
        pwa=False,
        enable_monitoring=False,
        root_path="",
        allowed_paths=[],
        show_api=False,
        app_kwargs={"middleware": [Middleware(LocalOnlyHeaders)]},
        prevent_thread_lock=prevent_thread_lock,
    )


def launch() -> int:
    demo = None
    try:
        demo, settings = build_app()
        launch_app(demo, settings)
        return 0
    except (WorkbenchError, OSError, ValueError) as exc:
        print(
            f"[FAIL] Cannot start the local UI: {exc}. Run python scripts/doctor.py.",
            file=sys.stderr,
        )
        return 1
    finally:
        if demo is not None:
            demo.close()
