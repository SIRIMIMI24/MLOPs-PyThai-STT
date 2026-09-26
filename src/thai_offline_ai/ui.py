from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.adapters.diarization import DiarizationAdapter
from thai_offline_ai.adapters.nlp import NLPAdapter
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.batch import BatchASRProcessor, export_batch_result
from thai_offline_ai.config import Settings, load_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import ChatMessage
from thai_offline_ai.runtime import prepare_runtime
from thai_offline_ai.speaker_composition import compose_speaker_transcript
from thai_offline_ai.streaming import StreamingASRService

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
    asr_adapter = ASRAdapter(settings)
    diarization_adapter = DiarizationAdapter(settings)
    streaming_service = StreamingASRService(settings)
    batch_processor = BatchASRProcessor(settings, adapter=asr_adapter)

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

    def transcribe(audio: str | None) -> tuple[str, list[dict[str, Any]], str]:
        try:
            if not audio:
                raise WorkbenchError("เลือกไฟล์ WAV ก่อนถอดเสียง")
            result = asr_adapter.transcribe_file(audio)
            return result.text, [segment.to_dict() for segment in result.segments], f"Duration: {result.duration:.2f}s · {result.backend} · {result.model}"
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def transcribe_batch(files: list[str] | None) -> tuple[list[list[object]], str, str | None, str | None]:
        try:
            if not files:
                raise WorkbenchError("เลือกไฟล์เสียงอย่างน้อยหนึ่งไฟล์ก่อนถอดเสียงแบบชุด")
            result = batch_processor.transcribe_paths(files)
            json_path, csv_path = export_batch_result(result, settings)
            rows = []
            for item in result.items:
                rows.append(
                    [
                        item.file_name,
                        item.status,
                        f"{item.duration:.2f}" if item.duration is not None else "",
                        f"{item.processing_time:.2f}" if item.processing_time is not None else "",
                        f"{item.rtf:.2f}" if item.rtf is not None else "",
                        item.asr_result.text[:120] if item.asr_result else "",
                        item.error or "",
                    ]
                )
            info = (
                f"Batch {result.batch_id}: {result.succeeded} สำเร็จ, {result.failed} ล้มเหลว · "
                f"Audio {result.total_audio_duration:.2f}s · Processing {result.total_processing_time:.2f}s"
            )
            return rows, info, str(json_path), str(csv_path)
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def diarize(audio: str | None) -> tuple[list[dict[str, Any]], str]:
        try:
            if not audio:
                raise WorkbenchError("เลือกไฟล์เสียงก่อนระบุผู้พูด")
            result = diarization_adapter.diarize_file(audio)
            return [turn.to_dict() for turn in result.turns], (
                f"Speakers: {', '.join(result.speakers) or '-'} · {result.backend} · "
                f"{result.model} · {result.device}"
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def transcribe_speakers(audio: str | None) -> tuple[list[dict[str, Any]], str]:
        try:
            if not audio:
                raise WorkbenchError("เลือกไฟล์เสียงก่อนถอดเสียงพร้อมผู้พูด")
            combined = compose_speaker_transcript(
                asr_adapter.transcribe_file(audio), diarization_adapter.diarize_file(audio)
            )
            return [segment.to_dict() for segment in combined.segments], combined.text
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def begin_live_preparation():
        """Render Preparing before the chained server-side setup action."""
        return (
            "",
            "",
            "",
            "Preparing · กำลังเตรียมโมเดล ASR ในเครื่อง",
            gr.update(value=None, interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def start_live_session():
        """Prepare local ASR before enabling browser microphone capture."""
        try:
            session_id = streaming_service.start_session()
            metrics = streaming_service.get_metrics(session_id)
            preparation = metrics.model_initialization_seconds
            preparation_text = f" · prepared in {preparation:.1f}s" if preparation else ""
            return (
                session_id,
                "",
                "",
                "Ready / Listening · เริ่มพูดได้ (16 kHz จะถูกแปลงอัตโนมัติ)" + preparation_text,
                gr.update(value=None, interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
        except WorkbenchError as exc:
            return (
                "",
                "",
                "",
                f"Preparation failed · {exc}",
                gr.update(value=None, interactive=False),
                gr.update(interactive=settings.asr.enabled),
                gr.update(interactive=False),
            )

    def process_live_chunk(
        audio: tuple[int, Any] | None, session_id: str | None
    ) -> tuple[str, str, str, str]:
        try:
            if not audio:
                return session_id or "", "", "", "ยังไม่ได้รับช่วงเสียง"
            if not session_id:
                return "", "", "", "Preparing · รอให้ session พร้อมก่อนเริ่มไมโครโฟน"
            active_session = session_id
            sample_rate, samples = audio
            streaming_service.push_audio(active_session, samples, int(sample_rate))
            events = streaming_service.get_events(active_session)
            metrics = streaming_service.get_metrics(active_session)
            final_text = "".join(event.text for event in events)
            latency = metrics.latest_chunk_latency_seconds
            latency_text = f"{latency * 1000:.0f} ms" if latency is not None else "n/a"
            first_audio_latency = metrics.first_audio_to_decoder_seconds
            first_audio_text = (
                f" · first audio {first_audio_latency * 1000:.0f} ms"
                if first_audio_latency is not None
                else ""
            )
            return (
                active_session,
                "",
                final_text,
                (
                    "Processing complete · "
                    f"{metrics.finalized_events} finalized · latest {latency_text} · "
                    f"session {metrics.session_duration_seconds:.1f}s{first_audio_text}"
                ),
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def stop_live_session(session_id: str | None):
        try:
            if not session_id:
                return (
                    "",
                    "",
                    "",
                    "ยังไม่มี Live session",
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=settings.asr.enabled),
                    gr.update(interactive=False),
                )
            streaming_service.finalize(session_id)
            events = streaming_service.get_events(session_id)
            metrics = streaming_service.get_metrics(session_id)
            final_text = "".join(event.text for event in events)
            streaming_service.close_session(session_id)
            rtf_text = f"{metrics.session_rtf:.2f}" if metrics.session_rtf is not None else "n/a"
            return (
                "",
                "",
                final_text,
                f"Stopped · {metrics.finalized_events} finalized · session RTF {rtf_text}",
                gr.update(value=None, interactive=False),
                gr.update(interactive=settings.asr.enabled),
                gr.update(interactive=False),
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def send_local_chat(
        message: str, system_prompt: str, history: list[dict[str, str]] | None
    ) -> tuple[list[dict[str, str]], str, str]:
        try:
            if not message.strip():
                raise WorkbenchError("พิมพ์ข้อความก่อนส่งไปยัง Local LLM")
            from thai_offline_ai.adapters.llm import OllamaLocalLLM

            messages = [ChatMessage("system", system_prompt or settings.llm.system_prompt)]
            messages.extend(ChatMessage(item["role"], item["content"]) for item in history or [])
            messages.append(ChatMessage("user", message.strip()))
            result = OllamaLocalLLM.from_config(settings.llm, settings.app.offline_runtime).chat(messages)
            updated = [
                *(history or []),
                {"role": "user", "content": message.strip()},
                {"role": "assistant", "content": result.text},
            ]
            detail = (
                f"{result.backend} · {result.model} · {result.duration:.2f}s"
                if result.duration is not None
                else f"{result.backend} · {result.model}"
            )
            return updated, "", detail
        except (WorkbenchError, ValueError) as exc:
            raise gr.Error(str(exc)) from None

    def clear_local_chat() -> tuple[list[dict[str, str]], str, str]:
        return [], "", "Cleared local chat history."

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
                "## Thai Offline AI Workbench\n"
                "วิเคราะห์ข้อความด้วย CPU และข้อมูลที่ติดตั้งในเครื่อง "
                "ข้อความและผลลัพธ์ไม่ถูกบันทึกเป็นไฟล์โดยแอป\n\n"
                "NLP, TTS, ASR และ Local LLM ใช้เฉพาะทรัพยากรในเครื่อง"
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
        with gr.Tab("ASR"):
            gr.Markdown("## ถอดเสียงภาษาไทย (Thai ASR)\nอัปโหลดไฟล์ WAV เพื่อถอดเสียงในเครื่อง")
            asr_audio = gr.Audio(label="ไฟล์เสียง WAV", type="filepath", sources=["upload"], interactive=settings.asr.enabled)
            asr_button = gr.Button("ถอดเสียง", interactive=settings.asr.enabled)
            asr_text = gr.Textbox(label="ข้อความถอดเสียง", lines=5, interactive=False)
            asr_segments = gr.JSON(label="ส่วนของเสียง (Segments)")
            asr_info = gr.Textbox(label="ข้อมูล", interactive=False)
            asr_button.click(transcribe, asr_audio, [asr_text, asr_segments, asr_info], api_name="transcribe")
            gr.Markdown("### Batch Files (Phase 3.6)\nถอดเสียงหลายไฟล์แบบลำดับเพื่อใช้ CPU และโมเดลร่วมกันอย่างปลอดภัย")
            batch_files = gr.File(
                label="ไฟล์เสียงหลายไฟล์ (WAV, FLAC, MP3, OGG)",
                file_count="multiple",
                type="filepath",
                file_types=["audio"],
                interactive=settings.asr.enabled,
            )
            batch_button = gr.Button("ถอดเสียงแบบชุด", interactive=settings.asr.enabled)
            batch_table = gr.Dataframe(
                headers=["File", "Status", "Duration (s)", "Processing (s)", "RTF", "Transcript", "Error"],
                datatype=["str"] * 7,
                interactive=False,
                label="ผลลัพธ์แบบชุด",
            )
            batch_info = gr.Textbox(label="สรุป Batch", interactive=False)
            batch_json = gr.File(label="ส่งออก JSON", interactive=False)
            batch_csv = gr.File(label="ส่งออก CSV", interactive=False)
            batch_button.click(
                transcribe_batch,
                batch_files,
                [batch_table, batch_info, batch_json, batch_csv],
                api_name="transcribe_batch",
                concurrency_limit=1,
            )
            gr.Markdown(
                "### Speaker Diarization (Phase 5)\n"
                "ระบุผู้พูดแบบไม่ระบุตัวตนเป็น SPEAKER_00, SPEAKER_01 โดยไม่อนุมาน Agent/Customer"
            )
            diarization_audio = gr.Audio(
                label="ไฟล์เสียงสำหรับ Speaker Diarization",
                type="filepath",
                sources=["upload"],
                interactive=settings.diarization.enabled,
            )
            with gr.Row():
                diarize_button = gr.Button("ระบุผู้พูด", interactive=settings.diarization.enabled)
                speakers_button = gr.Button(
                    "ถอดเสียงพร้อมผู้พูด", interactive=settings.diarization.enabled
                )
            diarization_turns = gr.JSON(label="Speaker Turns")
            diarization_info = gr.Textbox(label="ข้อมูล Speaker Diarization", interactive=False)
            speaker_transcript = gr.JSON(label="ข้อความถอดเสียงพร้อมผู้พูด")
            diarize_button.click(
                diarize,
                diarization_audio,
                [diarization_turns, diarization_info],
                api_name="diarize",
                concurrency_limit=1,
            )
            speakers_button.click(
                transcribe_speakers,
                diarization_audio,
                [speaker_transcript, diarization_info],
                api_name="transcribe_speakers",
                concurrency_limit=1,
            )
            if not settings.diarization.enabled:
                gr.Markdown(
                    "Speaker Diarization ถูกปิดอยู่ — ตั้งค่า `diarization.enabled=true` "
                    "หลังติดตั้งโมเดลในเครื่อง"
                )
            gr.Markdown(
                "### Near Real-Time / Live (Phase 3.5)\n"
                "เริ่ม session แล้วพูดผ่านไมโครโฟน ผลที่ถอดได้เป็นส่วนที่ยืนยันแล้ว; "
                "PyThaiASR รุ่นนี้ไม่ส่ง hypothesis ที่แก้ไขย้อนหลังได้"
            )
            live_session = gr.State("")
            live_start = gr.Button("เริ่ม Live transcription", interactive=settings.asr.enabled)
            live_audio = gr.Audio(
                label="ไมโครโฟน (Live)", type="numpy", sources=["microphone"],
                streaming=True, interactive=False,
            )
            live_stop = gr.Button("หยุดและสรุปผล", interactive=False)
            live_partial = gr.Textbox(label="Partial hypothesis", interactive=False)
            live_final = gr.Textbox(label="ข้อความที่ยืนยันแล้ว", lines=5, interactive=False)
            live_info = gr.Textbox(label="สถานะ Live", interactive=False)
            start_outputs = [
                live_session,
                live_partial,
                live_final,
                live_info,
                live_audio,
                live_start,
                live_stop,
            ]
            live_start.click(
                begin_live_preparation,
                outputs=start_outputs,
                api_name="begin_live_preparation",
            ).then(
                start_live_session,
                outputs=[
                    live_session,
                    live_partial,
                    live_final,
                    live_info,
                    live_audio,
                    live_start,
                    live_stop,
                ],
                api_name="start_live_asr",
            )
            live_audio.stream(
                process_live_chunk, [live_audio, live_session],
                [live_session, live_partial, live_final, live_info], api_name="live_asr_chunk",
                concurrency_limit=1,
            )
            live_stop.click(
                stop_live_session,
                live_session,
                [
                    live_session,
                    live_partial,
                    live_final,
                    live_info,
                    live_audio,
                    live_start,
                    live_stop,
                ],
                api_name="stop_live_asr",
            )
        with gr.Tab("Local LLM"):
            gr.Markdown(
                "## Local LLM\n"
                "Ollama บนเครื่องเท่านั้น: ไม่มี cloud fallback และจะเชื่อมต่อเมื่อกดส่งข้อความ"
            )
            gr.Markdown(
                f"Backend: `{settings.llm.backend}` · Model: `{settings.llm.model}` · "
                f"Endpoint: `{settings.llm.base_url}`"
            )
            llm_history = gr.State([])
            llm_system = gr.Textbox(
                label="System Prompt",
                value=settings.llm.system_prompt,
                lines=2,
                interactive=settings.llm.enabled,
            )
            llm_chat = gr.Chatbot(label="Local Chat", type="messages")
            llm_message = gr.Textbox(label="ข้อความ", lines=3, interactive=settings.llm.enabled)
            with gr.Row():
                llm_send = gr.Button("ส่งข้อความ", interactive=settings.llm.enabled)
                llm_clear = gr.Button("ล้างประวัติ", interactive=settings.llm.enabled)
            llm_status = gr.Textbox(label="สถานะ Local LLM", interactive=False)
            llm_send.click(
                send_local_chat,
                [llm_message, llm_system, llm_history],
                [llm_chat, llm_message, llm_status],
                api_name="local_llm_chat",
                concurrency_limit=1,
            ).then(lambda history: history, llm_chat, llm_history)
            llm_clear.click(
                clear_local_chat,
                outputs=[llm_chat, llm_message, llm_status],
                api_name="clear_local_llm",
            ).then(list, outputs=llm_history)
            if not settings.llm.enabled:
                gr.Markdown(
                    "Local LLM ถูกปิดอยู่ — ตั้งค่า `llm.enabled=true` หลังติดตั้ง Ollama "
                    "และโมเดลในเครื่อง"
                )
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
