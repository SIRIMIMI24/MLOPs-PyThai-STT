from __future__ import annotations

import sys
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.adapters.diarization import DiarizationAdapter
from thai_offline_ai.adapters.kokoro_tts import DEFAULT_KOKORO_VOICE, kokoro_voice_choices
from thai_offline_ai.adapters.nlp import NLPAdapter
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.agent.config import load_agent_settings
from thai_offline_ai.agent.orchestrator import AgentOrchestrator
from thai_offline_ai.agent.schemas import OperatingMode
from thai_offline_ai.agent.voice.voice_pipeline import JarvisVoicePipeline
from thai_offline_ai.batch import BatchASRProcessor, export_batch_result
from thai_offline_ai.benchmark import ASRBenchmark, export_benchmark_result
from thai_offline_ai.config import Settings, load_settings
from thai_offline_ai.conversation_intel import (
    ConversationIntelligenceService,
    parse_speaker_text,
    utterances_from_transcript,
)
from thai_offline_ai.conversation_intel import export_report as export_conversation_report
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import ChatMessage
from thai_offline_ai.local_rag import LocalRAGStore
from thai_offline_ai.model_manager import ModelManager
from thai_offline_ai.packaging import build_portable_bundle
from thai_offline_ai.runtime import prepare_runtime
from thai_offline_ai.speaker_composition import compose_speaker_transcript
from thai_offline_ai.streaming import StreamingASRService
from thai_offline_ai.streaming_diarization import StreamingSpeakerASRService
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
    render_voice_latency_card,
)
from thai_offline_ai.voice_assistant import LocalVoiceAssistantService

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
    streaming_speaker_service = StreamingSpeakerASRService(
        settings, asr_service=streaming_service
    )
    batch_processor = BatchASRProcessor(settings, adapter=asr_adapter)
    asr_benchmark = ASRBenchmark(settings, adapter=asr_adapter)
    agent_settings = load_agent_settings()
    agent_settings.memory.db_path = str(Path(settings.app.output_dir) / "jarvis_memory.db")
    agent_settings.browser.profile_dir = str(
        Path(settings.app.output_dir) / "jarvis_browser_profile"
    )
    local_rag_store = LocalRAGStore(
        Path(settings.app.output_dir) / "local_rag.db", settings=settings
    )
    model_manager = ModelManager(settings)
    jarvis_orchestrator = AgentOrchestrator(
        settings, agent_settings, rag_store=local_rag_store
    )
    jarvis_voice = JarvisVoicePipeline(
        jarvis_orchestrator,
        asr_adapter=asr_adapter,
        tts_adapter=tts_adapter,
        max_tts_chars=agent_settings.voice.max_tts_chars,
    )
    voice_assistant = LocalVoiceAssistantService(
        settings,
        asr_adapter=asr_adapter,
        tts_adapter=tts_adapter,
    )
    from thai_offline_ai.speech_lab.experiments import ExperimentRegistry
    from thai_offline_ai.speech_lab.synthetic import (
        AUGMENTATION_PRESETS,
        SyntheticCorpusItem,
        SyntheticSpeechGenerator,
    )

    speech_lab_dir = Path(settings.app.output_dir) / "speech_lab"
    speech_lab_registry = ExperimentRegistry(speech_lab_dir / "experiments.db")
    synthetic_generator = SyntheticSpeechGenerator(settings, tts_adapter=tts_adapter)
    active_asr = {
        "model": settings.asr.model or "typhoon_asr",
        "prompt": settings.asr.openthai_prompt,
    }

    def update_asr_selection(model_choice: str, prompt_choice: str) -> str:
        active_asr["model"] = model_choice
        active_asr["prompt"] = prompt_choice
        if model_choice == "openthai2_asr":
            return (
                f"กำลังเลือก: OpenThai 2.0 ASR Streaming ONNX "
                f"({settings.asr.openthai_precision}, prompt={prompt_choice})"
            )
        return "กำลังเลือก: Typhoon ASR (PyThaiASR ONNX)"

    def benchmark_asr_ui(
        files: list[str] | None, model_choice: str, prompt_choice: str
    ) -> tuple[list[list[object]], str, str | None, str | None]:
        """Measure one local ASR adapter path without retaining transcript text."""
        try:
            result = asr_benchmark.run(files or [], model=model_choice, prompt_mode=prompt_choice)
            json_path, csv_path = export_benchmark_result(result, settings)
            rows = [
                [
                    item.file_name,
                    item.status,
                    f"{item.audio_seconds:.3f}" if item.audio_seconds is not None else "-",
                    f"{item.elapsed_seconds:.3f}" if item.elapsed_seconds is not None else "-",
                    f"{item.rtf:.3f}" if item.rtf is not None else "-",
                    item.error or "",
                ]
                for item in result.items
            ]
            aggregate = f"{result.aggregate_rtf:.3f}" if result.aggregate_rtf is not None else "n/a"
            summary = (
                f"Measured locally · {result.succeeded}/{result.total_files} succeeded · "
                f"audio {result.total_audio_seconds:.2f}s · elapsed {result.total_elapsed_seconds:.2f}s · "
                f"aggregate RTF {aggregate}. RTF < 1 means processing faster than source duration."
            )
            return rows, summary, str(json_path), str(csv_path)
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def analyze(text: str) -> tuple[str, list[str], list[str], int, str]:
        try:
            result = adapter.analyze(text)
            summary = (
                f"ภาษาไทย: {result['thai_char_count']} ตัวอักษร · "
                f"ภาษาอังกฤษ (Latin): {result['latin_char_count']} ตัวอักษร · "
                f"รวม {result['token_count']} คำ"
            )
            return (
                result["normalized_text"],
                result["word_tokens"],
                result["sentence_tokens"],
                result["token_count"],
                summary,
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def system_status() -> tuple[str, str]:
        # Metadata/discovery only in UI; the doctor command also checks required imports.
        checks = collect_diagnostics(settings, check_imports=False)
        return render_system_cards(checks, settings), format_report(checks)

    def list_managed_models_ui() -> list[list[str]]:
        return model_manager.to_table_rows()

    def install_managed_model_ui(
        model_key: str, allow_setup: bool
    ) -> tuple[list[list[str]], str]:
        try:
            res = model_manager.run_setup(model_key, allow_setup_network=bool(allow_setup))
            msg = (
                f"ติดตั้ง '{res['key']}' สำเร็จ (ready={res['ready']}, size={res['size_mb']:.1f} MB)\n"
                f"{res['output']}"
            )
            return model_manager.to_table_rows(), msg
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def create_portable_bundle_ui() -> str:
        res = build_portable_bundle(settings)
        return (
            f"สร้างชุด Portable Desktop Bundle สำเร็จ: {res.bundle_dir}\n"
            f"Launcher: {res.launcher_bat}\n"
            f"Manifest: {res.manifest_path}"
        )

    def rag_list_docs_ui() -> list[list[str]]:
        return [
            [d.doc_id, d.title, str(d.chunk_count), str(d.char_count), d.created_at]
            for d in local_rag_store.list_documents()
        ]

    def rag_ingest_ui(
        files: list[str] | None, inline_title: str, inline_text: str
    ) -> tuple[list[list[str]], str]:
        try:
            ingested = []
            if files:
                ingested.extend(local_rag_store.ingest_files(files))
            if (inline_text or "").strip():
                ingested.append(
                    local_rag_store.ingest_text(
                        inline_title or "ข้อความที่วาง", inline_text
                    )
                )
            if not ingested:
                raise WorkbenchError("เลือกไฟล์เอกสาร หรือพิมพ์ข้อความก่อนกดนำเข้า Local RAG")
            total_chunks = sum(item.chunk_count for item in ingested)
            return (
                rag_list_docs_ui(),
                f"นำเข้าสำเร็จ {len(ingested)} เอกสาร (รวม {total_chunks} chunks)",
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def rag_ask_ui(question: str, use_llm: bool) -> tuple[str, list[dict[str, Any]]]:
        try:
            ans = local_rag_store.ask(
                question, top_k=4, use_llm=bool(use_llm and settings.llm.enabled)
            )
            return ans.answer, [m.to_dict() for m in ans.matches]
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def rag_clear_ui() -> tuple[list[list[str]], str]:
        local_rag_store.clear_all()
        return [], "ล้างฐานความรู้ Local RAG ทั้งหมดแล้ว"

    def conversation_intel_ui(
        audio: str | None, pasted: str, terms_text: str, use_llm: bool
    ) -> tuple[str, dict[str, Any], str]:
        try:
            if (pasted or "").strip():
                utterances = parse_speaker_text(pasted)
            elif audio:
                utterances = utterances_from_transcript(
                    compose_speaker_transcript(
                        asr_adapter.transcribe_file(
                            audio, model=active_asr["model"], prompt_mode=active_asr["prompt"]
                        ),
                        diarization_adapter.diarize_file(audio),
                    )
                )
            else:
                raise WorkbenchError("วางบทสนทนา (SPEAKER_00: ...) หรือเลือกไฟล์เสียงก่อนวิเคราะห์")
            terms = [t.strip() for t in (terms_text or "").replace("\n", ",").split(",") if t.strip()]
            report = ConversationIntelligenceService(settings).analyze(
                utterances,
                domain_terms=terms,
                use_llm=bool(use_llm and settings.llm.enabled),
            )
            json_path, _ = export_conversation_report(report, settings)
            info = (
                f"วิเคราะห์ {report.utterance_count} ช่วงคำพูด · ผู้พูด {len(report.speakers)} · "
                f"บันทึก: {json_path.name}"
            )
            return report.to_markdown(), report.to_dict(), info
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def count_chars(text: str) -> str:
        raw = text or ""
        thai = sum(1 for ch in raw if "\u0e00" <= ch <= "\u0e7f")
        latin = sum(1 for ch in raw if ch.isascii() and ch.isalpha())
        return (
            f'<div class="aiw-char-counter">'
            f"ทั้งหมด <strong>{len(raw)}</strong> ตัวอักษร · ไทย {thai} · อังกฤษ {latin}"
            "</div>"
        )

    def clear_tts_input() -> tuple[str, str]:
        return "", count_chars("")

    def clear_nlp_input() -> tuple[str, str, str, list[str], list[str], int, str]:
        return "", count_chars(""), "", [], [], 0, ""

    def synthesize(
        text: str,
        model_choice: str | None = None,
        voice_choice: str | None = None,
        speed_choice: float | None = None,
    ) -> tuple[str, str, Any, str]:
        try:
            started = time.perf_counter()
            if model_choice is None and voice_choice is None and speed_choice is None:
                result = tts_adapter.synthesize(text)
            else:
                result = tts_adapter.synthesize(
                    text, model=model_choice, voice=voice_choice, speed=speed_choice
                )
            elapsed = time.perf_counter() - started
            duration_s: float | None = None
            try:
                with wave.open(result.audio_path, "rb") as wf:
                    if wf.getframerate() > 0:
                        duration_s = wf.getnframes() / wf.getframerate()
            except OSError:
                duration_s = None
            filename = Path(result.audio_path).name
            summary_html = render_tts_audio_summary(
                filename=filename,
                duration_seconds=duration_s,
                sample_rate=result.sample_rate,
                model=result.model,
                voice=result.voice,
                speed=result.speed,
                elapsed_seconds=elapsed,
            )
            voice_detail = (
                f"\nVoice: {result.voice} · Speed: {result.speed:.2f}x"
                if result.voice and result.speed is not None
                else ""
            )
            tech_details = (
                f"Filename: {filename}\n"
                f"Full path: {result.audio_path}\n"
                f"Backend: {result.backend} · Model: {result.model} · "
                f"Sample rate: {result.sample_rate} Hz · Elapsed: {elapsed:.2f}s{voice_detail}"
            )
            return (
                result.audio_path,
                summary_html,
                gr.update(value=result.audio_path, interactive=True),
                tech_details,
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def transcribe(audio: str | None) -> tuple[str, list[dict[str, Any]], str]:
        try:
            if not audio:
                raise WorkbenchError("เลือกไฟล์เสียงก่อนถอดเสียง")
            result = asr_adapter.transcribe_file(
                audio, model=active_asr["model"], prompt_mode=active_asr["prompt"]
            )
            prompt_note = (
                f" · {result.metadata['openthai_prompt']}"
                if "openthai_prompt" in result.metadata
                else ""
            )
            return (
                result.text,
                [segment.to_dict() for segment in result.segments],
                f"Duration: {result.duration:.2f}s · {result.backend} · {result.model}{prompt_note}",
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def transcribe_batch(files: list[str] | None) -> tuple[list[list[object]], str, str | None, str | None]:
        try:
            if not files:
                raise WorkbenchError("เลือกไฟล์เสียงอย่างน้อยหนึ่งไฟล์ก่อนถอดเสียงแบบชุด")
            result = batch_processor.transcribe_paths(
                files, model=active_asr["model"], prompt_mode=active_asr["prompt"]
            )
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
                asr_adapter.transcribe_file(
                    audio, model=active_asr["model"], prompt_mode=active_asr["prompt"]
                ),
                diarization_adapter.diarize_file(audio),
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
            if (
                active_asr["model"] == (settings.asr.model or "typhoon_asr")
                and active_asr["prompt"] == settings.asr.openthai_prompt
            ):
                session_id = streaming_service.start_session()
            else:
                session_id = streaming_service.start_session(
                    model=active_asr["model"], prompt_mode=active_asr["prompt"]
                )
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

    live_speakers_enabled = settings.asr.enabled and settings.diarization.enabled

    def begin_live_speaker_preparation():
        return (
            "",
            "",
            [],
            "Preparing · กำลังเตรียมโมเดล ASR และ Speaker Diarization ในเครื่อง",
            gr.update(value=None, interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def start_live_speaker_session():
        try:
            if (
                active_asr["model"] == (settings.asr.model or "typhoon_asr")
                and active_asr["prompt"] == settings.asr.openthai_prompt
            ):
                session_id = streaming_speaker_service.start_session()
            else:
                session_id = streaming_speaker_service.start_session(
                    model=active_asr["model"], prompt_mode=active_asr["prompt"]
                )
            metrics = streaming_speaker_service.get_metrics(session_id)
            asr_prep = metrics.asr.model_initialization_seconds or 0.0
            diar_prep = metrics.diarization.model_initialization_seconds or 0.0
            prep_total = asr_prep + diar_prep
            prep_text = f" · prepared in {prep_total:.1f}s" if prep_total else ""
            return (
                session_id,
                "",
                [],
                (
                    "Ready / Listening · เริ่มพูดได้ "
                    "(สตรีม 16 kHz เดียวส่งให้ทั้ง ASR และ Diarization)" + prep_text
                ),
                gr.update(value=None, interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
        except WorkbenchError as exc:
            return (
                "",
                "",
                [],
                f"Preparation failed · {exc}",
                gr.update(value=None, interactive=False),
                gr.update(interactive=live_speakers_enabled),
                gr.update(interactive=False),
            )

    def process_live_speaker_chunk(
        audio: tuple[int, Any] | None, session_id: str | None
    ) -> tuple[str, str, list[dict[str, Any]], str]:
        try:
            if not audio:
                return session_id or "", "", [], "ยังไม่ได้รับช่วงเสียง"
            if not session_id:
                return "", "", [], "Preparing · รอให้ session พร้อมก่อนเริ่มไมโครโฟน"
            sample_rate, samples = audio
            streaming_speaker_service.push_audio(session_id, samples, int(sample_rate))
            transcript = streaming_speaker_service.format_transcript(session_id)
            turns = [
                turn.to_dict()
                for turn in streaming_speaker_service.get_turns(session_id, include_open=True)
            ]
            metrics = streaming_speaker_service.get_metrics(session_id)
            speakers_text = ", ".join(metrics.speakers) or "-"
            asr_lat = metrics.asr.latest_chunk_latency_seconds
            asr_lat_text = f"{asr_lat * 1000:.0f} ms" if asr_lat is not None else "n/a"
            return (
                session_id,
                transcript,
                turns,
                (
                    f"Listening · Speakers: {speakers_text} · "
                    f"attributed {metrics.attributed_events}/{metrics.attributed_events + metrics.unattributed_events} · "
                    f"ASR latest {asr_lat_text}"
                ),
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def stop_live_speaker_session(session_id: str | None):
        try:
            if not session_id:
                return (
                    "",
                    "",
                    [],
                    "ยังไม่มี Live Speaker session",
                    gr.update(value=None, interactive=False),
                    gr.update(interactive=live_speakers_enabled),
                    gr.update(interactive=False),
                )
            streaming_speaker_service.finalize(session_id)
            transcript = streaming_speaker_service.format_transcript(session_id)
            turns = [
                turn.to_dict()
                for turn in streaming_speaker_service.get_turns(session_id, include_open=False)
            ]
            metrics = streaming_speaker_service.get_metrics(session_id)
            streaming_speaker_service.close_session(session_id)
            asr_rtf = f"{metrics.asr.session_rtf:.2f}" if metrics.asr.session_rtf is not None else "n/a"
            diar_rtf = (
                f"{metrics.diarization.session_rtf:.2f}"
                if metrics.diarization.session_rtf is not None
                else "n/a"
            )
            speakers_text = ", ".join(metrics.speakers) or "-"
            return (
                "",
                transcript,
                turns,
                (
                    f"Stopped · Speakers: {speakers_text} · "
                    f"attributed {metrics.attributed_events} · "
                    f"ASR RTF {asr_rtf} · Diarization RTF {diar_rtf}"
                ),
                gr.update(value=None, interactive=False),
                gr.update(interactive=live_speakers_enabled),
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
        voice_assistant.clear_session("local-llm-ui")
        return [], "", "Cleared local chat history."

    def send_local_voice_chat(
        audio_file: str | None,
        system_prompt: str,
        speak_reply: bool,
        history: list[dict[str, str]] | None,
    ):
        try:
            turn = voice_assistant.process_audio_turn(
                "local-llm-ui",
                audio_file,
                system_prompt=system_prompt,
                external_history=history or [],
                speak_response=bool(speak_reply),
                asr_model=active_asr["model"],
                asr_prompt=active_asr["prompt"],
            )
            m = turn.metrics
            status_line = (
                f"Offline Voice · ASR {m.asr_seconds:.2f}s · LLM {m.llm_seconds:.2f}s · "
                f"TTS {m.tts_seconds:.2f}s · Total {m.total_seconds:.2f}s"
            )
            return (
                turn.history,
                turn.history,
                turn.transcript,
                turn.audio_path,
                render_voice_latency_card(m, tts_error=turn.tts_error),
                status_line,
            )
        except (WorkbenchError, ValueError) as exc:
            raise gr.Error(str(exc)) from None

    readiness = collect_ui_readiness(settings)
    playwright_ready = bool(readiness.get("playwright_ready"))

    def _render_jarvis_outputs(
        response: Any,
        history: list[dict[str, str]],
        *,
        user_msg: str | None = None,
        transcript: str = "",
        audio_path: str | None = None,
        extra_note: str = "",
    ):
        updated = list(history or [])
        if user_msg:
            updated.append({"role": "user", "content": user_msg})
        updated.append({"role": "assistant", "content": response.answer})
        pending = response.pending_approval
        approval_id = pending.approval_id if pending else ""
        detail = extra_note or response.plan_summary
        status_html = render_jarvis_status_card(
            mode=response.mode.value,
            status=response.status,
            browser_active=response.browser_active,
            browser_url=response.browser_url,
            playwright_ready=playwright_ready,
            detail=detail,
        )
        timeline_html = render_jarvis_timeline(response.timeline)
        sources_html = render_jarvis_sources(response.sources)
        approval_html = render_jarvis_approval_card(pending)
        tools_payload = [
            {
                "tool": t.tool_name,
                "ok": t.ok,
                "summary": t.summary,
                "blocked_reason": t.blocked_reason,
                "prompt_injection_warning": t.prompt_injection_warning,
                "duration_s": t.duration_seconds,
            }
            for t in response.tool_history
        ]
        return (
            updated,
            updated,
            "",
            transcript,
            audio_path,
            status_html,
            timeline_html,
            sources_html,
            approval_html,
            approval_id,
            gr.update(interactive=bool(pending)),
            gr.update(interactive=bool(pending)),
            tools_payload,
        )

    def jarvis_send_text(
        message: str,
        mode_value: str,
        auto_speak: bool,
        session_id: str,
        history: list[dict[str, str]] | None,
    ):
        try:
            text = (message or "").strip()
            if not text:
                raise WorkbenchError("พิมพ์คำสั่งให้ JARVIS ก่อนกดส่งครับ")
            mode = OperatingMode(mode_value)
            resp = jarvis_orchestrator.run_turn(session_id, text, mode=mode)
            audio_path: str | None = None
            tts_note = ""
            if auto_speak and settings.tts.enabled and resp.answer and resp.status == "completed":
                audio_path, tts_err = jarvis_voice.speak_answer(resp.answer)
                if tts_err:
                    tts_note = f"TTS notice: {tts_err}"
            return _render_jarvis_outputs(
                resp,
                history or [],
                user_msg=text,
                audio_path=audio_path,
                extra_note=tts_note,
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def jarvis_send_voice(
        audio_file: str | None,
        mode_value: str,
        auto_speak: bool,
        session_id: str,
        history: list[dict[str, str]] | None,
    ):
        try:
            mode = OperatingMode(mode_value)
            turn = jarvis_voice.process_voice_command(
                session_id,
                audio_file,
                mode=mode,
                speak_response=bool(auto_speak and settings.tts.enabled),
                asr_model=active_asr["model"],
                asr_prompt=active_asr["prompt"],
            )
            tts_note = f"TTS notice: {turn.tts_error}" if turn.tts_error else ""
            return _render_jarvis_outputs(
                turn.agent_response,
                history or [],
                user_msg=f"[Voice] {turn.transcript}",
                transcript=turn.transcript,
                audio_path=turn.audio_path,
                extra_note=tts_note,
            )
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def jarvis_handle_approval(
        approved: bool,
        approval_id: str,
        auto_speak: bool,
        session_id: str,
        history: list[dict[str, str]] | None,
    ):
        if not approval_id:
            raise gr.Error("ไม่มีคำขออนุมัติที่รออยู่")
        resp = jarvis_orchestrator.resume_after_approval(
            session_id, approval_id, approved=approved
        )
        audio_path: str | None = None
        tts_note = ""
        if auto_speak and settings.tts.enabled and resp.answer and resp.status == "completed":
            audio_path, tts_err = jarvis_voice.speak_answer(resp.answer)
            if tts_err:
                tts_note = f"TTS notice: {tts_err}"
        return _render_jarvis_outputs(
            resp,
            history or [],
            audio_path=audio_path,
            extra_note=tts_note,
        )

    def jarvis_approve(
        approval_id: str, auto_speak: bool, session_id: str, history: list[dict[str, str]] | None
    ):
        return jarvis_handle_approval(True, approval_id, auto_speak, session_id, history)

    def jarvis_deny(
        approval_id: str, auto_speak: bool, session_id: str, history: list[dict[str, str]] | None
    ):
        return jarvis_handle_approval(False, approval_id, auto_speak, session_id, history)

    def jarvis_stop(session_id: str, mode_value: str) -> str:
        jarvis_orchestrator.request_cancel(session_id)
        return render_jarvis_status_card(
            mode=mode_value,
            status="cancelled",
            browser_active=jarvis_orchestrator.browser.is_running,
            browser_url=jarvis_orchestrator.browser.current_url,
            playwright_ready=playwright_ready,
            detail="ส่งสัญญาณหยุดการทำงาน (Cancel/Emergency Stop) แล้ว",
        )

    def jarvis_close_browser(mode_value: str) -> str:
        msg = jarvis_orchestrator.close_browser()
        return render_jarvis_status_card(
            mode=mode_value,
            status="idle",
            browser_active=False,
            browser_url=None,
            playwright_ready=playwright_ready,
            detail=msg,
        )

    def jarvis_reset_session(session_id: str, mode_value: str):
        jarvis_orchestrator.clear_session(session_id)
        return (
            [],
            [],
            "",
            "",
            None,
            render_jarvis_status_card(
                mode=mode_value,
                status="idle",
                browser_active=jarvis_orchestrator.browser.is_running,
                browser_url=jarvis_orchestrator.browser.current_url,
                playwright_ready=playwright_ready,
                detail="เริ่มบทสนทนาใหม่เรียบร้อย",
            ),
            render_jarvis_timeline(),
            render_jarvis_sources(),
            render_jarvis_approval_card(),
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
        )

    def jarvis_mode_change(mode_value: str) -> str:
        note = (
            "เปิด Online Agent Mode แล้ว — JARVIS สามารถเปิด Browser ค้นหาข้อมูลเว็บสาธารณะได้"
            if mode_value == "online"
            else "อยู่ใน Offline Mode — ทำงานเฉพาะในเครื่อง ไม่เชื่อมต่ออินเทอร์เน็ต"
        )
        return render_jarvis_status_card(
            mode=mode_value,
            status="idle",
            browser_active=jarvis_orchestrator.browser.is_running,
            browser_url=jarvis_orchestrator.browser.current_url,
            playwright_ready=playwright_ready,
            detail=note,
        )

    def jarvis_list_memories() -> list[list[str]]:
        return [
            [m.key, m.value, m.category, m.updated_at]
            for m in jarvis_orchestrator.memory.list_memories(limit=50)
        ]

    def jarvis_delete_memory(key: str) -> tuple[list[list[str]], str]:
        if not (key or "").strip():
            return jarvis_list_memories(), "ระบุ Key ที่ต้องการลบก่อน"
        deleted = jarvis_orchestrator.memory.delete_memory(key)
        msg = f"ลบความจำ '{key.strip()}' แล้ว" if deleted else f"ไม่พบความจำชื่อ '{key.strip()}'"
        return jarvis_list_memories(), msg

    def lab_generate_synthetic_ui(
        corpus_text: str, profile_names: list[str] | None, dataset_id: str
    ) -> tuple[str, str]:
        try:
            lines = [line.strip() for line in (corpus_text or "").splitlines() if line.strip()]
            if not lines:
                raise WorkbenchError("พิมพ์ข้อความอย่างน้อย 1 บรรทัดเพื่อสร้างข้อมูลเสียงสังเคราะห์")
            selected = profile_names or ["clean"]
            augs = [AUGMENTATION_PRESETS[name] for name in selected if name in AUGMENTATION_PRESETS]
            if not augs:
                augs = [AUGMENTATION_PRESETS["clean"]]
            clean_id = (dataset_id or "synthetic_ui").strip().replace(" ", "_")
            items = [
                SyntheticCorpusItem(sample_id=f"{clean_id}_{i:03d}", text=line)
                for i, line in enumerate(lines, start=1)
            ]
            summary = synthetic_generator.generate_dataset(
                items,
                dataset_id=clean_id,
                dataset_version="v1",
                output_dir=speech_lab_dir / "synthetic" / clean_id,
                augmentations=augs,
            )
            info = (
                f"สร้างเสียงสังเคราะห์สำเร็จ {summary.generated_count} ไฟล์ "
                f"(fingerprint={summary.manifest.fingerprint})\n"
                f"Manifest: {summary.manifest_path}"
            )
            return info, str(summary.manifest_path)
        except WorkbenchError as exc:
            raise gr.Error(str(exc)) from None

    def lab_list_experiments_ui() -> list[list[str]]:
        rows: list[list[str]] = []
        for exp in speech_lab_registry.list_experiments(limit=30):
            rows.append(
                [
                    exp.experiment_id,
                    exp.candidate_model,
                    f"{exp.eval_dataset_id} ({exp.eval_split})",
                    f"{exp.cer:.4f}" if exp.cer is not None else "-",
                    f"{exp.wer:.4f}" if exp.wer is not None else "-",
                    f"{exp.rtf:.2f}x" if exp.rtf is not None else "-",
                    exp.real_synthetic_ratio,
                    exp.promotion_status,
                ]
            )
        return rows

    tts_badges: list[tuple[str, str]] = []
    if settings.tts.enabled:
        if readiness["kokoro_ready"]:
            tts_badges.append(("Kokoro 24 kHz Ready", "success"))
        if readiness["lunarlist_ready"]:
            tts_badges.append(("Lunarlist 22 kHz Ready", "blue"))
        if not tts_badges:
            tts_badges.append(("Model Setup Required", "warn"))
    else:
        tts_badges.append(("TTS Disabled", "neutral"))

    asr_badges: list[tuple[str, str]] = []
    if settings.asr.enabled:
        if readiness["openthai_precision"]:
            asr_badges.append((f"OpenThai 2.0 ({readiness['openthai_precision']})", "success"))
        if readiness["typhoon_ready"]:
            asr_badges.append(("Typhoon ASR Ready", "blue"))
        if readiness["diar_enabled"] and readiness["diar_ready"]:
            asr_badges.append(("Diarization Ready", "blue"))
        if not asr_badges:
            asr_badges.append(("Model Setup Required", "warn"))
    else:
        asr_badges.append(("ASR Disabled", "neutral"))

    llm_badges = [
        (f"Ollama · {settings.llm.model}", "blue"),
        (settings.llm.base_url, "success" if settings.llm.enabled else "neutral"),
    ]

    theme = build_theme(gr)
    with gr.Blocks(
        title="Thai Offline AI Workbench",
        analytics_enabled=False,
        theme=theme,
        css=CUSTOM_CSS,
        head=HEAD_SCRIPT,
        delete_cache=(3600, 3600),
    ) as demo:
        gr.HTML(render_navbar(settings, readiness))
        with gr.Tab("Home"):
            gr.HTML(render_home_html(settings, readiness))

        with gr.Tab("NLP"):
            gr.HTML(
                render_page_header(
                    "Thai & English NLP Studio",
                    "วิเคราะห์ ปรับรูปแบบ ตัดคำ และตัดประโยคภาษาไทยผสมอังกฤษแบบออฟไลน์",
                    badges=[
                        (f"Word: {settings.nlp.word_engine}", "blue"),
                        (f"Sentence: {settings.nlp.sentence_engine}", "blue"),
                        ("Ready" if settings.nlp.enabled else "Disabled", "success" if settings.nlp.enabled else "neutral"),
                    ],
                )
            )
            if not settings.nlp.enabled:
                gr.Markdown("NLP ถูกปิดอยู่ — ตั้งค่า `nlp.enabled=true` เพื่อเปิดใช้งาน")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">Text Input</div>'
                        '<div class="aiw-panel-caption">ป้อนข้อความภาษาไทยหรือไทยผสมอังกฤษ (ไม่ถูกบันทึกเป็นไฟล์)</div>'
                    )
                    source = gr.Textbox(
                        label="ข้อความภาษาไทย",
                        placeholder="พิมพ์หรือวางข้อความ เช่น วันนี้มาลองใช้ AI และ Python บน CPU แบบ Offline กันนะคะ",
                        lines=8,
                        interactive=settings.nlp.enabled,
                    )
                    nlp_counter = gr.HTML(count_chars(""))
                    with gr.Row():
                        nlp_clear = gr.Button(
                            "ล้างข้อความ",
                            variant="secondary",
                            interactive=settings.nlp.enabled,
                            scale=1,
                        )
                        button = gr.Button(
                            "วิเคราะห์ข้อความ",
                            variant="primary",
                            interactive=settings.nlp.enabled,
                            scale=2,
                        )
                with gr.Column(scale=7, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">Analysis Output</div>'
                        '<div class="aiw-panel-caption">ผลลัพธ์การปรับรูปแบบข้อความ จำนวนคำ และโทเคน</div>'
                    )
                    with gr.Row():
                        normalized = gr.Textbox(
                            label="ข้อความหลังปรับรูปแบบ (Normalized text)",
                            interactive=False,
                            scale=4,
                        )
                        count = gr.Number(
                            label="จำนวนคำ (Token count)",
                            interactive=False,
                            precision=0,
                            scale=1,
                        )
                    lang_summary = gr.Textbox(
                        label="สรุปภาษา (Thai / English)",
                        interactive=False,
                    )
                    with gr.Row():
                        words = gr.JSON(label="คำ (Word tokens)")
                        sentences = gr.JSON(label="ประโยค (Sentence tokens)")
            source.input(count_chars, source, nlp_counter, api_name=False)
            nlp_clear.click(
                clear_nlp_input,
                outputs=[source, nlp_counter, normalized, words, sentences, count, lang_summary],
                api_name="clear_nlp",
            )
            button.click(
                analyze,
                source,
                [normalized, words, sentences, count, lang_summary],
                api_name="analyze",
                concurrency_limit=1,
            )

        with gr.Tab("TTS"):
            gr.HTML(
                render_page_header(
                    "Text-to-Speech Studio",
                    "Thai & English Speech Synthesis · โหลดโมเดลบน CPU เมื่อกดสร้างเสียงเท่านั้น",
                    badges=tts_badges,
                )
            )
            with gr.Group(elem_classes=["aiw-config-card"]), gr.Row():
                tts_model_choice = gr.Dropdown(
                    label="โมเดล TTS (Model)",
                    info="Kokoro รองรับไทย + อังกฤษ 12 เสียง 24 kHz",
                    choices=[
                        ("Kokoro Thai ONNX · ไทย+อังกฤษ 12 เสียง 24 kHz (แนะนำ)", "kokoro_thai_onnx"),
                        ("PyThaiTTS Lunarlist ONNX · ไทยล้วน 22.05 kHz", "lunarlist_onnx"),
                    ],
                    value=settings.tts.model
                    if settings.tts.model in {"kokoro_thai_onnx", "lunarlist_onnx"}
                    else "kokoro_thai_onnx",
                    interactive=settings.tts.enabled,
                    scale=3,
                )
                tts_voice_choice = gr.Dropdown(
                    label="เสียงพูด (Voice · Kokoro)",
                    info="เลือกบุคลิกเสียงผู้หญิง/ผู้ชาย 12 สไตล์",
                    choices=kokoro_voice_choices(),
                    value=settings.tts.voice or DEFAULT_KOKORO_VOICE,
                    interactive=settings.tts.enabled,
                    scale=3,
                )
                tts_speed_choice = gr.Slider(
                    label="ความเร็วการพูด (Speech Speed)",
                    info="0.80x ช้านุ่มนวล · 1.05x กระชับ · 1.35x เร็ว",
                    minimum=0.8,
                    maximum=1.35,
                    step=0.05,
                    value=settings.tts.speed,
                    interactive=settings.tts.enabled,
                    scale=2,
                )
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">Text Input</div>'
                        '<div class="aiw-panel-caption">พิมพ์ข้อความภาษาไทยหรือไทยผสมอังกฤษที่ต้องการสังเคราะห์เสียง</div>'
                    )
                    tts_source = gr.Textbox(
                        label="ข้อความสำหรับสังเคราะห์เสียง",
                        placeholder="สวัสดีค่ะ วันนี้มาลองใช้ AI และ Python บน CPU แบบ Offline กันนะคะ รับรองว่าฟังเพลิน กระชับ และเข้าใจง่ายแน่นอนค่ะ",
                        lines=8,
                        interactive=settings.tts.enabled,
                    )
                    tts_counter = gr.HTML(count_chars(""))
                    with gr.Row():
                        tts_clear = gr.Button(
                            "ล้างข้อความ",
                            variant="secondary",
                            interactive=settings.tts.enabled,
                            scale=1,
                        )
                        tts_button = gr.Button(
                            "สร้างเสียง (Generate Speech)",
                            variant="primary",
                            interactive=settings.tts.enabled,
                            scale=2,
                        )
                with gr.Column(scale=6, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">Generated Audio</div>'
                        '<div class="aiw-panel-caption">เล่นเสียง ดูคลื่นเสียงจริง และดาวน์โหลดไฟล์ WAV</div>'
                    )
                    tts_audio = gr.Audio(
                        label="เสียงที่สร้าง",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                        show_share_button=False,
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#93C5FD",
                            waveform_progress_color="#2563EB",
                            show_controls=True,
                            skip_length=5,
                        ),
                    )
                    tts_status_card = gr.HTML(render_tts_audio_summary())
                    tts_download = gr.DownloadButton(
                        label="ดาวน์โหลดไฟล์เสียง (.wav)",
                        variant="secondary",
                        interactive=False,
                    )
                    with gr.Accordion("รายละเอียดทางเทคนิค (Technical Metadata)", open=False):
                        tts_info = gr.Textbox(label="ข้อมูลไฟล์", lines=4, interactive=False)
            tts_source.input(count_chars, tts_source, tts_counter, api_name=False)
            tts_clear.click(clear_tts_input, outputs=[tts_source, tts_counter], api_name="clear_tts")
            tts_button.click(
                synthesize,
                [tts_source, tts_model_choice, tts_voice_choice, tts_speed_choice],
                [tts_audio, tts_status_card, tts_download, tts_info],
                api_name="synthesize",
            )
            if not settings.tts.enabled:
                gr.Markdown("TTS ถูกปิดอยู่ — ตั้งค่า `tts.enabled=true` หลังติดตั้งโมเดลในเครื่อง")

        with gr.Tab("ASR"):
            gr.HTML(
                render_page_header(
                    "Speech Recognition Studio (ASR)",
                    "ถอดเสียงภาษาไทยและไทยผสมอังกฤษจากไฟล์ ไมโครโฟน ชุดไฟล์ และระบุผู้พูดบนเครื่อง",
                    badges=asr_badges,
                )
            )
            with gr.Group(elem_classes=["aiw-config-card"]), gr.Row():
                asr_model_choice = gr.Dropdown(
                    label="โมเดล ASR",
                    info="สลับโมเดลได้ทันทีโดยไม่ต้องรีสตาร์ตแอป",
                    choices=[
                        ("OpenThai 2.0 ASR Streaming (iApp ONNX · แนะนำ)", "openthai2_asr"),
                        ("Typhoon ASR (PyThaiASR ONNX)", "typhoon_asr"),
                    ],
                    value=active_asr["model"]
                    if active_asr["model"] in {"typhoon_asr", "openthai2_asr"}
                    else "typhoon_asr",
                    interactive=settings.asr.enabled,
                    scale=3,
                )
                asr_prompt_choice = gr.Radio(
                    label="โหมดภาษา OpenThai 2.0",
                    choices=[
                        ("ไทยผสมอังกฤษ (th-mix)", "th-mix"),
                        ("ภาษาไทยล้วน (th-TH)", "th-TH"),
                    ],
                    value=settings.asr.openthai_prompt,
                    interactive=settings.asr.enabled,
                    scale=2,
                )
                asr_model_note = gr.Textbox(
                    label="โมเดลที่เลือกอยู่",
                    value=update_asr_selection(active_asr["model"], active_asr["prompt"]),
                    interactive=False,
                    scale=3,
                )
            asr_model_choice.change(
                update_asr_selection,
                [asr_model_choice, asr_prompt_choice],
                asr_model_note,
                api_name="select_asr_model",
            )
            asr_prompt_choice.change(
                update_asr_selection,
                [asr_model_choice, asr_prompt_choice],
                asr_model_note,
                api_name="select_asr_prompt",
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">Audio Input</div>'
                        '<div class="aiw-panel-caption">อัปโหลด WAV/FLAC/MP3/OGG หรืออัดเสียงจากไมโครโฟน</div>'
                    )
                    asr_audio = gr.Audio(
                        label="ไฟล์เสียง (WAV/FLAC/MP3/OGG)",
                        type="filepath",
                        sources=["upload", "microphone"],
                        interactive=settings.asr.enabled,
                    )
                    asr_button = gr.Button(
                        "ถอดเสียง (Transcribe)",
                        variant="primary",
                        interactive=settings.asr.enabled,
                    )
                with gr.Column(scale=7, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">Transcription Result</div>'
                        '<div class="aiw-panel-caption">ข้อความถอดเสียงและช่วงเวลาของแต่ละประโยค</div>'
                    )
                    asr_text = gr.Textbox(label="ข้อความถอดเสียง", lines=6, interactive=False)
                    asr_info = gr.Textbox(label="ข้อมูล", interactive=False)
                    with gr.Accordion("รายละเอียดช่วงเวลา (Segments)", open=False):
                        asr_segments = gr.JSON(label="ส่วนของเสียง (Segments)")
            asr_button.click(
                transcribe, asr_audio, [asr_text, asr_segments, asr_info], api_name="transcribe"
            )

            with gr.Accordion("Near Real-Time / Live Mic (Phase 3.5)", open=False):
                gr.Markdown(
                    "เริ่ม session แล้วพูดผ่านไมโครโฟน ผลที่ถอดได้เป็นส่วนที่ยืนยันแล้ว "
                    "(รองรับทั้ง OpenThai 2.0 และ Typhoon ASR ตามโมเดลที่เลือกด้านบน)"
                )
                live_session = gr.State("")
                with gr.Row():
                    live_start = gr.Button(
                        "เริ่ม Live transcription",
                        variant="primary",
                        interactive=settings.asr.enabled,
                    )
                    live_stop = gr.Button("หยุดและสรุปผล", variant="secondary", interactive=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        live_audio = gr.Audio(
                            label="ไมโครโฟน (Live)",
                            type="numpy",
                            sources=["microphone"],
                            streaming=True,
                            interactive=False,
                        )
                        live_info = gr.Textbox(label="สถานะ Live", interactive=False)
                    with gr.Column(scale=7):
                        live_partial = gr.Textbox(label="Partial hypothesis", interactive=False)
                        live_final = gr.Textbox(
                            label="ข้อความที่ยืนยันแล้ว", lines=5, interactive=False
                        )
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
                    outputs=start_outputs,
                    api_name="start_live_asr",
                )
                live_audio.stream(
                    process_live_chunk,
                    [live_audio, live_session],
                    [live_session, live_partial, live_final, live_info],
                    api_name="live_asr_chunk",
                    concurrency_limit=1,
                )
                live_stop.click(
                    stop_live_session,
                    live_session,
                    start_outputs,
                    api_name="stop_live_asr",
                )

            with gr.Accordion(
                "Near Real-Time ASR + Speaker Diarization (Phase 5.5)", open=False
            ):
                gr.Markdown(
                    "รับไมโครโฟนสตรีมเดียวแล้วส่งต่อให้ทั้ง ASR และ Speaker Diarization แบบสด "
                    "(ระบุเป็น `SPEAKER_00`, `SPEAKER_01` ภายใน session นี้เท่านั้น ไม่อนุมานบทบาท)"
                )
                live_spk_session = gr.State("")
                with gr.Row():
                    live_spk_start = gr.Button(
                        "เริ่ม Live + ระบุผู้พูด",
                        variant="primary",
                        interactive=live_speakers_enabled,
                    )
                    live_spk_stop = gr.Button(
                        "หยุดและสรุปผลผู้พูด", variant="secondary", interactive=False
                    )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        live_spk_audio = gr.Audio(
                            label="ไมโครโฟน (Live + Speakers)",
                            type="numpy",
                            sources=["microphone"],
                            streaming=True,
                            interactive=False,
                        )
                        live_spk_info = gr.Textbox(
                            label="สถานะ Live + Diarization", interactive=False
                        )
                    with gr.Column(scale=7):
                        live_spk_transcript = gr.Textbox(
                            label="ข้อความที่ถอดพร้อมผู้พูด (Live Speaker Transcript)",
                            lines=6,
                            interactive=False,
                        )
                        live_spk_turns = gr.JSON(label="Speaker Turns (Live)")
                live_spk_outputs = [
                    live_spk_session,
                    live_spk_transcript,
                    live_spk_turns,
                    live_spk_info,
                    live_spk_audio,
                    live_spk_start,
                    live_spk_stop,
                ]
                live_spk_start.click(
                    begin_live_speaker_preparation,
                    outputs=live_spk_outputs,
                    api_name="begin_live_speaker_preparation",
                ).then(
                    start_live_speaker_session,
                    outputs=live_spk_outputs,
                    api_name="start_live_speaker_asr",
                )
                live_spk_audio.stream(
                    process_live_speaker_chunk,
                    [live_spk_audio, live_spk_session],
                    [live_spk_session, live_spk_transcript, live_spk_turns, live_spk_info],
                    api_name="live_speaker_asr_chunk",
                    concurrency_limit=1,
                )
                live_spk_stop.click(
                    stop_live_speaker_session,
                    live_spk_session,
                    live_spk_outputs,
                    api_name="stop_live_speaker_asr",
                )
                if not live_speakers_enabled:
                    gr.Markdown(
                        "Live ASR + Speaker Diarization ต้องเปิดทั้ง `asr.enabled=true` "
                        "และ `diarization.enabled=true`"
                    )

            with gr.Accordion("Speaker Diarization & ถอดเสียงพร้อมผู้พูด (Phase 5)", open=False):
                gr.Markdown(
                    "ระบุผู้พูดแบบไม่ระบุตัวตนเป็น `SPEAKER_00`, `SPEAKER_01` โดยไม่อนุมาน Agent/Customer"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        diarization_audio = gr.Audio(
                            label="ไฟล์เสียงสำหรับ Speaker Diarization",
                            type="filepath",
                            sources=["upload"],
                            interactive=settings.diarization.enabled,
                        )
                        with gr.Row():
                            diarize_button = gr.Button(
                                "ระบุผู้พูด",
                                variant="secondary",
                                interactive=settings.diarization.enabled,
                            )
                            speakers_button = gr.Button(
                                "ถอดเสียงพร้อมผู้พูด",
                                variant="primary",
                                interactive=settings.diarization.enabled,
                            )
                    with gr.Column(scale=7):
                        diarization_info = gr.Textbox(
                            label="ข้อมูล Speaker Diarization", interactive=False
                        )
                        diarization_turns = gr.JSON(label="Speaker Turns")
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

            with gr.Accordion("Batch Files · ถอดเสียงหลายไฟล์พร้อม Export JSON/CSV (Phase 3.6)", open=False):
                gr.Markdown("ถอดเสียงหลายไฟล์แบบลำดับเพื่อใช้ CPU และโมเดลร่วมกันอย่างปลอดภัย")
                batch_files = gr.File(
                    label="ไฟล์เสียงหลายไฟล์ (WAV, FLAC, MP3, OGG)",
                    file_count="multiple",
                    type="filepath",
                    file_types=["audio"],
                    interactive=settings.asr.enabled,
                )
                batch_button = gr.Button(
                    "ถอดเสียงแบบชุด",
                    variant="primary",
                    interactive=settings.asr.enabled,
                )
                batch_info = gr.Textbox(label="สรุป Batch", interactive=False)
                batch_table = gr.Dataframe(
                    headers=["File", "Status", "Duration (s)", "Processing (s)", "RTF", "Transcript", "Error"],
                    datatype=["str"] * 7,
                    interactive=False,
                    label="ผลลัพธ์แบบชุด",
                )
                with gr.Row():
                    batch_json = gr.File(label="ส่งออก JSON", interactive=False)
                    batch_csv = gr.File(label="ส่งออก CSV", interactive=False)
                batch_button.click(
                    transcribe_batch,
                    batch_files,
                    [batch_table, batch_info, batch_json, batch_csv],
                    api_name="transcribe_batch",
                    concurrency_limit=1,
                )

            with gr.Accordion(
                "Speech AI Continuous Improvement Lab (Phase 6.5 · Synthetic Data & Experiment Registry)",
                open=False,
            ):
                gr.Markdown(
                    "สร้างชุดเสียงสังเคราะห์พร้อม Augmentation (`train_synthetic` เท่านั้น ห้ามปน `validation_real`/`test_real`) "
                    "และติดตามผล Experiment / Promotion Gate บนชุดข้อมูลจริงที่คงเดิม"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        lab_dataset_id = gr.Textbox(
                            label="ชื่อชุดข้อมูลสังเคราะห์ (Dataset ID)",
                            value="thai_domain_synth",
                        )
                        lab_corpus = gr.Textbox(
                            label="ข้อความต้นฉบับสำหรับสร้างเสียงสังเคราะห์ (1 ประโยคต่อบรรทัด)",
                            placeholder="สวัสดีครับ ขอสอบถามเรื่องต่ออายุกรมธรรม์ประกันภัย\nยืนยันนัดชำระค่าเบี้ยประกันผ่านแอปครับ",
                            lines=4,
                            interactive=settings.tts.enabled,
                        )
                        lab_profiles = gr.CheckboxGroup(
                            label="โปรไฟล์ Augmentation",
                            choices=list(AUGMENTATION_PRESETS.keys()),
                            value=["clean", "telephone_bandpass"],
                            interactive=settings.tts.enabled,
                        )
                        lab_gen_btn = gr.Button(
                            "สร้างชุดเสียงสังเคราะห์ (Generate Synthetic Manifest)",
                            variant="primary",
                            interactive=settings.tts.enabled,
                        )
                        lab_gen_status = gr.Textbox(label="ผลลัพธ์ Synthetic Lab", lines=3, interactive=False)
                        lab_manifest_file = gr.File(label="ดาวน์โหลด Manifest JSONL", interactive=False)
                    with gr.Column(scale=6):
                        lab_exp_refresh = gr.Button(
                            "รีเฟรชตาราง Experiment & Promotion Gate", variant="secondary"
                        )
                        lab_exp_table = gr.Dataframe(
                            headers=[
                                "Experiment ID",
                                "Model",
                                "Holdout Eval",
                                "CER",
                                "WER",
                                "RTF",
                                "Real:Synth",
                                "Status",
                            ],
                            datatype=["str"] * 8,
                            value=[],
                            interactive=False,
                            label="ASR Experiment Registry",
                        )
                lab_gen_btn.click(
                    lab_generate_synthetic_ui,
                    [lab_corpus, lab_profiles, lab_dataset_id],
                    [lab_gen_status, lab_manifest_file],
                    api_name="lab_generate_synthetic",
                    concurrency_limit=1,
                )
                lab_exp_refresh.click(
                    lab_list_experiments_ui,
                    outputs=lab_exp_table,
                    api_name="lab_list_experiments",
                )

            with gr.Accordion(
                "Conversation Intelligence (Phase 7 · สรุปบทสนทนา & Action Items)", open=False
            ):
                gr.Markdown(
                    "วางบทสนทนาแบบ `SPEAKER_00: ข้อความ` หรือเลือกไฟล์เสียง (ASR + Diarization) "
                    "สรุปและ Action Items มาจาก Local LLM และอ้างอิงช่วงคำพูดจริงเสมอ "
                    "(ไม่เดาบทบาท Agent/Customer)"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        ci_pasted = gr.Textbox(
                            label="บทสนทนา (วางข้อความ)",
                            placeholder="SPEAKER_00: สวัสดีครับ ขอสอบถามเรื่องกรมธรรม์\nSPEAKER_01: ได้ครับ เดี๋ยวส่งเอกสารให้ภายในวันศุกร์",
                            lines=6,
                        )
                        ci_audio = gr.Audio(
                            label="หรือไฟล์เสียงบทสนทนา",
                            type="filepath",
                            sources=["upload"],
                            interactive=settings.asr.enabled and settings.diarization.enabled,
                        )
                        ci_terms = gr.Textbox(
                            label="คำศัพท์เฉพาะโดเมน (คั่นด้วยจุลภาค)",
                            placeholder="กรมธรรม์, เบี้ยประกัน, ต่ออายุ",
                        )
                        ci_use_llm = gr.Checkbox(
                            label="สรุปด้วย Local LLM",
                            value=settings.llm.enabled,
                            interactive=settings.llm.enabled,
                        )
                        ci_btn = gr.Button("วิเคราะห์บทสนทนา", variant="primary")
                    with gr.Column(scale=7):
                        ci_report_md = gr.Markdown()
                        ci_info = gr.Textbox(label="สถานะ", interactive=False)
                        with gr.Accordion("รายงาน JSON", open=False):
                            ci_report_json = gr.JSON(label="Conversation Report")
                ci_btn.click(
                    conversation_intel_ui,
                    [ci_audio, ci_pasted, ci_terms, ci_use_llm],
                    [ci_report_md, ci_report_json, ci_info],
                    api_name="conversation_intel",
                    concurrency_limit=1,
                )

        with gr.Tab("Benchmark"):
            gr.HTML(
                render_page_header(
                    "Local ASR Performance Benchmark",
                    "วัดเวลา decode + normalize + ASR inference จริงบนเครื่องนี้; ไม่วัด browser capture latency และไม่เก็บ transcript ในรายงาน",
                    badges=[("Local-only", "success"), ("Measured RTF", "blue")],
                )
            )
            with gr.Column(elem_classes=["aiw-panel"]):
                benchmark_files = gr.File(
                    label="ไฟล์เสียงสำหรับวัดผล (WAV/FLAC/MP3/OGG)",
                    file_count="multiple",
                    type="filepath",
                    file_types=["audio"],
                    interactive=settings.asr.enabled,
                )
                benchmark_button = gr.Button(
                    "วัด ASR RTF / Latency", variant="primary", interactive=settings.asr.enabled
                )
                benchmark_summary = gr.Textbox(label="สรุปผลที่วัดได้", interactive=False, lines=3)
                benchmark_table = gr.Dataframe(
                    headers=["File", "Status", "Audio (s)", "Elapsed (s)", "RTF", "Error"],
                    datatype=["str"] * 6,
                    interactive=False,
                    label="รายละเอียดการวัด",
                )
                with gr.Row():
                    benchmark_json = gr.File(label="รายงาน JSON", interactive=False)
                    benchmark_csv = gr.File(label="รายงาน CSV", interactive=False)
                benchmark_button.click(
                    benchmark_asr_ui,
                    [benchmark_files, asr_model_choice, asr_prompt_choice],
                    [benchmark_table, benchmark_summary, benchmark_json, benchmark_csv],
                    api_name="benchmark_asr",
                    concurrency_limit=1,
                )

        with gr.Tab("Local LLM"):
            gr.HTML(
                render_page_header(
                    "Local LLM Assistant",
                    "แชตกับโมเดลภาษาในเครื่องผ่าน Ollama loopback เท่านั้น ไม่ส่งข้อมูลออกคลาวด์",
                    badges=llm_badges,
                )
            )
            llm_history = gr.State([])
            with gr.Column(elem_classes=["aiw-panel", "aiw-chat-shell"]):
                with gr.Accordion("ตั้งค่า System Prompt", open=False):
                    llm_system = gr.Textbox(
                        label="System Prompt",
                        value=settings.llm.system_prompt,
                        lines=2,
                        interactive=settings.llm.enabled,
                    )
                llm_chat = gr.Chatbot(label="Local Chat", type="messages", height=460)
                llm_message = gr.Textbox(
                    label="ข้อความ",
                    placeholder="พิมพ์คำถามภาษาไทยหรือไทยผสมอังกฤษเพื่อคุยกับ Local LLM...",
                    lines=3,
                    interactive=settings.llm.enabled,
                )
                with gr.Row():
                    llm_clear = gr.Button(
                        "ล้างประวัติ",
                        variant="secondary",
                        interactive=settings.llm.enabled,
                        scale=1,
                    )
                    llm_send = gr.Button(
                        "ส่งข้อความ (Send)",
                        variant="primary",
                        interactive=settings.llm.enabled,
                        scale=3,
                    )
                llm_status = gr.Textbox(label="สถานะ Local LLM", interactive=False)
                with gr.Accordion(
                    "Local Voice Assistant (Phase 6 · 100% Offline Voice-to-Voice)",
                    open=False,
                ):
                    gr.Markdown(
                        "คุยด้วยเสียงแบบออฟไลน์ 100% โดยตรง (`ASR → Local LLM → TTS`) "
                        "ใช้โมเดลเดิมร่วมกันโดยไม่โหลดซ้ำ พร้อมวัดเวลาจริงแต่ละขั้นตอน"
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=6):
                            va_mic = gr.Audio(
                                label="ไมโครโฟน / ไฟล์เสียง (Local Voice Assistant)",
                                sources=["microphone", "upload"],
                                type="filepath",
                                interactive=settings.llm.enabled and settings.asr.enabled,
                            )
                            va_speak = gr.Checkbox(
                                label="ตอบกลับเป็นเสียง (TTS)",
                                value=settings.tts.enabled,
                                interactive=settings.tts.enabled,
                            )
                            va_send = gr.Button(
                                "พูดคุยด้วยเสียง (Voice Chat)",
                                variant="primary",
                                interactive=settings.llm.enabled and settings.asr.enabled,
                            )
                        with gr.Column(scale=6):
                            va_transcript = gr.Textbox(
                                label="ข้อความที่ถอดเสียง (Voice Input)",
                                lines=2,
                                interactive=False,
                            )
                            va_audio = gr.Audio(
                                label="เสียงตอบกลับ (Voice Assistant Reply)",
                                type="filepath",
                                interactive=False,
                                show_download_button=True,
                                show_share_button=False,
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#93C5FD",
                                    waveform_progress_color="#2563EB",
                                    show_controls=True,
                                ),
                            )
                            va_latency = gr.HTML(render_voice_latency_card())
                with gr.Accordion(
                    "Local RAG Knowledge Base (ค้นหาและถาม-ตอบเอกสารในเครื่อง · 100% Offline)",
                    open=False,
                ):
                    gr.Markdown(
                        "นำเข้าเอกสาร `.txt`, `.md`, `.csv`, `.json`, `.html` หรือวางข้อความลงฐานข้อมูล SQLite ในเครื่อง "
                        "(ค้นหาด้วย Hybrid PyThaiNLP Word + Character n-gram TF-IDF โดยไม่ต้องดาวน์โหลดโมเดล Embedding)"
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5):
                            rag_files = gr.File(
                                label="อัปโหลดไฟล์เอกสาร (TXT, MD, CSV, JSON, HTML)",
                                file_count="multiple",
                                type="filepath",
                            )
                            rag_title = gr.Textbox(
                                label="ชื่อเอกสาร (กรณีวางข้อความ)",
                                placeholder="เช่น คู่มือกรมธรรม์ประกันภัย",
                            )
                            rag_text = gr.Textbox(
                                label="เนื้อหาเอกสาร (วางข้อความ)",
                                lines=3,
                                placeholder="วางเนื้อหาภาษาไทยหรืออังกฤษที่ต้องการให้ค้นหาและถาม-ตอบ...",
                            )
                            with gr.Row():
                                rag_ingest_btn = gr.Button("นำเข้าเอกสาร (Ingest)", variant="primary")
                                rag_clear_btn = gr.Button("ล้างเอกสารทั้งหมด", variant="secondary")
                            rag_status = gr.Textbox(label="สถานะ Local RAG", interactive=False)
                        with gr.Column(scale=7):
                            rag_docs_table = gr.Dataframe(
                                headers=["Doc ID", "Title", "Chunks", "Chars", "Created"],
                                datatype=["str"] * 5,
                                value=[],
                                interactive=False,
                                label="เอกสารในฐานความรู้ (Indexed Documents)",
                            )
                            rag_question = gr.Textbox(
                                label="คำถามค้นหาในเอกสาร (RAG Question)",
                                placeholder="พิมพ์คำถามที่ต้องการค้นคำตอบจากเอกสารที่นำเข้า...",
                            )
                            rag_use_llm = gr.Checkbox(
                                label="สรุปคำตอบด้วย Local LLM พร้อมอ้างอิง [Doc #1]",
                                value=settings.llm.enabled,
                                interactive=settings.llm.enabled,
                            )
                            rag_ask_btn = gr.Button("ค้นหาและถามคำตอบ (Ask RAG)", variant="primary")
                            rag_answer = gr.Textbox(label="คำตอบจากเอกสาร", lines=5, interactive=False)
                            with gr.Accordion("ชิ้นส่วนเอกสารที่ค้นพบ (Retrieved Chunks)", open=False):
                                rag_chunks_json = gr.JSON(label="Top Matched Chunks", value=[])
            rag_ingest_btn.click(
                rag_ingest_ui,
                [rag_files, rag_title, rag_text],
                [rag_docs_table, rag_status],
                api_name="rag_ingest",
            )
            rag_clear_btn.click(
                rag_clear_ui,
                outputs=[rag_docs_table, rag_status],
                api_name="rag_clear",
            )
            rag_ask_btn.click(
                rag_ask_ui,
                [rag_question, rag_use_llm],
                [rag_answer, rag_chunks_json],
                api_name="rag_ask",
            )
            llm_send.click(
                send_local_chat,
                [llm_message, llm_system, llm_history],
                [llm_chat, llm_message, llm_status],
                api_name="local_llm_chat",
                concurrency_limit=1,
            ).then(lambda history: history, llm_chat, llm_history)
            va_send.click(
                send_local_voice_chat,
                [va_mic, llm_system, va_speak, llm_history],
                [llm_chat, llm_history, va_transcript, va_audio, va_latency, llm_status],
                api_name="local_voice_chat",
                concurrency_limit=1,
            )
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

        with gr.Tab("JARVIS AI Assistant"):
            jarvis_badges = [
                (f"LLM: {settings.llm.model}", "blue"),
                (
                    "Playwright Ready" if playwright_ready else "Playwright Setup Needed",
                    "success" if playwright_ready else "warn",
                ),
                ("SQLite Local Memory", "success"),
            ]
            gr.HTML(
                render_page_header(
                    "JARVIS AI Assistant (Phase 8)",
                    "ผู้ช่วยส่วนตัวสั่งงานด้วยเสียงหรือข้อความ · วางแผนหลายขั้นตอน · เปิด Browser ค้นเว็บจริง · ขอความยินยอมก่อนทำรายการสำคัญ",
                    badges=jarvis_badges,
                )
            )
            jarvis_session = gr.State("jarvis-ui-default")
            jarvis_history = gr.State([])
            jarvis_pending_approval_id = gr.State("")

            with gr.Group(elem_classes=["aiw-config-card"]), gr.Row():
                jarvis_mode = gr.Radio(
                    label="โหมดการทำงาน (Operating Mode)",
                    choices=[
                        ("Offline Mode (Local Only · ค่าเริ่มต้น)", "offline"),
                        ("Online Agent Mode (Web Search + Visible Browser)", "online"),
                    ],
                    value=agent_settings.agent.default_mode,
                    scale=4,
                )
                jarvis_auto_speak = gr.Checkbox(
                    label="พูดคำตอบกลับอัตโนมัติ (TTS)",
                    value=bool(agent_settings.voice.auto_speak_responses and settings.tts.enabled),
                    interactive=settings.tts.enabled,
                    scale=2,
                )
                jarvis_stop_btn = gr.Button("หยุดการทำงาน (Stop)", variant="secondary", scale=1)
                jarvis_close_browser_btn = gr.Button("ปิด Browser", variant="secondary", scale=1)
                jarvis_reset_btn = gr.Button("บทสนทนาใหม่", variant="secondary", scale=1)

            with gr.Row(equal_height=False):
                with gr.Column(scale=7, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">JARVIS Conversation &amp; Voice Command</div>'
                        '<div class="aiw-panel-caption">สนทนาต่อเนื่องหลายรอบทั้งไทยและอังกฤษ รองรับทั้งพิมพ์และ Push-to-Talk</div>'
                    )
                    jarvis_chat = gr.Chatbot(label="JARVIS Chat", type="messages", height=420)
                    jarvis_input = gr.Textbox(
                        label="คำสั่งถึง JARVIS",
                        placeholder="เช่น ช่วยหาสถานที่ท่องเที่ยวเชียงใหม่ 3 วัน 2 คืน เน้นธรรมชาติและคาเฟ่ หรือถามคำถามทั่วไป...",
                        lines=2,
                    )
                    jarvis_send_btn = gr.Button("ส่งคำสั่ง (Send to JARVIS)", variant="primary")

                    with (
                        gr.Accordion("สั่งงานด้วยเสียง (Push-to-Talk Voice Control)", open=True),
                        gr.Row(equal_height=False),
                    ):
                            with gr.Column(scale=6):
                                jarvis_mic = gr.Audio(
                                    label="Push-to-Talk (ไมโครโฟน / ไฟล์เสียง)",
                                    sources=["microphone", "upload"],
                                    type="filepath",
                                    interactive=settings.asr.enabled,
                                )
                                jarvis_voice_btn = gr.Button(
                                    "พูดสั่ง JARVIS (Voice Command)",
                                    variant="primary",
                                    interactive=settings.asr.enabled,
                                )
                            with gr.Column(scale=6):
                                jarvis_transcript = gr.Textbox(
                                    label="ข้อความที่ถอดเสียง (Transcript)",
                                    lines=2,
                                    interactive=False,
                                )
                                jarvis_reply_audio = gr.Audio(
                                    label="เสียงตอบกลับจาก JARVIS",
                                    type="filepath",
                                    interactive=False,
                                    show_download_button=True,
                                    show_share_button=False,
                                    waveform_options=gr.WaveformOptions(
                                        waveform_color="#93C5FD",
                                        waveform_progress_color="#2563EB",
                                        show_controls=True,
                                    ),
                                )

                with gr.Column(scale=5, elem_classes=["aiw-panel"]):
                    gr.HTML(
                        '<div class="aiw-panel-title">Agent Status, Timeline &amp; Safety</div>'
                        '<div class="aiw-panel-caption">ติดตามสถานะ Browser ขั้นตอนการวางแผน แหล่งอ้างอิง และคำขออนุมัติ</div>'
                    )
                    jarvis_status_html = gr.HTML(
                        render_jarvis_status_card(
                            mode=agent_settings.agent.default_mode,
                            status="idle",
                            playwright_ready=playwright_ready,
                        )
                    )
                    jarvis_approval_html = gr.HTML(render_jarvis_approval_card())
                    with gr.Row():
                        jarvis_approve_btn = gr.Button(
                            "อนุมัติ (Approve)", variant="primary", interactive=False
                        )
                        jarvis_deny_btn = gr.Button(
                            "ปฏิเสธ (Deny)", variant="secondary", interactive=False
                        )

                    gr.HTML('<div class="aiw-panel-title" style="margin-top:12px;">Task Progress Timeline</div>')
                    jarvis_timeline_html = gr.HTML(render_jarvis_timeline())

                    gr.HTML('<div class="aiw-panel-title" style="margin-top:12px;">Source Links (แหล่งอ้างอิง)</div>')
                    jarvis_sources_html = gr.HTML(render_jarvis_sources())

                    with gr.Accordion("ประวัติการใช้เครื่องมือ (Tool Execution History)", open=False):
                        jarvis_tools_json = gr.JSON(label="Tool Execution History", value=[])

                    with gr.Accordion("ความจำส่วนตัวของ JARVIS (SQLite Memory)", open=False):
                        jarvis_mem_table = gr.Dataframe(
                            headers=["Key", "Value", "Category", "Updated"],
                            datatype=["str", "str", "str", "str"],
                            value=[],
                            interactive=False,
                            label="Saved Preferences & Facts",
                        )
                        with gr.Row():
                            jarvis_mem_refresh = gr.Button("รีเฟรชความจำ", variant="secondary")
                            jarvis_mem_del_key = gr.Textbox(
                                label="Key ที่ต้องการลบ", placeholder="เช่น travel_style"
                            )
                            jarvis_mem_del_btn = gr.Button("ลบความจำ", variant="secondary")
                        jarvis_mem_msg = gr.Textbox(label="สถานะความจำ", interactive=False)

            jarvis_outputs = [
                jarvis_chat,
                jarvis_history,
                jarvis_input,
                jarvis_transcript,
                jarvis_reply_audio,
                jarvis_status_html,
                jarvis_timeline_html,
                jarvis_sources_html,
                jarvis_approval_html,
                jarvis_pending_approval_id,
                jarvis_approve_btn,
                jarvis_deny_btn,
                jarvis_tools_json,
            ]
            jarvis_mode.change(
                jarvis_mode_change, jarvis_mode, jarvis_status_html, api_name="jarvis_mode"
            )
            jarvis_send_btn.click(
                jarvis_send_text,
                [jarvis_input, jarvis_mode, jarvis_auto_speak, jarvis_session, jarvis_history],
                jarvis_outputs,
                api_name="jarvis_send",
                concurrency_limit=1,
            )
            jarvis_voice_btn.click(
                jarvis_send_voice,
                [jarvis_mic, jarvis_mode, jarvis_auto_speak, jarvis_session, jarvis_history],
                jarvis_outputs,
                api_name="jarvis_voice",
                concurrency_limit=1,
            )
            jarvis_approve_btn.click(
                jarvis_approve,
                [jarvis_pending_approval_id, jarvis_auto_speak, jarvis_session, jarvis_history],
                jarvis_outputs,
                api_name="jarvis_approve",
            )
            jarvis_deny_btn.click(
                jarvis_deny,
                [jarvis_pending_approval_id, jarvis_auto_speak, jarvis_session, jarvis_history],
                jarvis_outputs,
                api_name="jarvis_deny",
            )
            jarvis_stop_btn.click(
                jarvis_stop, [jarvis_session, jarvis_mode], jarvis_status_html, api_name="jarvis_stop"
            )
            jarvis_close_browser_btn.click(
                jarvis_close_browser, jarvis_mode, jarvis_status_html, api_name="jarvis_close_browser"
            )
            jarvis_reset_btn.click(
                jarvis_reset_session,
                [jarvis_session, jarvis_mode],
                jarvis_outputs,
                api_name="jarvis_reset",
            )
            jarvis_mem_refresh.click(
                jarvis_list_memories, outputs=jarvis_mem_table, api_name="jarvis_list_memory"
            )
            jarvis_mem_del_btn.click(
                jarvis_delete_memory,
                jarvis_mem_del_key,
                [jarvis_mem_table, jarvis_mem_msg],
                api_name="jarvis_delete_memory",
            )

        with gr.Tab("System"):
            gr.HTML(
                render_page_header(
                    "System & Runtime Monitor",
                    "ตรวจสอบฮาร์ดแวร์ สภาพแวดล้อม Python แคชโมเดล และนโยบายออฟไลน์ โดยไม่โหลดโมเดลหนัก",
                    badges=[
                        ("Offline Guard On" if settings.app.offline_runtime else "Local Mode", "success" if settings.app.offline_runtime else "warn"),
                        (f"{settings.app.host}:{settings.app.port}", "blue"),
                    ],
                )
            )
            with gr.Column(elem_classes=["aiw-panel"]):
                refresh = gr.Button("ตรวจสอบระบบ (Refresh Diagnostics)", variant="primary")
                system_cards = gr.HTML(
                    '<div class="aiw-audio-status">'
                    '<div class="aiw-audio-status-top">'
                    '<span class="aiw-pill aiw-pill-blue">พร้อมตรวจสอบ</span>'
                    '<span style="font-size:13px;color:#64748B;">กด <strong>ตรวจสอบระบบ</strong> เพื่อแสดงข้อมูล Runtime, แคชโมเดล และปลายทาง Local ทั้งหมด</span>'
                    "</div></div>"
                )
                with gr.Accordion("รายงานแบบละเอียด (Raw Doctor Report)", open=False):
                    status = gr.Textbox(label="สถานะระบบ", lines=22, interactive=False)
                with gr.Accordion(
                    "One-Click Model Manager & Portable Desktop Packaging", open=False
                ):
                    gr.Markdown(
                        "ตรวจสอบขนาดและสถานะแคชโมเดลทั้งหมดในเครื่อง การดาวน์โหลดจะทำงานเฉพาะเมื่อติ๊กเปิด **Setup Mode** เท่านั้น"
                    )
                    mm_refresh_btn = gr.Button("ตรวจสอบสถานะโมเดล (Inspect Model Caches)", variant="secondary")
                    mm_table = gr.Dataframe(
                        headers=["Model Key", "Category", "Status", "Cache Size", "Detail", "Setup Command"],
                        datatype=["str"] * 6,
                        value=model_manager.to_table_rows(),
                        interactive=False,
                        label="Managed Local Models",
                    )
                    with gr.Row():
                        mm_choice = gr.Dropdown(
                            label="เลือกโมเดลที่ต้องการติดตั้ง/อัปเดต",
                            choices=list(ModelManager.SCRIPT_MAP.keys()),
                            value="kokoro_thai_onnx",
                            scale=3,
                        )
                        mm_allow_net = gr.Checkbox(
                            label="เปิด Setup Mode (อนุญาตดาวน์โหลดโมเดล)",
                            value=False,
                            scale=2,
                        )
                        mm_install_btn = gr.Button("ติดตั้งโมเดลที่เลือก", variant="primary", scale=2)
                    with gr.Row():
                        mm_package_btn = gr.Button(
                            "สร้างชุด Portable Desktop Bundle (.bat)", variant="secondary"
                        )
                    mm_status = gr.Textbox(label="ผลลัพธ์ Model Manager / Packaging", lines=4, interactive=False)
            refresh.click(system_status, outputs=[system_cards, status], api_name="system_status")
            mm_refresh_btn.click(list_managed_models_ui, outputs=mm_table, api_name="models_list")
            mm_install_btn.click(
                install_managed_model_ui,
                [mm_choice, mm_allow_net],
                [mm_table, mm_status],
                api_name="models_install",
            )
            mm_package_btn.click(
                create_portable_bundle_ui, outputs=mm_status, api_name="package_portable"
            )
        gr.HTML(render_footer(settings))
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
