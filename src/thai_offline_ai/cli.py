from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from thai_offline_ai.adapters.asr import ASRAdapter
from thai_offline_ai.adapters.diarization import DiarizationAdapter
from thai_offline_ai.adapters.nlp import NLPAdapter
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.batch import BatchASRProcessor, export_batch_result
from thai_offline_ai.config import load_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.evaluation import (
    ASREvaluator,
    export_evaluation_report,
    load_domain_terms,
    load_manifest,
)
from thai_offline_ai.llm_types import ChatMessage
from thai_offline_ai.runtime import prepare_runtime
from thai_offline_ai.speaker_composition import compose_speaker_transcript
from thai_offline_ai.streaming import StreamingASRService
from thai_offline_ai.streaming_diarization import StreamingSpeakerASRService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thai-offline-ai")
    parser.add_argument("--config", help="Path to an existing YAML configuration")
    sub = parser.add_subparsers(dest="command", required=True)
    nlp = sub.add_parser("nlp", help="Analyze Thai text locally")
    nlp.add_argument("text", nargs="?", help="Thai text (quote text containing spaces)")
    nlp.add_argument("--text", dest="text_option", help="Compatible alternative to positional text")
    tts = sub.add_parser("tts", help="Generate Thai speech locally")
    tts.add_argument("text", nargs="?", help="Thai text (quote text containing spaces)")
    tts.add_argument("--text", dest="text_option", help="Compatible alternative to positional text")
    tts.add_argument("--model", help="TTS model: kokoro_thai_onnx or lunarlist_onnx")
    tts.add_argument("--voice", help="Kokoro Thai voice ID (e.g. f_young_bright, f_young_clear)")
    tts.add_argument("--speed", type=float, help="Speech speed multiplier (0.7 to 1.5)")
    asr = sub.add_parser("asr", help="Transcribe a supported local audio file")
    asr.add_argument("audio", nargs="?")
    asr.add_argument("--audio", dest="audio_option")
    asr.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    asr.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    diarize = sub.add_parser("diarize", help="Identify anonymous speakers in a local audio file")
    diarize.add_argument("audio", nargs="?")
    diarize.add_argument("--audio", dest="audio_option")
    speakers = sub.add_parser(
        "transcribe-speakers", help="Run local ASR plus timestamp-based speaker attribution"
    )
    speakers.add_argument("audio", nargs="?")
    speakers.add_argument("--audio", dest="audio_option")
    speakers.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    speakers.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    batch = sub.add_parser("asr-batch", help="Transcribe supported audio files in a local folder")
    batch.add_argument("directory", help="Local folder containing WAV, FLAC, MP3, or OGG files")
    batch.add_argument("--recursive", action="store_true", help="Include supported audio in subfolders")
    batch.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    batch.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    evaluate = sub.add_parser("evaluate-asr", help="Evaluate local ASR against a ground-truth CSV manifest")
    evaluate.add_argument("manifest", help="UTF-8 CSV with sample_id, audio_path, and reference_text")
    evaluate.add_argument("--domain-lexicon", help="Optional YAML file containing a terms list")
    evaluate.add_argument("--limit", type=int, help="Evaluate only the first N manifest rows")
    evaluate.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    evaluate.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    llm = sub.add_parser("llm", help="Send a local message to the configured Ollama model")
    llm.add_argument("text", nargs="?", help="Message text")
    llm.add_argument("--text", dest="text_option", help="Compatible alternative to positional text")
    llm.add_argument("--system", help="Optional local system instruction")
    stream = sub.add_parser("stream-asr", help="Simulate local streaming ASR from a local audio file")
    stream.add_argument("audio", help="Path to a readable local audio file (WAV, FLAC, MP3, OGG)")
    stream.add_argument("--realtime", action="store_true", help="Feed chunks at their real duration")
    stream.add_argument(
        "--with-speakers",
        action="store_true",
        help="Fan out the stream to local streaming speaker diarization (Phase 5.5)",
    )
    stream.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    stream.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    chat = sub.add_parser("chat", help="Reserved for a future phase")
    chat.add_argument("--message", required=True)
    chat.add_argument("--system")
    jarvis = sub.add_parser(
        "jarvis", help="Run a Phase 8 JARVIS agent turn (offline default; --mode online for web)"
    )
    jarvis.add_argument("text", nargs="?", help="Command or question for JARVIS")
    jarvis.add_argument("--text", dest="text_option", help="Compatible alternative to positional text")
    jarvis.add_argument("--mode", choices=("offline", "online"), default="offline")
    jarvis.add_argument("--session", default="cli-default", help="Conversation session ID")
    jarvis.add_argument("--agent-config", help="Optional path to agent_config.yaml")
    sub.add_parser("doctor", help="Check Phase 1 environment and configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 on success, 2 for invalid configuration/input or unavailable features."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from thai_offline_ai.diagnostics import main as doctor_main

        return doctor_main(["--config", args.config] if args.config else [])
    try:
        settings = load_settings(args.config)
        if getattr(args, "model", None):
            settings.asr.model = args.model
        if getattr(args, "prompt", None):
            settings.asr.openthai_prompt = args.prompt
        if args.command == "chat":
            raise WorkbenchError(
                "The legacy chat command is unavailable. Use: llm \"your local message\". "
                "No model was loaded."
            )
        if args.command in {"nlp", "tts", "llm"} and (args.text is None) == (args.text_option is None):
            raise WorkbenchError('Provide text once: nlp "Thai text" or nlp --text "Thai text".')
        prepare_runtime(settings)
        if args.command in {"asr", "diarize", "transcribe-speakers"}:
            if (args.audio is None) == (args.audio_option is None):
                raise WorkbenchError(
                    f"Provide one audio path: {args.command} audio.wav or {args.command} --audio audio.wav."
                )
            audio_path = args.audio or args.audio_option
            if args.command == "asr":
                result = ASRAdapter(settings).transcribe_file(audio_path)
            elif args.command == "diarize":
                result = DiarizationAdapter(settings).diarize_file(audio_path)
            else:
                result = compose_speaker_transcript(
                    ASRAdapter(settings).transcribe_file(audio_path),
                    DiarizationAdapter(settings).diarize_file(audio_path),
                )
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "asr-batch":
            processor = BatchASRProcessor(settings)
            result = processor.transcribe_paths(
                processor.paths_in_directory(args.directory, recursive=args.recursive)
            )
            json_path, csv_path = export_batch_result(result, settings)
            print(
                json.dumps(
                    {**result.to_dict(), "exports": {"json": str(json_path), "csv": str(csv_path)}},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "evaluate-asr":
            if args.limit is not None and args.limit <= 0:
                raise WorkbenchError("--limit must be a positive integer.")
            records = load_manifest(args.manifest, limit=args.limit)
            report = ASREvaluator(settings).evaluate(
                records,
                dataset_id=Path(args.manifest).stem,
                domain_terms=load_domain_terms(args.domain_lexicon),
            )
            json_path, csv_path = export_evaluation_report(report, settings)
            print(
                json.dumps(
                    {**report.to_dict(), "exports": {"json": str(json_path), "csv": str(csv_path)}},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "llm":
            from thai_offline_ai.adapters.llm import OllamaLocalLLM

            prompt = args.text if args.text is not None else args.text_option
            messages = [
                ChatMessage("system", args.system or settings.llm.system_prompt),
                ChatMessage("user", prompt),
            ]
            result = OllamaLocalLLM.from_config(settings.llm, settings.app.offline_runtime).chat(messages)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "jarvis":
            if (args.text is None) == (args.text_option is None):
                raise WorkbenchError('Provide text once: jarvis "your command".')
            from thai_offline_ai.agent.config import load_agent_settings
            from thai_offline_ai.agent.orchestrator import AgentOrchestrator
            from thai_offline_ai.agent.schemas import OperatingMode

            prompt = args.text if args.text is not None else args.text_option
            agent_cfg = load_agent_settings(args.agent_config)
            orchestrator = AgentOrchestrator(settings, agent_cfg)
            try:
                response = orchestrator.run_turn(
                    args.session, prompt, mode=OperatingMode(args.mode)
                )
                print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
            finally:
                orchestrator.close_browser()
            return 0
        if args.command == "stream-asr":
            import time

            import soundfile as sf

            samples, sample_rate = sf.read(args.audio, always_2d=True, dtype="float32")
            if samples.size == 0 or sample_rate <= 0:
                raise WorkbenchError("Streaming input must be a non-empty readable audio file.")
            chunk_samples = max(1, round(sample_rate * settings.asr.stream_chunk_seconds))
            began = time.perf_counter()
            if args.with_speakers:
                speaker_service = StreamingSpeakerASRService(settings)
                session_id = speaker_service.start_session()
                try:
                    for offset in range(0, len(samples), chunk_samples):
                        chunk = samples[offset : offset + chunk_samples]
                        for event in speaker_service.push_audio(session_id, chunk, sample_rate):
                            print(json.dumps(event.to_dict(), ensure_ascii=False))
                        if args.realtime:
                            time.sleep(len(chunk) / sample_rate)
                    for event in speaker_service.finalize(session_id):
                        print(json.dumps(event.to_dict(), ensure_ascii=False))
                    print(
                        json.dumps(
                            {
                                "attributed_events": [
                                    ev.to_dict() for ev in speaker_service.get_events(session_id)
                                ],
                                "speaker_turns": [
                                    turn.to_dict() for turn in speaker_service.get_turns(session_id)
                                ],
                                "formatted_transcript": speaker_service.format_transcript(session_id),
                                "session_metrics": speaker_service.get_metrics(session_id).to_dict(),
                            },
                            ensure_ascii=False,
                        )
                    )
                finally:
                    speaker_service.close_session(session_id)
            else:
                service = StreamingASRService(settings)
                session_id = service.start_session()
                try:
                    for offset in range(0, len(samples), chunk_samples):
                        chunk = samples[offset : offset + chunk_samples]
                        for event in service.push_audio(session_id, chunk, sample_rate):
                            print(json.dumps(event.to_dict(), ensure_ascii=False))
                        if args.realtime:
                            time.sleep(len(chunk) / sample_rate)
                    for event in service.finalize(session_id):
                        print(json.dumps(event.to_dict(), ensure_ascii=False))
                    print(
                        json.dumps(
                            {"session_metrics": service.get_metrics(session_id).to_dict()},
                            ensure_ascii=False,
                        )
                    )
                finally:
                    service.close_session(session_id)
            print(json.dumps({"elapsed_seconds": round(time.perf_counter() - began, 3)}, ensure_ascii=False))
            return 0
        if args.command == "tts":
            tts_text = args.text if args.text is not None else args.text_option
            if args.model is None and args.voice is None and args.speed is None:
                result = TTSAdapter.from_settings(settings).synthesize(tts_text)
            else:
                result = TTSAdapter.from_settings(settings).synthesize(
                    tts_text, model=args.model, voice=args.voice, speed=args.speed
                )
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return 0
        result = NLPAdapter.from_config(settings.nlp, settings.app.offline_runtime).analyze(
            args.text if args.text is not None else args.text_option
        )
        # JSON retains the original machine-readable interface and readable Thai text.
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (WorkbenchError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
