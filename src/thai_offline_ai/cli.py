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
from thai_offline_ai.benchmark import ASRBenchmark, export_benchmark_result
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
    benchmark = sub.add_parser(
        "benchmark-asr", help="Measure local ASR elapsed time and real-time factor (RTF)"
    )
    benchmark.add_argument("audio", nargs="+", help="One or more local WAV, FLAC, MP3, or OGG files")
    benchmark.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    benchmark.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
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
    va = sub.add_parser(
        "voice-assistant",
        help="Run Phase 6 offline Local Voice Assistant (audio -> ASR -> Local LLM -> TTS)",
    )
    va.add_argument("audio", nargs="?", help="Path to a local audio file")
    va.add_argument("--audio", dest="audio_option", help="Compatible alternative to positional audio")
    va.add_argument("--system", help="Optional local system instruction")
    va.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    va.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    va.add_argument("--tts-model", help="TTS model: kokoro_thai_onnx or lunarlist_onnx")
    va.add_argument("--voice", help="Kokoro Thai voice ID")
    va.add_argument("--speed", type=float, help="TTS speech speed multiplier")
    va.add_argument("--no-tts", action="store_true", help="Skip spoken TTS synthesis")
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
    lab_syn = sub.add_parser(
        "lab-generate-synthetic",
        help="Phase 6.5: generate local synthetic speech + augmentations + train_synthetic manifest",
    )
    lab_syn.add_argument("corpus", help="UTF-8 text file (one utterance per line)")
    lab_syn.add_argument("--dataset-id", default="synthetic_thai", help="Synthetic dataset identifier")
    lab_syn.add_argument("--version", default="v1", help="Dataset version tag")
    lab_syn.add_argument(
        "--profiles",
        default="clean,telephone_bandpass",
        help="Comma-separated augmentation presets: clean, gaussian_noise, telephone_bandpass, fast_quiet",
    )
    lab_syn.add_argument("--tts-model", help="TTS model: kokoro_thai_onnx or lunarlist_onnx")
    lab_syn.add_argument("--voice", help="Optional Kokoro voice ID")
    lab_syn.add_argument("--speed", type=float, help="Optional speech speed")
    lab_pl = sub.add_parser(
        "lab-pseudo-label",
        help="Phase 6.5: run fixed teacher ASR on unlabeled_real folder with deterministic filters",
    )
    lab_pl.add_argument("directory", help="Local folder containing unlabeled real audio files")
    lab_pl.add_argument("--dataset-id", default="unlabeled_thai", help="Dataset identifier")
    lab_pl.add_argument("--version", default="v1", help="Dataset version tag")
    lab_pl.add_argument("--recursive", action="store_true")
    lab_pl.add_argument("--model", help="Teacher ASR model: typhoon_asr or openthai2_asr")
    lab_pl.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    lab_eval = sub.add_parser(
        "lab-evaluate-promote",
        help="Phase 6.5: evaluate ASR on real holdout CSV, log experiment, and run promotion gate",
    )
    lab_eval.add_argument("manifest", help="Immutable test_real/validation_real UTF-8 CSV manifest")
    lab_eval.add_argument("--eval-version", default="v1", help="Holdout dataset version/fingerprint")
    lab_eval.add_argument("--eval-split", choices=("test_real", "validation_real"), default="test_real")
    lab_eval.add_argument("--base-model", default="openthai2_asr")
    lab_eval.add_argument("--candidate-model", default="openthai2_asr")
    lab_eval.add_argument("--baseline-id", help="Existing baseline experiment_id to compare against")
    lab_eval.add_argument("--register-baseline", action="store_true", help="Mark this run as baseline")
    lab_eval.add_argument("--domain-lexicon", help="Optional YAML domain terms list")
    lab_eval.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    lab_eval.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    ci = sub.add_parser(
        "conversation-intel",
        help="Phase 7: summarize and extract grounded action items from a speaker transcript",
    )
    ci.add_argument("audio", nargs="?", help="Audio file: runs local ASR + diarization first")
    ci.add_argument("--transcript-json", help="Output JSON of transcribe-speakers (segments)")
    ci.add_argument("--text-file", help="UTF-8 text with 'SPEAKER_00: ...' lines")
    ci.add_argument("--domain-lexicon", help="Optional YAML file containing a terms list")
    ci.add_argument("--no-llm", action="store_true", help="Deterministic analytics only")
    ci.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    ci.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    mm = sub.add_parser("models", help="Inspect or explicitly install local model caches")
    mm.add_argument("--install", choices=(
        "kokoro_thai_onnx",
        "lunarlist_onnx",
        "openthai2_asr",
        "typhoon_asr",
        "nemotron_diarization",
        "playwright_chromium",
    ), help="Model key to install via its verified setup script")
    mm.add_argument("--setup-mode", action="store_true", help="Explicitly authorize setup-time download")
    rag = sub.add_parser("rag", help="Ingest or query local documents in the offline Local RAG store")
    rag.add_argument("--ingest", nargs="+", help="One or more local document paths to index")
    rag.add_argument("--search", help="Search query to retrieve top chunks without LLM")
    rag.add_argument("--ask", help="Question to answer from local documents")
    rag.add_argument("--no-llm", action="store_true", help="Use extractive fallback instead of Ollama")
    rag.add_argument("--top-k", type=int, default=4, help="Maximum chunks to retrieve")
    sub.add_parser("package-portable", help="Build a portable Windows desktop launcher bundle")
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
        if args.command == "benchmark-asr":
            result = ASRBenchmark(settings).run(
                args.audio, model=args.model, prompt_mode=args.prompt
            )
            json_path, csv_path = export_benchmark_result(result, settings)
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
        if args.command == "voice-assistant":
            if (args.audio is None) == (args.audio_option is None):
                raise WorkbenchError(
                    "Provide one audio path: voice-assistant audio.wav or voice-assistant --audio audio.wav."
                )
            from thai_offline_ai.voice_assistant import LocalVoiceAssistantService

            audio_path = args.audio or args.audio_option
            turn = LocalVoiceAssistantService(settings).process_audio_turn(
                "cli-voice",
                audio_path,
                system_prompt=args.system,
                speak_response=not args.no_tts,
                asr_model=args.model,
                asr_prompt=args.prompt,
                tts_model=args.tts_model,
                tts_voice=args.voice,
                tts_speed=args.speed,
            )
            print(json.dumps(turn.to_dict(), ensure_ascii=False, indent=2))
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
        if args.command == "conversation-intel":
            from thai_offline_ai.conversation_intel import (
                ConversationIntelligenceService,
                export_report,
                load_utterances_from_json,
                parse_speaker_text,
                utterances_from_transcript,
            )

            sources = [args.audio, args.transcript_json, args.text_file]
            if sum(s is not None for s in sources) != 1:
                raise WorkbenchError(
                    "Provide exactly one source: audio path, --transcript-json, or --text-file."
                )
            if args.transcript_json:
                utterances = load_utterances_from_json(args.transcript_json)
            elif args.text_file:
                text_path = Path(args.text_file)
                if not text_path.is_file():
                    raise WorkbenchError(f"Text file not found: {text_path}")
                utterances = parse_speaker_text(text_path.read_text(encoding="utf-8-sig"))
            else:
                utterances = utterances_from_transcript(
                    compose_speaker_transcript(
                        ASRAdapter(settings).transcribe_file(args.audio),
                        DiarizationAdapter(settings).diarize_file(args.audio),
                    )
                )
            report = ConversationIntelligenceService(settings).analyze(
                utterances,
                domain_terms=load_domain_terms(args.domain_lexicon),
                use_llm=not args.no_llm,
            )
            json_path, md_path = export_report(report, settings)
            print(
                json.dumps(
                    {**report.to_dict(), "exports": {"json": str(json_path), "markdown": str(md_path)}},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "lab-generate-synthetic":
            from thai_offline_ai.speech_lab.synthetic import (
                AUGMENTATION_PRESETS,
                SyntheticCorpusItem,
                SyntheticSpeechGenerator,
            )

            corpus_path = Path(args.corpus)
            if not corpus_path.is_file():
                raise WorkbenchError(f"Corpus file not found: {corpus_path}")
            lines = [
                line.strip()
                for line in corpus_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            items = [
                SyntheticCorpusItem(
                    sample_id=f"{args.dataset_id}_{idx:04d}",
                    text=line,
                    voice=args.voice,
                    speed=args.speed,
                )
                for idx, line in enumerate(lines, start=1)
            ]
            preset_names = [p.strip() for p in args.profiles.split(",") if p.strip()]
            augs = []
            for name in preset_names:
                if name not in AUGMENTATION_PRESETS:
                    raise WorkbenchError(
                        f"Unknown augmentation profile '{name}'. Choices: {', '.join(AUGMENTATION_PRESETS)}."
                    )
                augs.append(AUGMENTATION_PRESETS[name])
            out_dir = Path(settings.app.output_dir) / "speech_lab" / "synthetic" / args.dataset_id
            summary = SyntheticSpeechGenerator(settings).generate_dataset(
                items,
                dataset_id=args.dataset_id,
                dataset_version=args.version,
                output_dir=out_dir,
                augmentations=augs,
                tts_model=args.tts_model,
            )
            print(
                json.dumps(
                    {
                        "dataset_id": summary.manifest.dataset_id,
                        "version": summary.manifest.version,
                        "fingerprint": summary.manifest.fingerprint,
                        "generated_count": summary.generated_count,
                        "failed_count": len(summary.failed_items),
                        "manifest_path": str(summary.manifest_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "lab-pseudo-label":
            from thai_offline_ai.speech_lab.pseudo_label import PseudoLabelingPipeline

            processor = BatchASRProcessor(settings)
            paths = processor.paths_in_directory(args.directory, recursive=args.recursive)
            out_dir = Path(settings.app.output_dir) / "speech_lab" / "pseudo_labels" / args.dataset_id
            report = PseudoLabelingPipeline(settings).run(
                paths,
                dataset_id=args.dataset_id,
                dataset_version=args.version,
                output_dir=out_dir,
                teacher_model=args.model,
                prompt_mode=args.prompt,
            )
            print(json.dumps(report.to_summary_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "lab-evaluate-promote":
            from thai_offline_ai.speech_lab.experiments import ExperimentRegistry

            records = load_manifest(args.manifest)
            eval_report = ASREvaluator(settings).evaluate(
                records,
                dataset_id=Path(args.manifest).stem,
                domain_terms=load_domain_terms(args.domain_lexicon),
            )
            registry = ExperimentRegistry(
                Path(settings.app.output_dir) / "speech_lab" / "experiments.db"
            )
            exp = registry.record_from_evaluation(
                eval_report,
                base_model=args.base_model,
                candidate_model=args.candidate_model,
                eval_dataset_version=args.eval_version,
                eval_split=args.eval_split,
                is_baseline=args.register_baseline,
            )
            output: dict[str, object] = {"experiment": exp.to_dict()}
            if args.baseline_id:
                decision = registry.promote_or_reject(args.baseline_id, exp.experiment_id)
                output["promotion_decision"] = decision.to_dict()
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        if args.command == "models":
            from thai_offline_ai.model_manager import ModelManager

            mgr = ModelManager(settings)
            if args.install:
                res = mgr.run_setup(args.install, allow_setup_network=bool(args.setup_mode))
                print(json.dumps(res, ensure_ascii=False, indent=2))
            else:
                print(
                    json.dumps(
                        {"models": [m.to_dict() for m in mgr.list_statuses()]},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            return 0
        if args.command == "rag":
            from thai_offline_ai.local_rag import LocalRAGStore

            store = LocalRAGStore(Path(settings.app.output_dir) / "local_rag.db", settings=settings)
            payload: dict[str, object] = {}
            if args.ingest:
                docs = store.ingest_files(args.ingest)
                payload["ingested"] = [d.to_dict() for d in docs]
            if args.search:
                matches = store.search(args.search, top_k=args.top_k)
                payload["matches"] = [m.to_dict() for m in matches]
            if args.ask:
                ans = store.ask(args.ask, top_k=args.top_k, use_llm=not args.no_llm)
                payload["answer"] = ans.to_dict()
            if not payload:
                payload["documents"] = [d.to_dict() for d in store.list_documents()]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "package-portable":
            from thai_offline_ai.packaging import build_portable_bundle

            bundle = build_portable_bundle(settings)
            print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
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
