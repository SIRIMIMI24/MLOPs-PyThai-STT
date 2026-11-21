"""Feed a local audio file into the streaming ASR (and optional diarization) service."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from thai_offline_ai.config import load_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_runtime
from thai_offline_ai.streaming import StreamingASRService
from thai_offline_ai.streaming_diarization import StreamingSpeakerASRService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--with-speakers",
        action="store_true",
        help="Also run local streaming speaker diarization on the same stream (Phase 5.5)",
    )
    parser.add_argument("--model", help="ASR model: typhoon_asr or openthai2_asr")
    parser.add_argument("--prompt", choices=("th-TH", "th-mix"), help="OpenThai2.0 prompt mode")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if args.model:
            settings.asr.model = args.model
        if args.prompt:
            settings.asr.openthai_prompt = args.prompt
        prepare_runtime(settings)
        samples, sample_rate = sf.read(args.audio, always_2d=True, dtype="float32")
        if samples.size == 0:
            raise WorkbenchError("Streaming input must be a non-empty readable audio file.")
        chunk_samples = max(1, round(sample_rate * settings.asr.stream_chunk_seconds))
        started = time.perf_counter()
        if args.with_speakers:
            service = StreamingSpeakerASRService(settings)
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
                        {
                            "attributed_events": [
                                ev.to_dict() for ev in service.get_events(session_id)
                            ],
                            "speaker_turns": [
                                turn.to_dict() for turn in service.get_turns(session_id)
                            ],
                            "formatted_transcript": service.format_transcript(session_id),
                            "session_metrics": service.get_metrics(session_id).to_dict(),
                        },
                        ensure_ascii=False,
                    )
                )
            finally:
                service.close_session(session_id)
        else:
            asr_service = StreamingASRService(settings)
            session_id = asr_service.start_session()
            try:
                for offset in range(0, len(samples), chunk_samples):
                    chunk = samples[offset : offset + chunk_samples]
                    for event in asr_service.push_audio(session_id, chunk, sample_rate):
                        print(json.dumps(event.to_dict(), ensure_ascii=False))
                    if args.realtime:
                        time.sleep(len(chunk) / sample_rate)
                for event in asr_service.finalize(session_id):
                    print(json.dumps(event.to_dict(), ensure_ascii=False))
                print(
                    json.dumps(
                        {"session_metrics": asr_service.get_metrics(session_id).to_dict()},
                        ensure_ascii=False,
                    )
                )
            finally:
                asr_service.close_session(session_id)
        print(json.dumps({"elapsed_seconds": round(time.perf_counter() - started, 3)}))
        return 0
    except (WorkbenchError, OSError, RuntimeError) as exc:
        print(f"[FAIL] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
