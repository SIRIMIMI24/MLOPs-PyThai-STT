from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

from thai_offline_ai.adapters.nlp import NLPAdapter
from thai_offline_ai.adapters.tts import TTSAdapter
from thai_offline_ai.config import load_settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.runtime import prepare_runtime


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
    asr = sub.add_parser("asr", help="Reserved for a future phase")
    asr.add_argument("--audio", required=True)
    chat = sub.add_parser("chat", help="Reserved for a future phase")
    chat.add_argument("--message", required=True)
    chat.add_argument("--system")
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
        if args.command in {"asr", "chat"}:
            raise WorkbenchError(
                f"{args.command} is not configured in Phase 1; inference is reserved for a future phase. "
                "Use the nlp command. No model was loaded."
            )
        if args.command in {"nlp", "tts"} and (args.text is None) == (args.text_option is None):
            raise WorkbenchError('Provide text once: nlp "Thai text" or nlp --text "Thai text".')
        prepare_runtime(settings)
        if args.command == "tts":
            result = TTSAdapter.from_settings(settings).synthesize(
                args.text if args.text is not None else args.text_option
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
