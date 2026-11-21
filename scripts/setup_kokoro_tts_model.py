"""Explicitly download the Apache-2.0 Kokoro Thai ONNX bundle (12 voices, 24 kHz).

Only this setup script accesses the network; normal app startup, TTS synthesis,
doctor, and unit tests remain strictly cache-only and offline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from thai_offline_ai.config import load_settings

REPO_ID = "kunato/wayu-kokoro-thai-v1"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

ONNX_FILES = (
    "onnx/onnx_manifest.json",
    "onnx/prosody_fp32.onnx",
    "onnx/curves_fp32.onnx",
    "onnx/decoder_fp32.onnx",
    "onnx/source_params.npz",
    "onnx/styles.npz",
    "onnx/voicepacks.npz",
    "data/seeds/roster.json",
)


def _download_file(url: str, destination: Path) -> None:
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with (
        httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or "0")
        downloaded = 0
        next_report = 25 * 1024 * 1024
        with temp_path.open("wb") as out:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                downloaded += len(chunk)
                if total >= 10 * 1024 * 1024 and (downloaded >= next_report or downloaded >= total):
                    print(
                        f"  {destination.name}: {downloaded / 1048576:.1f} / {total / 1048576:.1f} MB",
                        flush=True,
                    )
                    next_report += 25 * 1024 * 1024
    temp_path.replace(destination)


def download_kokoro_thai(target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for remote_path in ONNX_FILES:
        dest = target_dir / Path(remote_path).name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"Already present: {dest.name} ({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
            saved.append(dest)
            continue
        print(f"Downloading {REPO_ID}/{remote_path} -> {dest} ...", flush=True)
        _download_file(f"{BASE_URL}/{remote_path}", dest)
        saved.append(dest)
    missing = [p.name for p in saved if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing Kokoro Thai ONNX files: {', '.join(missing)}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML configuration path")
    parser.add_argument("--output-dir", type=Path, help="Override destination model directory")
    args = parser.parse_args()
    settings = load_settings(args.config)
    target_dir = args.output_dir or Path(
        getattr(settings.tts, "kokoro_model_dir", "models/kokoro-thai-onnx")
    )
    files = download_kokoro_thai(target_dir)
    print("Kokoro Thai ONNX (12 voices, 24 kHz) ready:", flush=True)
    for path in files:
        print(f"  - {path} ({path.stat().st_size / 1048576:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
