"""Explicitly download OpenThai2.0-ASR-Streaming ONNX files from Hugging Face.

Only this setup script accesses the network; the workbench app, CLI inference,
doctor, and tests remain cache-only and never download models.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from thai_offline_ai.config import load_settings

REPO_ID = "iapp/OpenThai2.0-ASR-Streaming-ONNX"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

PRECISION_ARTIFACTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "int8": (
        "encoder.int8.onnx.data",
        "openthai2.0-asr-streaming-int8.asr.zip",
        ("encoder.int8.onnx", "encoder.int8.onnx.data", "decoder_joint.onnx", "tokens.txt"),
    ),
    "fp32": (
        "encoder.onnx.data",
        "openthai2.0-asr-streaming-fp32.asr.zip",
        ("encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokens.txt"),
    ),
}


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    resolved_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            dest = (resolved_root / Path(member.filename).name).resolve()
            if dest.parent != resolved_root:
                raise ValueError(f"Unsafe archive entry: {member.filename}")
            with archive.open(member, "r") as src, dest.open("wb") as out:
                while chunk := src.read(1024 * 1024):
                    out.write(chunk)


def _download_file(url: str, destination: Path) -> None:
    import httpx

    temp_path = destination.with_suffix(destination.suffix + ".part")
    with (
        httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or "0")
        downloaded = 0
        next_report = 50 * 1024 * 1024
        with temp_path.open("wb") as out:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report or (total and downloaded >= total):
                    mb = downloaded / (1024 * 1024)
                    if total:
                        total_mb = total / (1024 * 1024)
                        print(f"  {destination.name}: {mb:.1f} / {total_mb:.1f} MB", flush=True)
                    else:
                        print(f"  {destination.name}: {mb:.1f} MB", flush=True)
                    next_report += 50 * 1024 * 1024
    temp_path.replace(destination)


def download_openthai_onnx(target_dir: Path, precision: str = "int8") -> list[Path]:
    if precision not in PRECISION_ARTIFACTS:
        raise ValueError(f"Unsupported precision: {precision}. Use int8 or fp32.")

    data_file, zip_file, required_files = PRECISION_ARTIFACTS[precision]
    target_dir.mkdir(parents=True, exist_ok=True)
    # Remove any old huggingface cache subdir if present from earlier attempts.
    cache_dir = target_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    for filename in (data_file, zip_file):
        dest = target_dir / filename
        if filename == data_file and dest.is_file() and dest.stat().st_size > 0:
            print(f"Already present: {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)
            continue
        print(f"Downloading {REPO_ID}/{filename} -> {target_dir} ...", flush=True)
        _download_file(f"{BASE_URL}/{filename}", dest)

    zip_path = target_dir / zip_file
    print(f"Extracting {zip_path.name} ...", flush=True)
    _safe_extract_zip(zip_path, target_dir)
    zip_path.unlink(missing_ok=True)
    verified = [target_dir / name for name in required_files]
    missing = [path.name for path in verified if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing expected OpenThai2.0 ONNX files: {', '.join(missing)}")
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML configuration path")
    parser.add_argument(
        "--precision",
        choices=("int8", "fp32"),
        default="int8",
        help="ONNX precision: int8 (~0.99 GB, CPU-friendly default) or fp32 (~2.60 GB)",
    )
    parser.add_argument("--output-dir", type=Path, help="Override destination model directory")
    args = parser.parse_args()
    settings = load_settings(args.config)
    target_dir = args.output_dir or Path(
        getattr(settings.asr, "openthai_model_dir", "models/openthai2-asr-streaming")
    )
    files = download_openthai_onnx(target_dir, precision=args.precision)
    print("OpenThai2.0-ASR-Streaming ONNX ready:", flush=True)
    for path in files:
        print(f"  - {path} ({path.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
