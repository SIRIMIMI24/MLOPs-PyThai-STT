"""Explicit setup helper for Phase 8.1 Local Browser Agent (Playwright + Chromium).

Run deliberately during setup when internet access is available:
    python scripts/setup_playwright.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    print("[Phase 8.1 Setup] Installing playwright package into current Python environment...")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "playwright>=1.50"])
    if rc != 0:
        print("[FAIL] Could not install playwright package.", file=sys.stderr)
        return rc
    print("[Phase 8.1 Setup] Installing Playwright Chromium browser binary...")
    rc = subprocess.call([sys.executable, "-m", "playwright", "install", "chromium"])
    if rc != 0:
        print("[FAIL] Could not download Chromium binary for Playwright.", file=sys.stderr)
        return rc
    print("[OK] Playwright + Chromium installed. You can now use Online Agent Mode in JARVIS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
