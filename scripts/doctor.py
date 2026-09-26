"""Run from the project root after installing the project into the active venv."""

import sys


def main() -> int:
    try:
        from thai_offline_ai.diagnostics import main as doctor_main
    except ImportError:
        print("[FAIL] Project import failed. Activate .venv and run: python -m pip install -e .")
        return 1
    return doctor_main()


if __name__ == "__main__":
    sys.exit(main())
