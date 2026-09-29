#!/usr/bin/env python3
"""Entry point chạy từ repo root: python ask.py "Bệnh Addison là gì?" """
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from medical_agentic_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
