"""Windows / cross-platform entry point.

Mirrors ``scripts/run_all.sh`` for operators on machines without bash.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.runner import run   # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run())
