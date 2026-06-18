from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def configure_runtime() -> None:
    for path in (REPO_ROOT, REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
