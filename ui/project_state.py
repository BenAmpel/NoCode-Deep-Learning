from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = _PROJECT_ROOT / "outputs" / "project_state.json"


_LOCK = threading.Lock()


def save_project_state(state: dict[str, Any], *, replace: bool = False) -> None:
    """Merge ``state`` into the saved project state.

    Several callbacks save different keys (dataset, recommendation,
    preprocessing). Overwriting the whole file let one of them erase the
    others, and concurrent read-modify-write cycles could interleave, so
    saves merge under a lock and write atomically. ``replace=True`` writes
    ``state`` as the complete new state.
    """
    try:
        with _LOCK:
            merged = dict(state) if replace else {**load_project_state(), **state}
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE_FILE.with_name(f".{STATE_FILE.name}.tmp")
            tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            os.replace(tmp, STATE_FILE)
    except Exception as exc:
        logger.warning("Could not save project state: %s", exc)


def load_project_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load project state: %s", exc)
        return {}
