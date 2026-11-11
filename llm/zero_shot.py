"""Zero-shot text classification with a language model.

Labels a folder of ``.txt`` files and returns the same columns as folder
batch prediction (``file``, ``prediction``, ``confidence``,
``confidence_score``), so the saved CSV works with the same scripts. A
language model gives no calibrated probability, so ``confidence`` is blank.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from llm.client import LLMConfig, LLMError, Transport, chat

MAX_CHARS = 6000
MAX_FILES = 5000
TEXT_EXTS = {".txt"}
COLUMNS = ["file", "prediction", "confidence", "confidence_score"]


def parse_labels(text: str) -> list[str]:
    """Labels from a comma- or newline-separated string, deduplicated in order."""
    parts = re.split(r"[,\n]", text or "")
    seen: dict[str, str] = {}
    for p in parts:
        label = p.strip().strip("\"'")
        if label and label.lower() not in seen:
            seen[label.lower()] = label
    return list(seen.values())


def build_messages(text: str, labels: list[str], instructions: str = "") -> list[dict]:
    text = (text or "").strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " …[truncated]"
    system = (
        "You are a text classifier. Read the text and answer with exactly one label from this list: "
        + ", ".join(labels)
        + ". Reply with the label only, no explanation."
    )
    if instructions and instructions.strip():
        system += "\n\nLabel definitions and rules:\n" + instructions.strip()
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Text:\n\"\"\"\n{text}\n\"\"\"\n\nLabel:"}]


def parse_label(reply: str, labels: list[str]) -> str | None:
    """Map a model reply to one of ``labels``, or None if it is ambiguous."""
    if not reply:
        return None
    s = reply.strip()
    # {"label": "..."} replies
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for key in ("label", "category", "class", "answer"):
                    if key in obj:
                        s = str(obj[key])
                        break
        except ValueError:
            pass
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I).strip()
    cleaned = re.sub(r"^(label|answer|category|class)\s*[:\-]\s*", "", s, flags=re.I)
    cleaned = cleaned.strip().strip("`*\"'.:;!").strip()
    by_lower = {lab.lower(): lab for lab in labels}
    if cleaned.lower() in by_lower:
        return by_lower[cleaned.lower()]
    first_line = cleaned.splitlines()[0].strip().strip("`*\"'.:;!") if cleaned else ""
    if first_line.lower() in by_lower:
        return by_lower[first_line.lower()]
    # Exactly one label mentioned as a whole word. Longer labels first, so
    # "very positive" wins over "positive" when both match the same span.
    found = []
    taken: list[tuple[int, int]] = []
    for lab in sorted(labels, key=len, reverse=True):
        for m in re.finditer(r"(?<![\w-])" + re.escape(lab) + r"(?![\w-])", s, flags=re.I):
            if not any(a < m.end() and m.start() < b for a, b in taken):
                taken.append((m.start(), m.end()))
                if lab not in found:
                    found.append(lab)
    return found[0] if len(found) == 1 else None


def list_text_files(folder: str | Path) -> list[Path]:
    folder = Path(str(folder).strip()) if folder else None
    if not folder or not folder.is_dir():
        raise LLMError(f"Folder not found: `{folder}`")
    files = sorted(f for f in folder.rglob("*")
                   if f.is_file() and f.suffix.lower() in TEXT_EXTS and not f.name.startswith("."))
    if not files:
        raise LLMError(f"No .txt files found in `{folder}`.")
    if len(files) > MAX_FILES:
        raise LLMError(f"{len(files):,} files found; the limit is {MAX_FILES:,} per run. Split the folder.")
    return files


def classify_folder(folder, labels: list[str], cfg: LLMConfig, instructions: str = "",
                    *, transport: Transport | None = None,
                    progress: Callable[[int, int], None] | None = None,
                    sleep: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Label every ``.txt`` file. Per-file failures become ``ERROR`` rows."""
    if len(labels) < 2:
        raise LLMError("Enter at least two labels, separated by commas.")
    cfg.validate()
    files = list_text_files(folder)
    rows = []
    kwargs = {"transport": transport}
    if sleep is not None:
        kwargs["sleep"] = sleep
    consecutive_conn_errors = 0
    for i, path in enumerate(files):
        if progress:
            progress(i, len(files))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            rows.append([path.name, "ERROR", f"Could not read file: {exc}", None])
            continue
        if not text.strip():
            rows.append([path.name, "ERROR", "Empty file", None])
            continue
        try:
            reply = chat(cfg, build_messages(text, labels, instructions), **kwargs)
            consecutive_conn_errors = 0
        except LLMError as exc:
            msg = cfg.scrub(str(exc))
            rows.append([path.name, "ERROR", msg, None])
            # Stop early when the server is clearly down or the key is wrong,
            # instead of failing every file one timeout at a time.
            if "Cannot reach" in msg or "API key" in msg:
                consecutive_conn_errors += 1
                if consecutive_conn_errors >= 3:
                    for rest in files[i + 1:]:
                        rows.append([rest.name, "ERROR", "Skipped: " + msg, None])
                    break
            continue
        label = parse_label(reply, labels)
        if label is None:
            rows.append([path.name, "ERROR", f"Unclear reply: {reply[:120]!r}", None])
        else:
            rows.append([path.name, label, "", None])
    if progress:
        progress(len(files), len(files))
    return pd.DataFrame(rows, columns=COLUMNS)
