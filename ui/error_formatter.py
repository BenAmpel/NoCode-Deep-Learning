"""
Maps common Python/ML exceptions to friendly, student-facing markdown messages.

Usage
-----
    from ui.error_formatter import format_error, wrap_pipeline_error

    try:
        load_dataset(path)
    except Exception as exc:
        print(format_error(exc, context="data_loading"))

    @wrap_pipeline_error
    def run_training(...):
        ...
"""
from __future__ import annotations

import functools
import traceback
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

# ---------------------------------------------------------------------------
# Common fix suggestions keyed by error-message substrings.
# Used internally to enrich the generic fallback message.
# ---------------------------------------------------------------------------
COMMON_FIXES: dict[str, str] = {
    "cuda out of memory":   "Reduce **Batch size** (try halving it) or choose a smaller model.",
    "mps":                  "Reduce **Batch size** or switch device to CPU in config.",
    "memory":               "Try a smaller batch size, or load a smaller portion of your dataset.",
    "csv":                  "Open your CSV and verify the column names exactly match what you entered.",
    "column":               "Check your **Label column** and **Text column** fields match the CSV header row.",
    "permission":           "Run your terminal/IDE with sufficient permissions, or move data to a writable folder.",
    "ultralytics":          "Run `pip install ultralytics` in your terminal then restart the app.",
    "timm":                 "Run `pip install timm` in your terminal then restart the app.",
    "transformers":         "Run `pip install transformers` in your terminal then restart the app.",
    "no such file":         "Double-check the path and make sure the folder/file exists.",
    "zero":                 "Confirm your data path contains labelled samples before training.",
}


def _lookup_fix(message: str) -> str | None:
    """Return the first matching fix suggestion for *message*, or ``None``."""
    lower = message.lower()
    for keyword, fix in COMMON_FIXES.items():
        if keyword in lower:
            return fix
    return None


# ---------------------------------------------------------------------------
# Context-specific preambles
# ---------------------------------------------------------------------------
_CONTEXT_PREAMBLE: dict[str, str] = {
    "data_loading": "There was a problem **loading your data**.",
    "training":     "An error occurred **during training**.",
    "evaluation":   "An error occurred **during evaluation**.",
    "export":       "An error occurred **while exporting** your model.",
}


def format_error(exc: Exception, context: str = "") -> str:
    """
    Convert an exception into a friendly, markdown-formatted error message.

    Parameters
    ----------
    exc:
        The exception that was raised.
    context:
        Optional hint describing which pipeline stage failed.
        One of: ``"data_loading"``, ``"training"``, ``"evaluation"``, ``"export"``.

    Returns
    -------
    str
        A markdown string suitable for display in a Gradio ``gr.Markdown`` component.
        Always begins with ❌ and a bolded headline.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    exc_msg_lower = exc_msg.lower()

    preamble = _CONTEXT_PREAMBLE.get(context, "")

    # ------------------------------------------------------------------
    # Specific exception matching
    # ------------------------------------------------------------------

    # FileNotFoundError
    if isinstance(exc, FileNotFoundError):
        headline = "Dataset not found"
        detail = "Check the path is correct and the folder exists."
        return _build_message(headline, detail, preamble)

    # ValueError with CSV / column hints
    if isinstance(exc, ValueError) and (
        "csv" in exc_msg_lower or "column" in exc_msg_lower
    ):
        headline = "Column name mismatch"
        detail = (
            "Check your **Label** / **Text** column names match the CSV headers exactly "
            "(they are case-sensitive)."
        )
        return _build_message(headline, detail, preamble)

    # CUDA / MPS out-of-memory
    if isinstance(exc, RuntimeError) and (
        "cuda out of memory" in exc_msg_lower or "mps" in exc_msg_lower
    ):
        headline = "Out of GPU memory"
        detail = "Try reducing **Batch size** or switching to a smaller model."
        return _build_message(headline, detail, preamble)

    # ImportError — ultralytics
    if isinstance(exc, ImportError) and "ultralytics" in exc_msg_lower:
        headline = "ultralytics not installed"
        detail = "Run: `pip install ultralytics`"
        return _build_message(headline, detail, preamble)

    # ImportError — timm
    if isinstance(exc, ImportError) and "timm" in exc_msg_lower:
        headline = "timm not installed"
        detail = "Run: `pip install timm`"
        return _build_message(headline, detail, preamble)

    # ImportError — transformers
    if isinstance(exc, ImportError) and "transformers" in exc_msg_lower:
        headline = "transformers not installed"
        detail = "Run: `pip install transformers`"
        return _build_message(headline, detail, preamble)

    # PermissionError
    if isinstance(exc, PermissionError):
        headline = "Permission denied"
        detail = "Make sure you have **read access** to the data folder."
        return _build_message(headline, detail, preamble)

    # MemoryError or RuntimeError mentioning memory
    if isinstance(exc, MemoryError) or (
        isinstance(exc, RuntimeError) and "memory" in exc_msg_lower
    ):
        headline = "Not enough RAM"
        detail = "Try a **smaller batch size** or dataset."
        return _build_message(headline, detail, preamble)

    # KeyError — missing field
    if isinstance(exc, KeyError):
        headline = f"Missing data field: {exc}"
        detail = "Check your CSV column names."
        return _build_message(headline, detail, preamble)

    # ZeroDivisionError or empty-dataset signals
    if isinstance(exc, ZeroDivisionError) or (
        isinstance(exc, (ValueError, RuntimeError))
        and any(kw in exc_msg_lower for kw in ("empty", "zero sample", "no sample", "0 sample"))
    ):
        headline = "Dataset appears empty"
        detail = "Make sure your data path contains valid files."
        return _build_message(headline, detail, preamble)

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------
    extra_fix = _lookup_fix(exc_msg)
    fix_line = f"\n\n> **Suggestion:** {extra_fix}" if extra_fix else ""

    lines = [
        f"❌ **Unexpected error: `{exc_type}`**",
    ]
    if preamble:
        lines.insert(0, preamble + "\n")
    lines += [
        "",
        f"> {exc_msg}",
        fix_line,
        "",
        "_See the console for the full traceback._",
    ]
    return "\n".join(lines)


def _build_message(headline: str, detail: str, preamble: str = "") -> str:
    """Assemble the standard two-line friendly error card."""
    parts: list[str] = []
    if preamble:
        parts.append(preamble)
        parts.append("")
    parts.append(f"❌ **{headline}** — {detail}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def wrap_pipeline_error(func: F) -> F:
    """
    Decorator that wraps a pipeline stage function.

    Catches all exceptions, formats them with :func:`format_error`, logs the
    original traceback to stderr, and re-raises a :class:`RuntimeError` whose
    message is the friendly markdown string.

    Parameters
    ----------
    func:
        The function to wrap.  The ``context`` hint is inferred from the
        function name if it contains one of the recognised keywords.

    Returns
    -------
    Callable
        The wrapped function with identical signature.

    Example
    -------
    ::

        @wrap_pipeline_error
        def run_training(cfg):
            ...
    """
    _CONTEXT_KEYWORDS = {
        "data":     "data_loading",
        "load":     "data_loading",
        "train":    "training",
        "fit":      "training",
        "eval":     "evaluation",
        "metric":   "evaluation",
        "export":   "export",
        "save":     "export",
    }

    # Infer context from function name
    fn_name_lower = func.__name__.lower()
    inferred_context = ""
    for keyword, ctx in _CONTEXT_KEYWORDS.items():
        if keyword in fn_name_lower:
            inferred_context = ctx
            break

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # Print full traceback to console for debugging
            traceback.print_exc()
            friendly = format_error(exc, context=inferred_context)
            raise RuntimeError(friendly) from exc

    return wrapper  # type: ignore[return-value]
