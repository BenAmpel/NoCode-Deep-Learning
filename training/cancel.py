"""Cooperative stop for long training jobs.

Gradio's ``cancels=`` only takes effect at the job's next ``yield``, so a long
epoch, cross-validation fold or sweep kept running after the UI said it had
stopped. Training loops call :func:`check` between batches instead.

Heavy jobs share one concurrency lane (one at a time), so a single
process-wide flag is enough: Stop sets it, each job clears it when it starts.
"""
from __future__ import annotations

import threading

_stop = threading.Event()


class TrainingStopped(Exception):
    """Raised inside a training loop after the user pressed Stop."""


def request_stop() -> None:
    _stop.set()


def reset() -> None:
    _stop.clear()


def stop_requested() -> bool:
    return _stop.is_set()


def check() -> None:
    if _stop.is_set():
        raise TrainingStopped("Stopped by user.")
