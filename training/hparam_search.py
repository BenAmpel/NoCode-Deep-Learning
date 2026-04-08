"""
Hyperparameter search utilities — random search and grid search — built on
top of the existing ``train_pytorch`` generator.

Both ``random_search`` and ``grid_search`` are generator functions that yield
incremental progress dicts so they can be streamed to a UI.
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Generator

import torch.nn as nn

from training.trainer import train_pytorch

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_param_combinations(
    param_space: dict[str, list[Any]],
    n_trials: int,
) -> list[dict[str, Any]]:
    """Return up to *n_trials* unique random combinations from *param_space*.

    If the total grid is smaller than *n_trials* the full grid is returned
    (without replacement by construction).
    """
    keys = list(param_space.keys())
    all_combos = list(itertools.product(*[param_space[k] for k in keys]))
    random.shuffle(all_combos)
    selected = all_combos[:n_trials]
    return [{keys[i]: combo[i] for i in range(len(keys))} for combo in selected]


def _grid_param_combinations(
    param_space: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Return every combination in *param_space* (full Cartesian product)."""
    keys = list(param_space.keys())
    all_combos = list(itertools.product(*[param_space[k] for k in keys]))
    return [{keys[i]: combo[i] for i in range(len(keys))} for combo in all_combos]


def _run_trial(
    model_factory: Callable[[dict[str, Any]], nn.Module],
    train_loader,
    val_loader,
    params: dict[str, Any],
    epochs_per_trial: int,
    task: str,
) -> dict[str, Any]:
    """Train one trial to completion and return the final result dict."""
    model = model_factory(params)

    lr             = float(params.get("lr", 1e-3))
    optimizer_name = str(params.get("optimizer", "adam"))
    scheduler_name = str(params.get("scheduler", "cosine"))
    use_amp        = bool(params.get("use_amp", False))
    patience       = int(params.get("patience", epochs_per_trial))  # no early-stop by default

    final: dict[str, Any] = {}
    for update in train_pytorch(
        model,
        train_loader,
        val_loader,
        epochs=epochs_per_trial,
        lr=lr,
        optimizer_name=optimizer_name,
        patience=patience,
        scheduler_name=scheduler_name,
        use_amp=use_amp,
        task=task,
    ):
        if update.get("done"):
            final = update
    return final


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def random_search(
    model_factory: Callable[[dict[str, Any]], nn.Module],
    train_loader,
    val_loader,
    param_space: dict[str, list[Any]],
    n_trials: int = 10,
    epochs_per_trial: int = 5,
    task: str = "classification",
) -> Generator[dict[str, Any], None, None]:
    """Run random hyperparameter search over *param_space*.

    Parameters
    ----------
    model_factory:
        Callable that accepts a ``params`` dict and returns an ``nn.Module``.
        The dict contains whichever keys are in *param_space* (e.g. ``lr``,
        ``dropout``).  Any non-training keys (dropout, hidden_size, …) are
        forwarded to the factory so it can build the model accordingly; only
        the training-specific keys (``lr``, ``optimizer``, ``scheduler``,
        ``patience``, ``use_amp``) are forwarded to ``train_pytorch``.
    train_loader / val_loader:
        Standard PyTorch ``DataLoader`` objects reused across every trial.
    param_space:
        Mapping of hyperparameter name → list of candidate values.
        Example::

            {"lr": [1e-4, 1e-3, 1e-2], "dropout": [0.1, 0.3, 0.5]}

    n_trials:
        Number of random combinations to evaluate.  If the total grid is
        smaller than *n_trials*, all combinations are evaluated once.
    epochs_per_trial:
        Number of training epochs per trial.
    task:
        ``"classification"`` or ``"regression"``.

    Yields
    ------
    dict
        Per-trial progress dict::

            {"trial": int, "total_trials": int, "params": dict,
             "val_acc": float, "done": False}

        After all trials::

            {"done": True, "best_params": dict, "best_val_acc": float,
             "all_results": list[dict]}
    """
    combos = _sample_param_combinations(param_space, n_trials)
    total  = len(combos)
    all_results: list[dict[str, Any]] = []

    best_val_acc  = -float("inf")
    best_params: dict[str, Any] = {}

    for i, params in enumerate(combos):
        final = _run_trial(
            model_factory, train_loader, val_loader,
            params, epochs_per_trial, task,
        )

        history  = final.get("history", [])
        val_acc  = history[-1]["val_acc"] if history else 0.0

        result = {
            "trial":        i + 1,
            "total_trials": total,
            "params":       params,
            "val_acc":      val_acc,
        }
        all_results.append(result)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = params

        yield {**result, "done": False}

    yield {
        "done":         True,
        "best_params":  best_params,
        "best_val_acc": best_val_acc,
        "all_results":  all_results,
    }


def grid_search(
    model_factory: Callable[[dict[str, Any]], nn.Module],
    train_loader,
    val_loader,
    param_space: dict[str, list[Any]],
    epochs_per_trial: int = 5,
    task: str = "classification",
) -> Generator[dict[str, Any], None, None]:
    """Exhaustive grid search over all combinations in *param_space*.

    Interface is identical to :func:`random_search` except there is no
    *n_trials* argument — every combination is evaluated.

    Parameters
    ----------
    model_factory:
        Callable ``(params: dict) -> nn.Module``.
    train_loader / val_loader:
        Standard PyTorch ``DataLoader`` objects.
    param_space:
        Full grid of hyperparameter values.
    epochs_per_trial:
        Number of training epochs per trial.
    task:
        ``"classification"`` or ``"regression"``.

    Yields
    ------
    dict
        Same schema as :func:`random_search`.
    """
    combos = _grid_param_combinations(param_space)
    total  = len(combos)
    all_results: list[dict[str, Any]] = []

    best_val_acc  = -float("inf")
    best_params: dict[str, Any] = {}

    for i, params in enumerate(combos):
        final = _run_trial(
            model_factory, train_loader, val_loader,
            params, epochs_per_trial, task,
        )

        history  = final.get("history", [])
        val_acc  = history[-1]["val_acc"] if history else 0.0

        result = {
            "trial":        i + 1,
            "total_trials": total,
            "params":       params,
            "val_acc":      val_acc,
        }
        all_results.append(result)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = params

        yield {**result, "done": False}

    yield {
        "done":         True,
        "best_params":  best_params,
        "best_val_acc": best_val_acc,
        "all_results":  all_results,
    }


def format_search_results(all_results: list[dict[str, Any]]) -> str:
    """Render *all_results* as a Markdown table sorted by ``val_acc`` descending.

    Parameters
    ----------
    all_results:
        List of per-trial result dicts as yielded by :func:`random_search` or
        :func:`grid_search` (``done=False`` variants).

    Returns
    -------
    str
        Markdown table string, e.g.::

            ### Hyperparameter Search Results (10 trials)
            | Trial | Val Acc | lr     | dropout |
            |-------|---------|--------|---------|
            | 3     | 91.20%  | 0.001  | 0.1     |
            ...
            **Best Val Acc: 91.20%**
    """
    if not all_results:
        return "_No results to display._"

    sorted_results = sorted(all_results, key=lambda r: r["val_acc"], reverse=True)
    n_trials = len(sorted_results)

    # Collect all param keys that appear across results
    param_keys: list[str] = []
    seen: set[str] = set()
    for r in sorted_results:
        for k in r.get("params", {}).keys():
            if k not in seen:
                param_keys.append(k)
                seen.add(k)

    # Header row
    header_cols = ["Trial", "Val Acc"] + param_keys
    header  = "| " + " | ".join(header_cols) + " |"
    divider = "| " + " | ".join(["---"] * len(header_cols)) + " |"

    rows: list[str] = []
    for r in sorted_results:
        cells = [str(r["trial"]), f"{r['val_acc']:.2f}%"]
        for k in param_keys:
            val = r.get("params", {}).get(k, "—")
            cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")

    best_acc = sorted_results[0]["val_acc"]
    lines = [
        f"### Hyperparameter Search Results ({n_trials} trial{'s' if n_trials != 1 else ''})",
        "",
        header,
        divider,
        *rows,
        "",
        f"**Best Val Acc: {best_acc:.2f}%**",
    ]
    return "\n".join(lines)
