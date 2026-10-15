"""Static checks of every Gradio event registered on ``app.demo``.

The smoke test calls handlers directly, so it never notices when an event's
``inputs``/``outputs`` lists drift away from the handler they feed. These
tests enumerate ``demo.fns`` (Gradio 6 BlockFunction objects) and check:

* every component an event references (targets, inputs, outputs) is part of
  the Blocks tree;
* the number of inputs fits the handler's positional parameters;
* every ``return``/``yield`` of a tuple or list literal has exactly as many
  items as the event has outputs (following calls to helper functions
  defined inside the handler, e.g. ``yield _emit(...)`` in run_pipeline).
"""
from __future__ import annotations

import ast
import functools
import inspect
import textwrap

import pytest

import app

demo = app.demo

# Parameters Gradio fills in itself rather than from ``inputs``.
_INJECTED_ANNOTATIONS = ("Request", "EventData", "Progress", "SelectData", "LikeData", "KeyUpData")


def _events():
    return [bf for bf in demo.fns.values() if bf.fn is not None]


def _unwrap(fn):
    while isinstance(fn, functools.partial):
        fn = fn.func
    return inspect.unwrap(fn)


def _event_id(bf) -> str:
    fn = _unwrap(bf.fn)
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", repr(fn))
    return f"{bf._id}-{name}"


def _is_injected(param: inspect.Parameter) -> bool:
    ann = param.annotation
    ann_text = ann if isinstance(ann, str) else getattr(ann, "__name__", "")
    if any(tag in str(ann_text) for tag in _INJECTED_ANNOTATIONS):
        return True
    return type(param.default).__name__ == "Progress"


def _param_bounds(fn) -> tuple[int, float]:
    """(minimum, maximum) number of positional inputs ``fn`` accepts."""
    sig = inspect.signature(fn)
    required = 0
    total: float = 0
    for p in sig.parameters.values():
        if _is_injected(p):
            continue
        if p.kind is p.VAR_POSITIONAL:
            total = float("inf")
        elif p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            total += 1
            if p.default is p.empty:
                required += 1
    return required, total


EVENTS = _events()
IDS = [_event_id(bf) for bf in EVENTS]


def test_events_were_found():
    # A broken enumeration would make the other tests pass vacuously.
    assert len(EVENTS) > 50


@pytest.mark.parametrize("bf", EVENTS, ids=IDS)
def test_referenced_components_exist(bf):
    known = set(demo.blocks.keys())
    for comp in list(bf.inputs) + list(bf.outputs):
        assert comp._id in known, f"{comp!r} is not part of app.demo"
    for target_id, event in bf.targets:
        if event == "then":  # chained step: its trigger is the earlier event
            assert bf.trigger_after in demo.fns, f".then() follows unknown event {bf.trigger_after}"
            continue
        if event == "load":  # demo.load() targets the Blocks itself
            assert target_id == demo._id
            continue
        assert target_id in known, f"event target {target_id} is not part of app.demo"


@pytest.mark.parametrize("bf", EVENTS, ids=IDS)
def test_input_count_matches_handler(bf):
    lo, hi = _param_bounds(bf.fn)
    # With a set of inputs Gradio passes one {component: value} dict.
    n = 1 if bf.inputs_as_dict else len(bf.inputs)
    assert lo <= n <= hi, f"{n} inputs for a handler taking {lo}..{hi} positional parameters"


# ── return / yield arity ─────────────────────────────────────────────────────

class _ReturnCollector(ast.NodeVisitor):
    """Collect return/yield values of one function, not of nested defs."""

    def __init__(self):
        self.values: list[ast.expr] = []
        self.local_defs: dict[str, ast.FunctionDef] = {}
        self._depth = 0

    def _visit_def(self, node):
        if self._depth == 0:
            self._depth += 1
            self.generic_visit(node)
            self._depth -= 1
        else:
            self.local_defs.setdefault(node.name, node)

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def

    def visit_Lambda(self, node):
        pass  # a nested lambda's body is not the handler's return value

    def visit_Return(self, node):
        if node.value is not None:
            self.values.append(node.value)

    def visit_Yield(self, node):
        if node.value is not None:
            self.values.append(node.value)

    def visit_YieldFrom(self, node):
        self.values.append(node.value)  # a call: its yields are followed


def _function_node(fn):
    """AST of a plain function, with line numbers matching its source file."""
    try:
        lines, first_line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return None
    tree = ast.parse(textwrap.dedent("".join(lines)))
    ast.increment_lineno(tree, first_line - 1)
    return tree.body[0]


def _literal_arity(value: ast.expr, local_defs: dict, fn_globals: dict, seen: frozenset) -> list[tuple[int, int]]:
    """(lineno, item count) for each tuple/list literal ``value`` can evaluate to.

    Calls are followed into helpers defined inside the handler and into
    module-level functions of the handler's module (``return helper(...)``).
    """
    if isinstance(value, (ast.Tuple, ast.List)):
        if any(isinstance(e, ast.Starred) for e in value.elts):
            return []
        return [(value.lineno, len(value.elts))]
    if isinstance(value, ast.IfExp):
        return (_literal_arity(value.body, local_defs, fn_globals, seen)
                + _literal_arity(value.orelse, local_defs, fn_globals, seen))
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
        return []
    name = value.func.id
    if name in seen:
        return []
    if name in local_defs:
        helper, helper_globals = local_defs[name], fn_globals
    else:
        target = fn_globals.get(name)
        if not inspect.isfunction(target) or getattr(target, "__module__", None) != fn_globals.get("__name__"):
            return []
        helper, helper_globals = _function_node(target), target.__globals__
        if helper is None or isinstance(helper, ast.ClassDef):
            return []
    collector = _ReturnCollector()
    collector.visit(helper)
    scope = {**local_defs, **collector.local_defs} if name in local_defs else collector.local_defs
    out = []
    for v in collector.values:
        out += _literal_arity(v, scope, helper_globals, seen | {name})
    return out


def _handler_node(fn):
    fn = _unwrap(fn)
    lines, first_line = inspect.getsourcelines(fn)
    if fn.__name__ == "<lambda>":
        # getsource returns the whole line(s) holding the lambda.
        src = textwrap.dedent("".join(lines)).strip()
        tree = None
        for candidate in (src, "(" + src.rstrip(",") + ")", src.rstrip(",)") + ")"):
            try:
                tree = ast.parse(candidate)
                break
            except SyntaxError:
                continue
        if tree is None:
            return None
        lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
        return lambdas[0] if len(lambdas) == 1 else None
    return _function_node(fn)


@pytest.mark.parametrize("bf", EVENTS, ids=IDS)
def test_return_arity_matches_outputs(bf):
    n_out = len(bf.outputs)
    node = _handler_node(bf.fn)
    if node is None:
        pytest.skip("handler source is ambiguous (several lambdas on one line)")
    collector = _ReturnCollector()
    if isinstance(node, ast.Lambda):
        values = [node.body]
    else:
        collector.visit(node)
        values = collector.values
    problems = []
    for value in values:
        for line, count in _literal_arity(value, collector.local_defs, _unwrap(bf.fn).__globals__, frozenset()):
            # A single output may legitimately receive a tuple or list value.
            if n_out != 1 and count != n_out:
                problems.append(f"line {line}: {count} values for {n_out} outputs")
    assert not problems, "; ".join(problems)


# ── handlers driven through their early-exit paths ───────────────────────────

def _event_for(fn_name: str):
    matches = [bf for bf in EVENTS if _unwrap(bf.fn).__name__ == fn_name]
    assert len(matches) == 1, f"expected one {fn_name} event, found {len(matches)}"
    return matches[0]


def _default_values(bf) -> dict:
    return {comp: getattr(comp, "value", None) for comp in bf.inputs}


def test_train_button_yields_one_value_per_output():
    bf = _event_for("train_from_ui")
    values = _default_values(bf)
    values[app.c.data_path] = ""  # fails pre-flight before any work starts
    updates = list(bf.fn(values))
    assert len(updates) == 1
    assert len(updates[0]) == len(bf.outputs)
    assert "No data path provided" in updates[0][0]


def test_sweep_button_yields_one_value_per_output():
    bf = _event_for("sweep_from_ui")
    values = _default_values(bf)
    values[app.c.sweep_models] = ["MLP"]  # fewer than two models: stops early
    updates = list(bf.fn(values))
    assert updates and all(len(u) == len(bf.outputs) for u in updates)
    assert "at least two models" in updates[0][0]
