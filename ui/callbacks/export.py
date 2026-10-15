"""Export & History tab: FastAPI, Docker, model card and Streamlit
generators, and the run history."""
from __future__ import annotations

import logging, os

import gradio as gr

from ui.helpers import HISTORY_COLUMNS, _format_status_block, _list_bundles
from ui.layout import Components

logger = logging.getLogger("nocode_dl")


# ── FastAPI generator ─────────────────────────────────────────────────────

def gen_fastapi(bundle_path, mod):
    if not bundle_path:
        return _format_status_block("FastAPI export", "No bundle available. Train a model first.", success=False)
    try:
        from export.fastapi_generator import generate_fastapi_app
        path = generate_fastapi_app(bundle_path, mod)
        return _format_status_block(
            "FastAPI export",
            (f"FastAPI app written to:\n  {path}\n\n"
             f"Start server (reachable only from this computer):\n"
             f'  cd "{bundle_path}"\n'
             f"  pip install -r requirements_serve.txt\n"
             f"  python main.py\n\n"
             f"Then open http://127.0.0.1:8000/docs"),
        )
    except Exception as e:
        return _format_status_block("FastAPI export", f"Error: {e}", success=False)

# ── Docker generator ──────────────────────────────────────────────────────


def gen_docker(bundle_path, mod):
    if not bundle_path:
        return _format_status_block("Docker export", "No bundle available. Train a model first.", success=False)
    try:
        from export.docker_generator import generate_docker_bundle
        paths = generate_docker_bundle(bundle_path, mod)
        files = "\n  ".join(paths)
        return _format_status_block(
            "Docker export",
            (f"Docker files written:\n  {files}\n\n"
             f"Build & run (reachable only from this computer):\n"
             f'  cd "{bundle_path}"\n'
             f"  docker compose up --build\n\n"
             f"See README_deploy.md for exposing it with an API key."),
        )
    except Exception as e:
        return _format_status_block("Docker export", f"Error: {e}", success=False)

# ── Model card generator ──────────────────────────────────────────────────


def gen_model_card(bundle_path, mod, mname, tmode, tsk, eval_sum):
    if not bundle_path:
        return _format_status_block("Model card", "No bundle available. Train a model first.", success=False)
    try:
        from export.model_card import generate_model_card
        import json
        with open(f"{bundle_path}/labels.json") as f:
            labels_dict = json.load(f)
        classes = list(labels_dict.values())
        path = generate_model_card(
            bundle_path=bundle_path, modality=mod, model_name=mname,
            training_mode=tmode, task=tsk, classes=classes,
            hyperparams={}, history=[], eval_summary=eval_sum)
        return _format_status_block("Model card", f"Model card saved:\n  {path}")
    except Exception as e:
        return _format_status_block("Model card", f"Error: {e}", success=False)

# ── Streamlit dashboard generator ─────────────────────────────────────────


def gen_streamlit(bundle_path_val, mod, mname, tmode, tsk, eval_summary_val):
    if not bundle_path_val:
        return "Set the bundle path first (train a model or select one)."
    try:
        from export.streamlit_generator import generate_streamlit_dashboard
        # Try to load history and metrics from run_history.json
        run_history = []
        run_metrics = {}
        run_classes = []
        try:
            from training.run_comparison import load_history
            all_runs = load_history()
            for run in reversed(all_runs):
                if run.get("bundle_path") == bundle_path_val:
                    run_history = run.get("history", [])
                    run_metrics = run.get("metrics") or {}
                    break
        except Exception:
            pass
        # Try to load classes from labels.json
        try:
            import json as _json
            labels_file = os.path.join(bundle_path_val, "labels.json")
            if os.path.isfile(labels_file):
                with open(labels_file, encoding="utf-8") as fh:
                    run_classes = list(_json.load(fh).values())
        except Exception:
            pass

        out_path = generate_streamlit_dashboard(
            bundle_path=bundle_path_val,
            modality=mod,
            model_name=mname,
            task=tsk,
            training_mode=tmode,
            history=run_history,
            metrics=run_metrics,
            classes=run_classes,
            eval_summary=eval_summary_val,
        )
        return (
            f"### Streamlit dashboard exported\n\n"
            f"```\n{out_path}\n```\n\n"
            f"Run it with:\n\n"
            f"```bash\npip install streamlit pandas matplotlib numpy\n"
            f"streamlit run \"{out_path}\"\n```"
        )
    except Exception as e:
        return f"### Export failed\n\n```\n{e}\n```"

# ── Run history ───────────────────────────────────────────────────────────


def refresh_history():
    try:
        from training.run_comparison import history_dataframe, HISTORY_FILE
        if not HISTORY_FILE.exists():
            return gr.update(value=[], headers=HISTORY_COLUMNS)
        df = history_dataframe()
        if not len(df):
            return gr.update(value=[], headers=HISTORY_COLUMNS)
        ordered = df.reindex(columns=HISTORY_COLUMNS)
        return gr.update(value=ordered.values.tolist(), headers=HISTORY_COLUMNS)
    except Exception as e:
        logger.warning("refresh_history failed: %s", e)
        return gr.update(value=[], headers=HISTORY_COLUMNS)


def compare_runs(ids_str):
    try:
        ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
        if not ids:
            return (
                "### ⚠️ Run comparison\n\nEnter one or more run IDs from the history table, for example `0` or `0,1`.",
                None,
            )
        from training.run_comparison import compare_runs_plot
        return (
            f"### ✅ Run comparison\n\nComparing run IDs: `{', '.join(str(i) for i in ids)}`",
            compare_runs_plot(ids),
        )
    except Exception as e:
        logger.warning("compare_runs failed: %s", e)
        return (
            f"### ⚠️ Run comparison\n\nCould not compare runs.\n\n```text\n{e}\n```",
            None,
        )


def register(c: Components) -> None:
    """Wire this area's events to the components in ``c``."""
    c.fastapi_btn.click(gen_fastapi, inputs=[c.bundle_out, c.modality], outputs=c.fastapi_status_box)

    c.docker_btn.click(gen_docker, inputs=[c.bundle_out, c.modality], outputs=c.docker_status_box)

    c.model_card_btn.click(gen_model_card,
                         inputs=[c.bundle_out, c.modality, c.model_name,
                                 c.training_mode, c.task, c.eval_summary_box],
                         outputs=c.model_card_status_box)

    c.streamlit_btn.click(
        gen_streamlit,
        inputs=[c.bundle_out, c.modality, c.model_name, c.training_mode, c.task, c.eval_summary_box],
        outputs=c.streamlit_status,
    )

    c.refresh_btn.click(refresh_history, outputs=c.history_df)
    c.compare_btn.click(compare_runs, inputs=c.compare_ids, outputs=[c.compare_status_box, c.compare_plot])
    # Only auto-load history on page load if the history file already exists —
    # avoids a redundant disk read (and silent failure) on a fresh install.
    from training.run_comparison import HISTORY_FILE as _HIST_FILE
    if _HIST_FILE.exists():
        c.demo.load(refresh_history, outputs=c.history_df)

    c.demo.load(
        lambda: gr.update(choices=_list_bundles()),
        outputs=c.infer_bundle_path,
    )
