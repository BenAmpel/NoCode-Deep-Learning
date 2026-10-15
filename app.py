"""
NoCode-DL — No-Code Deep Learning Platform
Run with:  python run_local.py

This module only assembles the app: ui/layout.py builds the components,
ui/callbacks/* wire their events, and launch_app() serves the result.
"""
from __future__ import annotations
import logging, os

# Must be set before gradio is imported. Gradio's usage analytics contact
# api.gradio.app and checkip.amazonaws.com, which contradicts the app's
# "your data never leaves this machine" promise.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from config import DEVICE
from pipeline import RunConfig
from ui.callbacks import annotate, data, detect, evaluate, export, infer, train
from ui.helpers import _OUTPUTS_ROOT
from ui.layout import build_layout
from ui.theme import APP_THEME, _APP_CSS_PATH, _SYNC_DARK_CLASS_JS

# Entry points that run_local.py, the smoke tests and the unit tests import from app.
from ui.callbacks.data import load_builtin_walkthrough, preview_dataset
from ui.callbacks.detect import detect_image, detect_video
from ui.callbacks.evaluate import run_model_sweep
from ui.callbacks.export import compare_runs, gen_docker, gen_fastapi, gen_streamlit
from ui.callbacks.infer import _batch_predict
from ui.callbacks.train import run_pipeline
from ui.helpers import _loss_plot

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nocode_dl")

__all__ = [
    "RunConfig", "c", "demo", "launch_app",
    "run_pipeline", "run_model_sweep", "preview_dataset", "load_builtin_walkthrough",
    "detect_image", "detect_video", "compare_runs", "gen_docker", "gen_fastapi",
    "gen_streamlit", "_batch_predict", "_loss_plot",
]

demo, c = build_layout()

# Registration order matches the old single-file app, so events that share a
# trigger still fire in the same order.
with demo:
    data.register(c)
    heavy_events = train.register(c)
    heavy_events |= evaluate.register(c)
    export.register(c)
    heavy_events |= detect.register(c)
    train.register_stop_button(c, [
        heavy_events[name] for name in (
            "train_event", "cv_event", "cv_sig_event", "hparam_event",
            "sweep_event", "det_vid_event", "yolo_event",
        )
    ])
    annotate.register(c)
    infer.register(c)


def launch_app(
    *,
    inbrowser: bool = True,
    server_port: int = 7860,
    prevent_thread_lock: bool = False,
    frontend: bool = True,
    show_api: bool = True,
):
    logger.info("Device: %s", DEVICE)
    return demo.launch(
        share=False,
        inbrowser=inbrowser,
        server_name="127.0.0.1",
        server_port=server_port,
        prevent_thread_lock=prevent_thread_lock,
        theme=APP_THEME,
        css=_APP_CSS_PATH.read_text(encoding="utf-8"),
        js=_SYNC_DARK_CLASS_JS,
        # Gradio 6 replaced show_api with footer_links.
        footer_links=["api", "gradio", "settings"] if show_api else ["gradio", "settings"],
        # Gradio 6 only serves files from the working directory, the temp dir
        # and allowed_paths; outputs/ holds trained bundles, previews and plots.
        allowed_paths=[str(_OUTPUTS_ROOT)],
        _frontend=frontend,
    )


if __name__ == "__main__":
    launch_app()

