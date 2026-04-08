"""
NoCode-DL — No-Code Deep Learning Platform
Run with:  python run_local.py
"""
from __future__ import annotations
import atexit, html, logging, os, re, time, traceback, tempfile, shutil, zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from config import DEVICE, DEFAULTS, VIDEO_DEFAULTS, TEXT_DEFAULTS
from models.registry import get_models, get_compatible_tasks, get_modes, is_sklearn, is_vit, is_whisper
from ui.project_state import load_project_state, save_project_state
from ui.tooltips import TOOLTIPS

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nocode_dl")

# Project root — used to build CWD-independent absolute paths
_PROJECT_ROOT = Path(__file__).resolve().parent
_OUTPUTS_ROOT = _PROJECT_ROOT / "outputs"
_BUILTIN_IMAGE_TUTORIAL_PATH      = _PROJECT_ROOT / "fixtures" / "mnist_tutorial"
_BUILTIN_TABULAR_TUTORIAL_PATH    = _PROJECT_ROOT / "fixtures" / "iris_tutorial"
_BUILTIN_TEXT_TUTORIAL_PATH       = _PROJECT_ROOT / "fixtures" / "newsgroups_tutorial"
_BUILTIN_AUDIO_TUTORIAL_PATH      = _PROJECT_ROOT / "fixtures" / "speechcommands_tutorial"
_BUILTIN_TIMESERIES_TUTORIAL_PATH = _PROJECT_ROOT / "fixtures" / "timeseries_tutorial"
_BUILTIN_VIDEO_TUTORIAL_PATH      = _PROJECT_ROOT / "fixtures" / "video_tutorial"
_LIVE_PLOT_UPDATE_EVERY = 2


def _path_signature(dpath: str | os.PathLike | None) -> tuple[str, float, bool]:
    if not dpath:
        return ("", 0.0, False)
    try:
        path = Path(dpath).expanduser().resolve()
        stat = path.stat()
        return (str(path), stat.st_mtime, path.is_dir())
    except Exception:
        return (str(dpath), 0.0, False)


def _get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _patch_gradio_schema_parser() -> None:
    """
    Gradio 4.44 can emit JSON schema fragments where ``additionalProperties`` is
    a boolean. Its client-side schema parser assumes a dict and crashes while
    building API metadata, preventing the local app from loading.
    """
    try:
        from gradio_client import utils as client_utils
    except Exception:
        return

    if getattr(client_utils, "_nocode_dl_schema_patch", False):
        return

    original = client_utils._json_schema_to_python_type

    def patched(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original(schema, defs)

    client_utils._json_schema_to_python_type = patched
    client_utils._nocode_dl_schema_patch = True


_patch_gradio_schema_parser()

# Keep YOLO options local so the app doesn't touch cv2/ultralytics at startup.
YOLO_MODELS = {
    "YOLOv8 Nano  (~6 MB  · fastest)":   "yolov8n.pt",
    "YOLOv8 Small (~22 MB · balanced)":  "yolov8s.pt",
    "YOLOv8 Medium (~52 MB · accurate)": "yolov8m.pt",
}

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODALITIES = ["image", "text", "tabular", "timeseries", "audio", "video"]
PROJECT_MODES = ["Beginner", "Guided", "Advanced"]
OPTIMIZERS = ["adam", "adamw", "sgd"]
SCHEDULERS = ["cosine", "warmup_cosine", "step", "none"]
AUG_LEVELS = ["none", "light", "medium", "heavy"]
TASKS      = ["classification", "multi-label", "clustering", "regression", "anomaly"]

APP_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#f6f8f4",
    background_fill_primary="#ffffff",
    background_fill_secondary="#edf3ef",
    block_background_fill="#ffffff",
    block_border_color="#d6e2d8",
    block_label_background_fill="#ffffff",
    block_title_text_color="#16312a",
    body_text_color="#2f4b43",
    color_accent="#16856f",
    color_accent_soft="#d9f2ea",
    input_background_fill="#fbfdfb",
    input_border_color="#bfd4c8",
    button_primary_background_fill="#115d52",
    button_primary_background_fill_hover="#0f7263",
    button_primary_text_color="#f8fffd",
    button_secondary_background_fill="#ecf5f0",
    button_secondary_background_fill_hover="#dceee6",
    button_secondary_text_color="#115d52",
)

APP_CSS = """
/* ── Design tokens ────────────────────────────────────────────────────────── */
:root {
  /* Text */
  --color-text-primary: #1f3b35;
  --color-text-secondary: #47655d;
  --color-text-muted: #6a857c;
  --color-text-on-primary: #f8fffd;
  --color-text-on-surface: #27554d;

  /* Accent */
  --color-accent: #16856f;
  --color-accent-dark: #115d52;
  --color-gold: #f0ad22;

  /* Surfaces */
  --color-surface: #ffffff;
  --color-surface-soft: #f9fcfa;
  --color-surface-tint: #f6fbf8;
  --color-surface-raised: rgba(255, 255, 255, 0.78);
  --color-page-start: #fafbf7;
  --color-page-end: #eef4f0;

  /* Borders */
  --color-border: #d7e5db;
  --color-border-soft: rgba(196, 218, 209, 0.92);
  --color-border-row: #e6efea;

  /* Tab */
  --color-tab-text: #45655d;
  --color-tab-active: #105d52;
  --color-tab-active-border: #b7d4c9;
  --color-label: #284c44;

  /* Gradio overrides (dark mode) */
  --color-input-bg: var(--color-surface);
  --color-input-border: var(--color-border);
  --color-input-text: var(--color-text-primary);
  --color-placeholder: var(--color-text-muted);
  --color-table-bg: var(--color-surface);
  --color-table-text: var(--color-text-primary);
  --color-table-border: var(--color-border-row);
  --color-code-bg: var(--color-surface-soft);
  --color-code-text: var(--color-text-primary);
  --color-code-border: var(--color-border);
  --color-hr: var(--color-border);

  /* Shadows */
  --shadow-xl: 0 28px 70px rgba(31, 65, 57, 0.10);
  --shadow-lg: 0 16px 34px rgba(22, 56, 49, 0.07);
  --shadow-md: 0 12px 24px rgba(17, 93, 82, 0.12);
  --shadow-sm: 0 10px 22px rgba(20, 55, 48, 0.05);

  /* Radius */
  --radius-full: 999px;
  --radius-xl: 26px;
  --radius-lg: 20px;
  --radius-md: 16px;
  --radius-sm: 14px;

  /* Type scale (1.25 ratio, 1rem = 16px base) */
  --text-display: 3.052rem;
  --text-display-sm: 2.441rem;
  --text-h2: 1.563rem;
  --text-h3: 1.25rem;
  --text-lg: 1.125rem;
  --text-base: 1rem;
  --text-sm: 0.875rem;
  --text-xs: 0.75rem;

  /* Spacing (4px grid) */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-7: 28px;
  --space-8: 34px;
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
body {
  font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
  background:
    radial-gradient(ellipse at 20% 0%, rgba(48, 152, 125, 0.14), transparent 50%),
    linear-gradient(180deg, var(--color-page-start) 0%, var(--color-page-end) 100%);
  color: var(--color-text-primary);
}

.gradio-container {
  max-width: 1460px !important;
  padding: var(--space-6) var(--space-5) 48px !important;
}

footer {
  display: none !important;
}

.studio-shell {
  gap: 18px;
}

.studio-shell .gradio-container,
.studio-shell .contain {
  background: transparent !important;
}

/* ── Hero ─────────────────────────────────────────────────────────────────── */
.studio-hero {
  margin-bottom: var(--space-3);
  padding: var(--space-8) var(--space-8) 30px;
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-xl);
  background:
    radial-gradient(ellipse at 10% 0%, rgba(72, 176, 149, 0.18), transparent 50%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(242, 249, 245, 0.96) 50%, rgba(231, 244, 239, 0.98) 100%);
  box-shadow: var(--shadow-xl);
  color: var(--color-text-primary);
}

.studio-hero__grid {
  display: grid;
  gap: var(--space-7);
}

.studio-hero__eyebrow {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
  padding: 7px var(--space-3);
  border: 1px solid rgba(183, 214, 203, 0.95);
  border-radius: var(--radius-full);
  background: rgba(255, 255, 255, 0.74);
  font-size: var(--text-xs);
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--color-text-muted);
}

.studio-hero__eyebrow::before {
  content: "";
  width: var(--space-2);
  height: var(--space-2);
  border-radius: var(--radius-full);
  background: linear-gradient(180deg, #17b37b 0%, #0d8b60 100%);
  box-shadow: 0 0 0 5px rgba(23, 179, 123, 0.14);
}

.studio-hero__title {
  margin: 0;
  max-width: 720px;
  font-size: var(--text-display);
  line-height: 1.1;
  letter-spacing: -0.05em;
  color: var(--color-text-primary);
}

.studio-hero__subtitle {
  max-width: 740px;
  margin: var(--space-4) 0 22px;
  font-size: var(--text-lg);
  line-height: 1.6;
  color: var(--color-text-secondary);
}

.studio-hero__pills {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  margin-bottom: var(--space-5);
}

.studio-hero__pill {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: 10px var(--space-3);
  border: 1px solid rgba(191, 215, 205, 0.95);
  border-radius: var(--radius-full);
  background: var(--color-surface-raised);
  color: var(--color-text-on-surface);
  font-size: var(--text-sm);
  font-weight: 600;
  box-shadow: var(--shadow-sm);
}

.studio-hero__pill::before {
  content: "✦";
  color: var(--color-gold);
  font-size: var(--text-xs);
}

.studio-hero__meta {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-3);
}

.studio-hero__meta-card {
  padding: var(--space-4);
  border: 1px solid rgba(184, 210, 199, 0.95);
  border-radius: var(--radius-lg);
  background: var(--color-surface-raised);
  box-shadow: var(--shadow-lg);
}

.studio-hero__meta-label {
  display: block;
  margin-bottom: 6px;
  font-size: var(--text-xs);
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--color-text-muted);
}

.studio-hero__meta-value {
  font-size: var(--text-base);
  font-weight: 600;
  color: var(--color-text-primary);
}

/* ── Banner ───────────────────────────────────────────────────────────────── */
.studio-banner {
  margin-bottom: var(--space-3);
  padding: 18px var(--space-5);
  border: 1px solid rgba(197, 216, 204, 0.95);
  border-radius: var(--radius-lg);
  background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(243,249,245,0.96) 100%);
  box-shadow: var(--shadow-lg);
}

.studio-banner__grid {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: var(--space-5);
  align-items: center;
}

.studio-banner__eyebrow {
  margin-bottom: 6px;
  font-size: var(--text-xs);
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--color-text-muted);
}

.studio-banner h2 {
  margin: 0 0 var(--space-2);
  font-size: var(--text-h2);
  line-height: 1.15;
  letter-spacing: -0.03em;
  color: var(--color-text-primary);
}

.studio-banner p {
  margin: 0;
  max-width: 65ch;
  font-size: var(--text-base);
  line-height: 1.6;
  color: var(--color-text-secondary);
}

.studio-badge-list {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
}

.studio-badge {
  padding: 10px var(--space-3);
  border: 1px solid rgba(191, 215, 205, 0.9);
  border-radius: var(--radius-md);
  background: rgba(255,255,255,0.74);
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--color-text-on-surface);
}

.studio-showcase {
  display: grid;
  gap: var(--space-3);
}

.studio-showcase__card {
  padding: var(--space-3) var(--space-3);
  border: 1px solid rgba(196, 215, 205, 0.92);
  border-radius: var(--radius-lg);
  background: rgba(255,255,255,0.82);
}

.studio-showcase__card strong {
  display: block;
  margin-bottom: var(--space-1);
  color: var(--color-text-primary);
}

.studio-showcase__card span {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
  line-height: 1.45;
}

/* ── Cards & Panels ───────────────────────────────────────────────────────── */
.studio-panel {
  border: 1px solid rgba(205, 216, 231, 0.78);
  border-radius: var(--radius-xl);
  background: var(--color-surface-raised);
  box-shadow: var(--shadow-xl);
}

.studio-card {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: linear-gradient(180deg, var(--color-surface) 0%, #fbfdfb 100%);
  box-shadow: var(--shadow-sm);
  padding: 18px !important;
  contain: layout style;
}

.studio-card--compact {
  padding: var(--space-4) !important;
}

.studio-card--soft {
  background: linear-gradient(180deg, var(--color-surface-soft) 0%, #eff7f2 100%);
}

/* ── Section Intros ───────────────────────────────────────────────────────── */
.studio-section-intro {
  margin-bottom: var(--space-3);
}

.studio-section-intro h3 {
  margin: 0 0 6px;
  font-size: var(--text-h3);
  letter-spacing: -0.02em;
  color: var(--color-text-primary);
}

.studio-section-intro p {
  margin: 0;
  max-width: 65ch;
  font-size: var(--text-base);
  line-height: 1.6;
  color: var(--color-text-secondary);
}

.studio-tab-panel {
  padding-top: var(--space-3);
}

/* ── Tabs (Gradio overrides) ──────────────────────────────────────────────── */
button[role="tab"] {
  margin-right: var(--space-2) !important;
  border-radius: var(--radius-full) !important;
  border: 1px solid var(--color-border) !important;
  background: rgba(255, 255, 255, 0.82) !important;
  color: var(--color-tab-text) !important;
  font-weight: 700 !important;
  transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease !important;
}

button[role="tab"][aria-selected="true"] {
  border-color: var(--color-tab-active-border) !important;
  background: linear-gradient(180deg, var(--color-surface) 0%, #e8f5ef 100%) !important;
  color: var(--color-tab-active) !important;
  box-shadow: var(--shadow-md);
}

.studio-shell .form {
  border-radius: var(--radius-md);
}

.studio-shell .gr-group,
.studio-shell .gr-box,
.studio-shell .gr-form,
.studio-shell .gr-accordion {
  border-color: var(--color-border) !important;
}

.studio-shell .gr-button-primary {
  box-shadow: var(--shadow-md);
}

.studio-shell .gr-button-secondary {
  box-shadow: var(--shadow-sm);
}

.studio-shell label,
.studio-shell .gradio-textbox label,
.studio-shell .gradio-dropdown label {
  color: var(--color-label) !important;
  font-weight: 600 !important;
}

/* ── Cockpit & KPIs ───────────────────────────────────────────────────────── */
.studio-cockpit {
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.studio-cockpit .gr-markdown {
  min-height: 160px;
}

.studio-kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.studio-kpi-grid--compact {
  margin-top: var(--space-2);
  margin-bottom: 10px;
}

.studio-kpi {
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: linear-gradient(180deg, var(--color-surface) 0%, var(--color-surface-tint) 100%);
}

.studio-kpi strong {
  display: block;
  margin-bottom: 6px;
  font-size: var(--text-xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--color-text-muted);
}

.studio-kpi span {
  font-size: var(--text-base);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--color-text-primary);
}

.studio-inline-note {
  margin-top: calc(-1 * var(--space-1));
  margin-bottom: var(--space-3);
  font-size: var(--text-sm);
  line-height: 1.5;
  color: var(--color-text-muted);
}

/* ── Reports ──────────────────────────────────────────────────────────────── */
.studio-report {
  display: grid;
  gap: var(--space-3);
}

.studio-report h3 {
  margin: 0;
  font-size: var(--text-h3);
  line-height: 1.3;
  color: var(--color-text-primary);
}

.studio-report-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
}

.studio-report-chip {
  min-width: 140px;
  padding: var(--space-3) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: linear-gradient(180deg, var(--color-surface) 0%, var(--color-surface-tint) 100%);
}

.studio-report-chip strong {
  display: block;
  margin-bottom: var(--space-1);
  font-size: var(--text-xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--color-text-muted);
}

.studio-report-chip span {
  font-size: 1.25rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--color-text-primary);
}

.studio-report-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
}

.studio-report-table {
  width: 100%;
  border-collapse: collapse;
}

.studio-report-table th,
.studio-report-table td {
  padding: 10px var(--space-3);
  border-bottom: 1px solid var(--color-border-row);
  text-align: left;
  font-size: var(--text-base);
  font-variant-numeric: tabular-nums;
}

.studio-report-table th {
  background: var(--color-surface-tint);
  color: var(--color-text-muted);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* ── Gradio element overrides ─────────────────────────────────────────────── */
.studio-shell .gr-plot,
.studio-shell .gr-dataframe,
.studio-shell .gradio-plot {
  border-radius: var(--radius-lg) !important;
}

.studio-shell .gr-dataframe table {
  font-size: var(--text-sm) !important;
}

.studio-shell textarea,
.studio-shell input,
.studio-shell .wrap,
.studio-shell .gr-dropdown,
.studio-shell .gr-dataframe {
  border-radius: var(--radius-sm) !important;
}

/* ── Dark mode ────────────────────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;

    /* Override tokens for dark mode */
    --color-text-primary: #eff8f5;
    --color-text-secondary: #b8cfc9;
    --color-text-muted: #b8cfc9;
    --color-text-on-primary: #f4fffb;
    --color-text-on-surface: #d7ebe4;

    --color-accent: #3dd4a8;
    --color-accent-dark: #1c7d6d;

    --color-surface: rgba(20, 29, 33, 0.96);
    --color-surface-soft: rgba(23, 35, 38, 0.96);
    --color-surface-tint: rgba(20, 29, 33, 0.96);
    --color-surface-raised: rgba(20, 30, 33, 0.84);
    --color-page-start: #0d1518;
    --color-page-end: #121e22;


    --color-border: rgba(62, 92, 87, 0.9);
    --color-border-soft: rgba(62, 91, 86, 0.92);
    --color-border-row: #2f4743;

    --color-tab-text: #a8c2bb;
    --color-tab-active: #effbf7;
    --color-tab-active-border: rgba(89, 152, 137, 0.98);
    --color-label: #deece8;

    --color-input-bg: #121b1f;
    --color-input-border: #39544f;
    --color-input-text: #e6f1ee;
    --color-placeholder: #7f9891;
    --color-table-bg: #121b1f;
    --color-table-text: #e5f2ee;
    --color-table-border: #2f4743;
    --color-code-bg: #0e1518;
    --color-code-text: #daf0ea;
    --color-code-border: #2d4541;
    --color-hr: rgba(67, 99, 93, 0.7);

    --shadow-xl: 0 28px 70px rgba(2, 8, 10, 0.42);
    --shadow-lg: 0 16px 34px rgba(0, 0, 0, 0.22);
    --shadow-md: 0 12px 24px rgba(9, 38, 35, 0.38);
    --shadow-sm: 0 10px 22px rgba(0, 0, 0, 0.22);
  }

  /* Override Gradio theme variables for dark mode */
  gradio-app {
    --block-background-fill: #121b1f !important;
    --background-fill-primary: #121b1f !important;
    --background-fill-secondary: #172024 !important;
    --body-background-fill: #0d1518 !important;
    --block-border-color: rgba(62, 92, 87, 0.9) !important;
    --block-label-background-fill: #121b1f !important;
    --block-title-text-color: #eff8f5 !important;
    --body-text-color: #e6f1ee !important;
    --input-background-fill: #121b1f !important;
    --input-border-color: #39544f !important;
    --color-accent: #3dd4a8 !important;
    --color-accent-soft: #1a3d35 !important;
  }

  body,
  .gradio-container {
    background:
      radial-gradient(ellipse at 20% 0%, rgba(47, 146, 121, 0.14), transparent 50%),
      linear-gradient(180deg, var(--color-page-start) 0%, var(--color-page-end) 100%) !important;
    color: var(--color-text-primary) !important;
  }

  .studio-shell .gradio-container,
  .studio-shell .contain {
    background: transparent !important;
  }

  /* Hero dark overrides (structural — gradients can't be tokenized) */
  .studio-hero {
    border-color: var(--color-border-soft);
    background:
      radial-gradient(ellipse at 10% 0%, rgba(57, 153, 126, 0.14), transparent 50%),
      linear-gradient(135deg, rgba(19, 28, 32, 0.98) 0%, rgba(18, 35, 35, 0.98) 50%, rgba(20, 41, 39, 0.98) 100%);
  }

  .studio-hero__eyebrow,
  .studio-hero__pill,
  .studio-hero__meta-card {
    border-color: rgba(71, 104, 97, 0.94);
    background: var(--color-surface-raised);
  }

  /* Card & surface dark overrides (structural gradients) */
  .studio-panel {
    border-color: rgba(62, 92, 87, 0.78) !important;
    background: rgba(16, 24, 27, 0.92) !important;
  }

  .studio-banner,
  .studio-card,
  .studio-kpi,
  .studio-report-chip,
  .studio-report-table-wrap,
  .studio-showcase__card,
  .studio-badge {
    border-color: var(--color-border) !important;
    background: linear-gradient(180deg, rgba(20, 29, 33, 0.96) 0%, rgba(16, 25, 28, 0.98) 100%) !important;
  }

  .studio-card--soft {
    background: linear-gradient(180deg, rgba(23, 35, 38, 0.96) 0%, rgba(18, 30, 33, 0.98) 100%) !important;
  }

  button[role="tab"] {
    border-color: rgba(73, 105, 100, 0.94) !important;
    background: rgba(23, 34, 37, 0.9) !important;
  }

  button[role="tab"][aria-selected="true"] {
    border-color: var(--color-tab-active-border) !important;
    background: linear-gradient(180deg, rgba(25, 52, 50, 0.98) 0%, rgba(22, 66, 58, 0.98) 100%) !important;
  }

  .studio-shell .gr-group,
  .studio-shell .gr-box,
  .studio-shell .gr-form,
  .studio-shell .gr-accordion,
  .studio-shell .form {
    border-color: var(--color-border) !important;
    background: transparent !important;
  }

  .studio-shell input,
  .studio-shell textarea,
  .studio-shell select,
  .studio-shell .gr-text-input,
  .studio-shell .gr-textbox,
  .studio-shell .gr-dropdown,
  .studio-shell .gr-number,
  .studio-shell .gr-dataframe,
  .studio-shell .gradio-dataframe {
    background: var(--color-input-bg) !important;
    color: var(--color-input-text) !important;
    border-color: var(--color-input-border) !important;
  }

  .studio-shell input::placeholder,
  .studio-shell textarea::placeholder {
    color: var(--color-placeholder) !important;
  }

  .studio-shell label,
  .studio-shell .block-title,
  .studio-shell .block-label,
  .studio-shell .gr-markdown,
  .studio-shell .gradio-markdown,
  .studio-shell .prose,
  .studio-shell .prose p,
  .studio-shell .prose li,
  .studio-shell .prose strong {
    color: var(--color-label) !important;
  }

  .studio-shell .gr-button-primary {
    background: linear-gradient(180deg, #1c7d6d 0%, #15695b 100%) !important;
    color: var(--color-text-on-primary) !important;
  }

  .studio-shell .gr-button-secondary {
    background: #172429 !important;
    color: #dceee8 !important;
    border-color: #38554f !important;
  }

  .studio-shell table,
  .studio-shell thead,
  .studio-shell tbody,
  .studio-shell tr,
  .studio-shell th,
  .studio-shell td {
    background: var(--color-table-bg) !important;
    color: var(--color-table-text) !important;
    border-color: var(--color-table-border) !important;
  }

  .studio-shell code,
  .studio-shell pre {
    background: var(--color-code-bg) !important;
    color: var(--color-code-text) !important;
    border-color: var(--color-code-border) !important;
  }

  .studio-shell hr {
    border-color: var(--color-hr) !important;
  }
}

/* ── Responsive ───────────────────────────────────────────────────────────── */
@media (max-width: 1120px) {
  .studio-banner__grid,
  .studio-cockpit {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .studio-hero {
    padding: var(--space-6) 22px;
  }

  .studio-hero__title {
    font-size: var(--text-display-sm);
  }

  .studio-hero__meta,
  .studio-kpi-grid {
    grid-template-columns: 1fr;
  }
}

/* ── Accessibility: reduced motion ────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* ── Accessibility: focus indicators ──────────────────────────────────────── */
:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
  border-radius: var(--space-1);
}

button[role="tab"]:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
  box-shadow: 0 0 0 4px rgba(22, 133, 111, 0.18) !important;
}

.studio-shell button:focus-visible,
.studio-shell input:focus-visible,
.studio-shell textarea:focus-visible,
.studio-shell select:focus-visible,
.studio-shell [tabindex]:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
}

@media (prefers-color-scheme: dark) {
  button[role="tab"]:focus-visible {
    box-shadow: 0 0 0 4px rgba(61, 212, 168, 0.2) !important;
  }
}

/* ── Accessibility: skip-to-content link ──────────────────────────────────── */
.studio-skip-link {
  position: fixed !important;
  top: 0 !important;
  left: var(--space-4);
  z-index: 9999;
  padding: var(--space-3) var(--space-5);
  border-radius: var(--space-2);
  background: var(--color-accent-dark);
  color: var(--color-text-on-primary);
  font-size: var(--text-base);
  font-weight: 700;
  text-decoration: none;
  white-space: nowrap;
  transform: translateY(-100%) !important;
}

.studio-skip-link:focus {
  transform: translateY(var(--space-4)) !important;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _loss_plot(history: list[dict]) -> Any:
    """
    Reuse a single named figure on every call (num='nocode_loss') so matplotlib
    never accumulates open figures across training epochs — preventing a memory
    leak that would otherwise grow linearly with epoch count.
    """
    plt = _get_pyplot()
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_metric = [h["val_acc"]    for h in history]

    # clear=True redraws into the existing figure instead of allocating a new one.
    # dpi=130 keeps plots sharp on high-DPI (Retina) displays.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=130,
                                    num="nocode_loss_curves", clear=True)
    ax1.plot(epochs, train_loss, label="Train loss", marker="o", markersize=3)
    ax1.plot(epochs, val_loss,   label="Val loss",   marker="o", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, val_metric, color="steelblue", marker="o", markersize=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy % / MAE")
    ax2.set_title("Validation Metric")
    fig.tight_layout()
    return fig


HISTORY_COLUMNS = [
    "id",
    "timestamp",
    "model",
    "modality",
    "task",
    "mode",
    "val_acc",
    "val_loss",
    "bundle",
]


def _format_eval_summary(summary: str) -> str:
    """Render evaluation text as readable markdown."""
    if not summary or not str(summary).strip():
        return "> ℹ️ Evaluation metrics will appear here after training completes."

    text = str(summary).strip()
    if "precision" in text and "recall" in text and "f1-score" in text:
        headline, _, report_block = text.partition("\n")
        accuracy_match = re.search(r"Validation Accuracy:\s*([0-9.]+)%", text)
        macro_match = re.search(r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text)
        weighted_match = re.search(r"weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text)

        rows = []
        for raw_line in report_block.splitlines():
            line = raw_line.strip()
            if not line or "precision" in line or line.startswith("accuracy"):
                continue
            match = re.match(r"(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", line)
            if not match:
                continue
            label, precision, recall, f1_score, support = match.groups()
            rows.append((label.strip(), precision, recall, f1_score, support))

        chips = []
        if accuracy_match:
            chips.append(("Accuracy", f"{accuracy_match.group(1)}%"))
        if macro_match:
            chips.append(("Macro F1", macro_match.group(3)))
        if weighted_match:
            chips.append(("Weighted F1", weighted_match.group(3)))

        chip_html = "".join(
            f'<div class="studio-report-chip"><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>'
            for label, value in chips
        )
        rows_html = "".join(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(precision)}</td>"
            f"<td>{html.escape(recall)}</td>"
            f"<td>{html.escape(f1_score)}</td>"
            f"<td>{html.escape(support)}</td>"
            "</tr>"
            for label, precision, recall, f1_score, support in rows
        )

        return (
            f'<div class="studio-report">'
            f'<h3>{html.escape(headline.strip())}</h3>'
            f'<div class="studio-report-chip-row">{chip_html}</div>'
            f'<div class="studio-report-table-wrap">'
            f'<table class="studio-report-table">'
            f'<thead><tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>'
            f'</div>'
            f'</div>'
        )

    return text


def _format_status_block(title: str, body: str, *, success: bool = True) -> str:
    """Format action feedback as markdown."""
    cleaned = (body or "").strip()
    icon = "✅" if success else "⚠️"
    if not cleaned:
        return f"*{title} status will appear here.*"
    return f"### {icon} {title}\n\n```text\n{cleaned}\n```"


_DEFAULT_CLASSIFICATION_NOTE = (
    "*Classification plots will appear here for classification or multi-label runs.*"
)
_DEFAULT_EXPLANATION_NOTE = (
    "*Explanation tools will appear here when the current model and modality support them.*"
)
_DEFAULT_DIAGNOSTICS_NOTE = (
    "*Regression and anomaly diagnostics will appear here when they apply to the selected task.*"
)
_DEFAULT_SAMPLE_REVIEW_NOTE = (
    "*Sample review updates after evaluation with misclassifications or clustering structure when available.*"
)

CLASSIFICATION_METRIC_OPTIONS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "balanced_accuracy",
    "auc_ovr",
    "mcc",
]
REGRESSION_METRIC_OPTIONS = [
    "mae",
    "rmse",
    "r2",
    "mape",
]


def tip(key: str, fallback: str = "") -> str:
    return TOOLTIPS.get(key, fallback)


def _list_bundles() -> list[str]:
    """Return absolute paths of saved model bundles, most recent first."""
    if not _OUTPUTS_ROOT.is_dir():
        return []
    bundles = []
    for entry in sorted(_OUTPUTS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if entry.is_dir() and (entry / "labels.json").exists():
            bundles.append(str(entry))
    return bundles


def _section_intro(title: str, desc: str) -> str:
    """Reusable HTML block for section headers inside cards/groups."""
    return f'<div class="studio-section-intro"><h3>{html.escape(title)}</h3><p>{html.escape(desc)}</p></div>'


_TUTORIAL_MODALITIES = ["image", "tabular", "text", "audio", "timeseries", "video"]

_TUTORIAL_META: dict[str, dict] = {
    "image": {
        "title":    "MNIST handwritten digits",
        "desc":     "10 digit classes · 300 images each · downloads ~12 MB on first use",
        "loaded":   (
            "### Image tutorial loaded — MNIST\n\n"
            "300 PNG images per digit class (0–9). "
            "**Suggested flow:** preview class balance → train a small CNN → evaluate with GradCAM → export.\n\n"
            f"**Dataset**: `{_BUILTIN_IMAGE_TUTORIAL_PATH}`"
        ),
    },
    "tabular": {
        "title":    "Iris species classification",
        "desc":     "3 species · 150 rows · 4 numeric features · no download needed",
        "loaded":   (
            "### Tabular tutorial loaded — Iris\n\n"
            "Classic 3-class dataset: predict iris species from sepal/petal measurements. "
            "**Suggested flow:** check feature distributions → train a gradient-boosted model → inspect SHAP values.\n\n"
            f"**Dataset**: `{_BUILTIN_TABULAR_TUTORIAL_PATH / 'iris.csv'}`"
        ),
    },
    "text": {
        "title":    "20 Newsgroups topic classification",
        "desc":     "3 topic categories · ~900 documents · downloads via scikit-learn on first use",
        "loaded":   (
            "### Text tutorial loaded — 20 Newsgroups\n\n"
            "Classify news posts into Space, Hockey, or Politics. "
            "**Suggested flow:** preview class balance → train a text classifier → evaluate confusion matrix.\n\n"
            f"**Dataset**: `{_BUILTIN_TEXT_TUTORIAL_PATH / 'newsgroups.csv'}`"
        ),
    },
    "audio": {
        "title":    "Spoken digit recognition",
        "desc":     "Digits 0–9 · ~50 WAV clips per class · downloads ~10 MB on first use",
        "loaded":   (
            "### Audio tutorial loaded — Spoken Digits\n\n"
            "Classify short spoken digit recordings (Free Spoken Digit Dataset). "
            "**Suggested flow:** preview waveform distribution → train a 1-D CNN → evaluate accuracy per digit.\n\n"
            f"**Dataset**: `{_BUILTIN_AUDIO_TUTORIAL_PATH}`"
        ),
    },
    "timeseries": {
        "title":    "Synthetic signal classification",
        "desc":     "3 signal types · 200 series each · 50 timesteps · generated locally",
        "loaded":   (
            "### Timeseries tutorial loaded — Synthetic Signals\n\n"
            "Classify sinusoidal vs. ramp signals with added noise — a minimal time-series benchmark. "
            "**Suggested flow:** preview signal statistics → train a 1-D CNN → evaluate per-class accuracy.\n\n"
            f"**Dataset**: `{_BUILTIN_TIMESERIES_TUTORIAL_PATH / 'timeseries.csv'}`"
        ),
    },
    "video": {
        "title":    "Synthetic shape recognition",
        "desc":     "3 shape classes · 30 MP4 clips each · generated locally using OpenCV",
        "loaded":   (
            "### Video tutorial loaded — Synthetic Shapes\n\n"
            "Classify short video clips by moving shape (circle / square / triangle). "
            "**Suggested flow:** preview frame statistics → train a video CNN → evaluate confusion matrix.\n\n"
            f"**Dataset**: `{_BUILTIN_VIDEO_TUTORIAL_PATH}`"
        ),
    },
}


def _tutorial_markdown(modality: str = "image", loaded: bool = False) -> str:
    meta = _TUTORIAL_META.get(modality, _TUTORIAL_META["image"])
    if loaded:
        return meta["loaded"]
    return (
        f"### Tutorial: {meta['title']}\n\n"
        f"{meta['desc']}\n\n"
        "Click **Load Tutorial** to download the dataset (if needed) and pre-fill all settings."
    )


def _build_evaluation_notes(
    *,
    task: str,
    modality: str,
    model_name: str,
    sklearn_model: bool,
    cm_fig,
    roc_fig,
    gcam_fig,
    shap_fig,
    reg_fig,
    anomaly_fig,
    mis_img_fig,
    mis_df_val,
    tsne_fig,
) -> tuple[str, str, str, str]:
    """Create guided notes for each Evaluate section."""
    if task in {"classification", "multi-label"}:
        class_lines = ["### Classification views"]
        class_lines.append("- Confusion matrix is ready." if cm_fig is not None else "- Confusion matrix is not available for this run.")
        class_lines.append("- ROC curves are ready." if roc_fig is not None else "- ROC curves were not generated for this run.")
    else:
        class_lines = [
            "### Classification views",
            f"- This section is not the primary focus for **{task}** runs.",
        ]

    explain_lines = ["### Explanations"]
    if modality in {"image", "audio", "video"} and not sklearn_model:
        explain_lines.append("- Grad-CAM is ready." if gcam_fig is not None else "- Grad-CAM was not generated for this run.")
    else:
        explain_lines.append(f"- Grad-CAM does not apply to **{modality}** runs with the current model.")
    if modality == "tabular":
        explain_lines.append("- SHAP feature importance is ready." if shap_fig is not None else "- SHAP was not available for this model or run.")
    else:
        explain_lines.append("- SHAP is mainly used for tabular runs.")

    diagnostics_lines = ["### Prediction diagnostics"]
    if task == "regression":
        diagnostics_lines.append("- Regression diagnostics are ready." if reg_fig is not None else "- Regression diagnostics were not generated.")
    else:
        diagnostics_lines.append(f"- Regression diagnostics do not apply to **{task}** runs.")
    if task == "anomaly" or model_name == "Autoencoder":
        diagnostics_lines.append("- Anomaly reconstruction analysis is ready." if anomaly_fig is not None else "- Anomaly diagnostics were not generated.")
    else:
        diagnostics_lines.append("- Anomaly diagnostics are reserved for anomaly or autoencoder workflows.")

    review_lines = ["### Sample review"]
    if mis_df_val is not None or mis_img_fig is not None:
        review_lines.append("- Misclassified examples are available below.")
    elif task in {"classification", "multi-label", "regression"} and model_name != "Autoencoder":
        review_lines.append("- No sample review artifacts were produced for this run.")
    else:
        review_lines.append("- Misclassification review is not central to this task.")
    if tsne_fig is not None:
        review_lines.append("- Clustering structure is available in the t-SNE plot.")
    elif task == "clustering" or model_name == "Autoencoder":
        review_lines.append("- Clustering/t-SNE was skipped or unavailable for this run.")
    else:
        review_lines.append("- t-SNE visualisation is reserved for clustering-style runs.")

    return (
        "\n".join(class_lines),
        "\n".join(explain_lines),
        "\n".join(diagnostics_lines),
        "\n".join(review_lines),
    )


def _format_eval_kpis(summary: str, history: list[dict], task: str) -> str:
    """Render a compact KPI strip for the Evaluate tab."""
    metric_label = "Validation metric"
    metric_value = "Not available"
    accuracy_match = re.search(r"Validation Accuracy:\s*([0-9.]+)%", summary or "")
    if accuracy_match:
        metric_label = "Validation accuracy"
        metric_value = f"{accuracy_match.group(1)}%"
    elif history:
        last_metric = history[-1].get("val_acc")
        if last_metric is not None:
            metric_value = str(last_metric)

    val_loss = "Not available"
    if history and history[-1].get("val_loss") is not None:
        val_loss = str(history[-1]["val_loss"])

    cards = [
        ("Run type", f"{task.replace('_', ' ').title()} run"),
        (metric_label, metric_value),
        ("Validation loss", val_loss),
    ]
    card_html = "".join(
        f'<div class="studio-kpi"><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>'
        for label, value in cards
    )
    return f'<div class="studio-kpi-grid studio-kpi-grid--compact">{card_html}</div>'


def _metric_display_value(metric_name: str, value):
    if value is None:
        return None
    if metric_name in {"accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy"}:
        return round(float(value) * 100.0, 2)
    return round(float(value), 4)


# ── Dashboard helpers ─────────────────────────────────────────────────────

def _build_dashboard_kpi(
    task: str, modality: str, history: list[dict],
    metrics_payload: dict, model_name: str, training_mode: str,
) -> str:
    """Build the KPI strip HTML for the executive dashboard."""
    task_label = (task or "unknown").replace("_", " ").title()

    # Primary metric
    if task in ("classification", "multi-label"):
        metric_val = metrics_payload.get("accuracy")
        if metric_val is not None:
            metric_val = f"{float(metric_val) * 100:.1f}%"
            raw = float(metrics_payload["accuracy"])
            color = "#16856f" if raw >= 0.8 else "#c4820e" if raw >= 0.6 else "#c42b1c"
        else:
            metric_val = f"{history[-1]['val_acc']:.1f}%" if history else "—"
            color = "var(--color-text-primary)"
        metric_label = "Accuracy"
    elif task == "regression":
        metric_val = metrics_payload.get("mae")
        metric_label = "MAE"
        metric_val = f"{float(metric_val):.4f}" if metric_val else "—"
        color = "var(--color-text-primary)"
    elif task == "clustering":
        metric_val = metrics_payload.get("silhouette")
        metric_label = "Silhouette"
        metric_val = f"{float(metric_val):.3f}" if metric_val else "—"
        color = "var(--color-text-primary)"
    else:
        metric_val, metric_label, color = "—", "Metric", "var(--color-text-primary)"

    # Val loss
    val_loss = f"{history[-1]['val_loss']:.4f}" if history else "—"

    # Epochs
    n_epochs = len(history) if history else 0
    epoch_text = f"{n_epochs}"

    cards = [
        ("Task", f"{task_label} · {modality}"),
        (metric_label, f'<span style="color:{color};font-size:1.5rem;">{metric_val}</span>'),
        ("Val Loss", val_loss),
        ("Model", f"{model_name} ({training_mode})"),
    ]
    card_html = "".join(
        f'<div class="studio-kpi"><strong>{html.escape(label)}</strong><span>{value}</span></div>'
        for label, value in cards
    )
    return f'<div class="studio-kpi-grid">{card_html}</div>'


def _pick_primary_chart(task, cm_fig, reg_fig, tsne_fig, anomaly_fig):
    """Return the most relevant diagnostic chart for the task."""
    if task in ("classification", "multi-label"):
        return cm_fig
    if task == "regression":
        return reg_fig
    if task == "clustering":
        return tsne_fig
    if task == "anomaly":
        return anomaly_fig
    return cm_fig or reg_fig or tsne_fig or anomaly_fig


def _build_action_items(
    task: str, history: list[dict], metrics_payload: dict, model_name: str,
) -> str:
    """Generate actionable recommendations based on training results."""
    items = []

    if not history:
        return "### Recommendations\n\n*Train a model to see recommendations.*"

    # Check for overfitting
    if len(history) >= 3:
        last = history[-1]
        if last["val_loss"] > last["train_loss"] * 1.3:
            gap = (last["val_loss"] - last["train_loss"]) / max(last["train_loss"], 1e-6) * 100
            items.append(
                f"**Possible overfitting** — validation loss is {gap:.0f}% higher than training loss. "
                "Consider more training data, stronger regularization (dropout), or data augmentation."
            )

    # Check for underfitting
    if task in ("classification", "multi-label"):
        acc = metrics_payload.get("accuracy")
        if acc is not None and float(acc) < 0.7:
            items.append(
                f"**Low accuracy ({float(acc)*100:.1f}%)** — the model may be underfitting. "
                "Try a larger architecture, more epochs, or a higher learning rate."
            )
        f1 = metrics_payload.get("f1_macro")
        if acc and f1 and abs(float(acc) - float(f1)) > 0.15:
            items.append(
                "**Accuracy/F1 gap** — high accuracy but lower F1 suggests class imbalance. "
                "Check the confusion matrix for classes the model ignores."
            )

    # Check early stopping
    if len(history) < 10 and len(history) >= 3:
        last_losses = [h["val_loss"] for h in history[-3:]]
        if all(last_losses[i] >= last_losses[i-1] for i in range(1, len(last_losses))):
            items.append(
                f"**Early stopping likely triggered** at epoch {len(history)}. "
                "The model converged quickly — try a lower learning rate for finer tuning."
            )

    # Check for good results
    if task in ("classification", "multi-label"):
        acc = metrics_payload.get("accuracy")
        if acc and float(acc) >= 0.9:
            items.append(
                f"**Strong performance ({float(acc)*100:.1f}% accuracy)** — "
                "verify on held-out data and check the confusion matrix for edge cases before deploying."
            )

    # Always suggest next steps
    items.append(
        "**Next steps** — review the Evaluate tab for detailed diagnostics, "
        "try the model on new inputs in Try Your Model, and export via Export & History."
    )

    lines = ["### Recommendations", ""]
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _slug_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return cleaned or "project"


def suggest_bundle_name(dpath: str, mod: str, model: str, task: str, current_name: str) -> str:
    current = (current_name or "").strip()
    if current and current not in {"my_model", "project"}:
        return current

    source = ""
    if dpath:
        source = Path(str(dpath)).stem
    parts = [
        _slug_part(source) if source else _slug_part(mod or "dataset"),
        _slug_part(model or "model"),
        _slug_part(task or "task"),
    ]
    return "_".join(parts[:3])


def _workspace_status_markdown(
    dpath: str,
    mod: str,
    label: str,
    model: str,
    task: str,
    bundle_name: str,
    bundle_path: str,
) -> tuple[str, str]:
    dataset_ready = bool(str(dpath or "").strip())
    model_ready = bool(str(model or "").strip())
    trained = bool(str(bundle_path or "").strip())
    effective_bundle_name = suggest_bundle_name(dpath, mod, model, task, bundle_name)

    dataset_status = "Ready" if dataset_ready else "Waiting"
    model_status = "Ready" if model_ready else "Waiting"
    train_status = "Complete" if trained else ("Ready to run" if dataset_ready and model_ready else "Blocked")

    snapshot = [
        "### Workspace cockpit",
        "",
        f"- **Dataset**: {dataset_status}",
        f"- **Modality**: `{mod or 'not set'}`",
        f"- **Label column**: `{label or 'not set'}`",
        f"- **Model**: `{model or 'not selected'}`",
        f"- **Task**: `{task or 'not selected'}`",
        f"- **Bundle name**: `{effective_bundle_name or 'not set'}`",
        f"- **Training**: {train_status}",
    ]
    if trained:
        snapshot.append(f"- **Latest bundle**: `{bundle_path}`")

    if not dataset_ready:
        next_step = (
            "### Next step\n\n"
            "Upload a dataset or paste a local path, then click **Preview Dataset**. "
            "The app will infer columns and help set up the workflow for you."
        )
    elif not model_ready:
        next_step = (
            "### Next step\n\n"
            "Move to **Model**, keep the suggested defaults if you want a quick start, "
            "and adjust only the task or architecture you actually want to explore."
        )
    elif not trained:
        next_step = (
            "### Next step\n\n"
            "Go to **Train** and run a short baseline first. After that, review **Evaluate** "
            "and try the saved bundle in **Try Your Model**."
        )
    else:
        next_step = (
            "### Next step\n\n"
            "Your latest bundle is ready. Review the evaluation dashboard, try a prediction, "
            "or export a serving bundle from **Export & History**."
        )

    return "\n".join(snapshot), next_step


def _project_mode_default() -> str:
    return str(load_project_state().get("project_mode", "Guided") or "Guided")


def _workflow_mode_updates(project_mode: str):
    mode = project_mode or "Guided"
    is_beginner = mode == "Beginner"
    is_guided = mode == "Guided"
    is_advanced = mode == "Advanced"
    return (
        gr.update(visible=True),             # split_cleaning_group
        gr.update(visible=not is_beginner),  # advanced_training_group
        gr.update(visible=is_advanced),      # sequence_options_group
        gr.update(visible=is_advanced),      # sklearn_options_group
        gr.update(visible=True),             # column_mapping_group
        gr.update(visible=not is_beginner),  # hyperparams_group
        gr.update(visible=True),             # model_selection_group
        gr.update(visible=not is_beginner),  # evaluation_report_group
        gr.update(visible=not is_beginner),  # export_utilities_group
        gr.update(visible=not is_beginner),  # history_group
        gr.update(visible=True),             # explanation_group_top
    )


def _why_this_matters_markdown(project_mode: str, mod: str, model: str, task: str, recommendation: dict | None = None) -> str:
    mode = project_mode or "Guided"
    recommendation = recommendation or {}
    if not model:
        return (
            "### Why this matters\n\n"
            f"**{mode} mode** is active. Start by previewing the dataset so the workspace can recommend a safe baseline. "
            "The goal of the first pass is not perfect tuning. It is to confirm the data, task, and model family line up."
        )

    rec_model = recommendation.get("model_name")
    rec_reason = recommendation.get("rationale", [])
    next_step = "Run a short baseline next, then trust the evaluation report before changing many settings."
    lines = ["### Why this matters", ""]
    if rec_model:
        lines.append(f"- The current guided baseline favours **{rec_model}** for **{mod}** data.")
    elif model:
        lines.append(f"- You are working with **{model}** for a **{task}** task.")
    if rec_reason:
        lines.append(f"- {rec_reason[0]}")
    if task == "classification":
        lines.append("- Accuracy is useful, but class imbalance means F1 and per-class review matter too.")
    elif task == "regression":
        lines.append("- Focus on MAE/RMSE and residual patterns, not just a single headline score.")
    elif task == "clustering":
        lines.append("- Clustering quality is about whether the learned structure is stable and interpretable, not just whether a chart looks separated.")
    if mode == "Beginner":
        lines = lines[:4]
    elif mode == "Advanced":
        lines.append("- Advanced mode keeps more controls visible, so change one variable at a time if you want clean comparisons.")
    lines.append(f"- {next_step}")
    return "\n".join(lines)


def _apply_recommendation_payload(modality_name: str, rec: dict | None):
    rec = rec or {}
    model_choices = get_models(modality_name, rec.get("training_mode", "fine-tune"))
    task_choices = get_compatible_tasks(modality_name)
    if modality_name in {"tabular", "timeseries"} and "regression" not in task_choices:
        task_choices = task_choices + ["regression"]
    model_value = rec.get("model_name")
    if model_value not in model_choices and model_choices:
        model_value = model_choices[0]
    task_value = rec.get("task")
    if task_value not in task_choices and task_choices:
        task_value = task_choices[0]
    return (
        gr.update(value=rec.get("training_mode", "fine-tune")),
        gr.update(choices=model_choices, value=model_value),
        gr.update(choices=task_choices, value=task_value),
        gr.update(value=rec.get("augmentation", "light")),
        gr.update(value=rec.get("image_size", 160)),
        gr.update(value=rec.get("batch_size", 16)),
        gr.update(value=rec.get("epochs", 8)),
        gr.update(value=rec.get("dropout", 0.3)),
        gr.update(value=rec.get("scheduler", "cosine")),
        gr.update(value=bool(rec.get("use_class_weights", False))),
    )


def _suggest_tabular_example(dpath: str, mod: str, label_col: str, current_value: str) -> str:
    if (current_value or "").strip():
        return current_value
    if mod != "tabular":
        return current_value or ""
    try:
        df = _read_structured_preview(dpath)
        if df is None or df.empty:
            return current_value or ""
        row = df.iloc[0].to_dict()
        if label_col in row:
            row.pop(label_col, None)
        if not row:
            return current_value or ""
        return "{\n" + ",\n".join(
            f'  "{k}": {repr(v) if isinstance(v, str) else v}'
            for k, v in row.items()
        ) + "\n}"
    except Exception:
        return current_value or ""


def _suggest_timeseries_example(dpath: str, mod: str, label_col: str, time_col: str, current_value: str) -> str:
    if (current_value or "").strip():
        return current_value
    if mod != "timeseries":
        return current_value or ""
    try:
        df = _read_structured_preview(dpath)
        if df is None or df.empty:
            return current_value or ""
        feature_cols = [c for c in df.columns if c not in {label_col, time_col}]
        if not feature_cols:
            return current_value or ""
        window = df[feature_cols].head(6).to_dict(orient="records")
        lines = []
        for row in window:
            entries = ", ".join(
                f'"{k}": {repr(v) if isinstance(v, str) else float(v) if v is not None else 0}'
                for k, v in row.items()
            )
            lines.append(f"  {{{entries}}}")
        return "[\n" + ",\n".join(lines) + "\n]"
    except Exception:
        return current_value or ""


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:   return f"ETA: {int(seconds)}s"
    if seconds < 3600: return f"ETA: {int(seconds/60)}m {int(seconds%60)}s"
    return f"ETA: {seconds/3600:.1f}h"


def _phase_eta_text(phase: str) -> str:
    phase_name = (phase or "working").replace("_", " ")
    return f"ETA: {phase_name}"


def _format_live_training_metrics_card(metrics: dict | None = None) -> str:
    metrics = metrics or {}

    def _value(key: str, default: str = "—") -> str:
        value = metrics.get(key)
        if value in (None, ""):
            return default
        return str(value)

    cards = [
        ("Phase", _value("phase", "Waiting")),
        ("Epoch", _value("epoch_text")),
        ("Train loss", _value("train_loss")),
        ("Val loss", _value("val_loss")),
        ("Metric", _value("metric")),
        ("Learning rate", _value("lr")),
    ]
    card_html = "".join(
        f'<div class="studio-kpi"><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>'
        for label, value in cards
    )
    return f'<div class="studio-kpi-grid studio-kpi-grid--compact">{card_html}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    # Data
    modality, data_path, text_col, label_col, feature_cols, time_col,
    window_size, n_frames, image_size, sample_rate, audio_image_size, audio_n_mels, augmentation,
    val_split, use_random_subset, subset_percent, subset_seed, use_data_cleaning, tabular_missing_strategy, tabular_clip_outliers, tabular_scaling,
    text_lowercase, text_strip_urls, text_strip_punctuation, text_remove_stopwords,
    text_deduplicate, text_apply_stemming, text_apply_lemmatization, text_use_ngrams,
    timeseries_sort_by_time, timeseries_fill_strategy,
    image_verify_files, image_aug_flip, image_aug_vertical, image_aug_rotation,
    image_aug_color, image_aug_gray, image_aug_perspective, image_normalization, image_force_grayscale,
    audio_verify_files, audio_normalize_waveform, audio_aug_noise, audio_aug_shift,
    audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask, video_verify_files,
    # Model
    training_mode, model_name, task,
    # Hyperparams
    epochs, lr, batch_size, dropout, optimizer, scheduler_name, use_amp,
    hidden_size, num_layers,
    n_estimators, max_depth, C_param, max_iter, lr_xgb,
    n_clusters,
    use_class_weights, checkpoint_every,
    # Export
    bundle_name,
    progress=gr.Progress(track_tqdm=False),
):
    _NONE     = gr.update(value=None)
    _DEFAULT_TELEMETRY_CARD = _format_live_training_metrics_card()

    telemetry = {
        "phase": "Waiting",
        "status_log": "Initialising…",
        "eta_text": "ETA: estimating…",
        "epoch_text": "—",
        "latest_metrics": {},
        "progress_label": "Preparing run",
    }

    # 22-tuple: original training outputs + live telemetry card + evaluation KPI + guidance notes
    def _emit(status=None, loss_fig=_NONE, eta=None, telemetry_card=None, eval_sum="", cm=_NONE, roc=_NONE,
              tsne=_NONE, gcam=_NONE, shap=_NONE, mis_img=_NONE, mis_df=_NONE,
              bundle="", arch=_NONE,
              train_summary="", reg_plot=_NONE, anomaly_plot=_NONE,
              eval_kpis="",
              class_note=_DEFAULT_CLASSIFICATION_NOTE,
              explanation_note_val=_DEFAULT_EXPLANATION_NOTE,
              diagnostics_note_val=_DEFAULT_DIAGNOSTICS_NOTE,
              sample_review_note_val=_DEFAULT_SAMPLE_REVIEW_NOTE,
              text_shap=_NONE,
              dash_kpi=_NONE, dash_primary=_NONE, dash_explain_plot=_NONE,
              dash_explain_html=_NONE, dash_loss=_NONE, dash_actions=""):
        status_val = status if status is not None else telemetry["status_log"]
        eta_val = eta if eta is not None else telemetry["eta_text"]
        telemetry_val = telemetry_card if telemetry_card is not None else _format_live_training_metrics_card({
            "phase": telemetry.get("phase", "Waiting"),
            "epoch_text": telemetry.get("epoch_text", "—"),
            "train_loss": telemetry.get("latest_metrics", {}).get("train_loss", "—"),
            "val_loss": telemetry.get("latest_metrics", {}).get("val_loss", "—"),
            "metric": telemetry.get("latest_metrics", {}).get("metric", "—"),
            "lr": telemetry.get("latest_metrics", {}).get("lr", "—"),
        })
        return (status_val, loss_fig, eta_val, telemetry_val, eval_sum, cm, roc, tsne, gcam, shap,
                mis_img, mis_df, bundle, arch,
                train_summary, reg_plot, anomaly_plot, eval_kpis,
                class_note, explanation_note_val, diagnostics_note_val,
                sample_review_note_val, text_shap,
                dash_kpi, dash_primary, dash_explain_plot, dash_explain_html,
                dash_loss, dash_actions)

    def _set_phase(phase: str, *, eta_text: str | None = None, status_line: str | None = None) -> None:
        telemetry["phase"] = phase
        telemetry["progress_label"] = phase
        telemetry["eta_text"] = eta_text or _phase_eta_text(phase)
        if status_line:
            telemetry["status_log"] = status_line

    def _update_epoch_metrics(*, epoch_text=None, train_loss=None, val_loss=None, metric=None, lr_value=None):
        if epoch_text is not None:
            telemetry["epoch_text"] = epoch_text
        metrics = telemetry["latest_metrics"]
        if train_loss is not None:
            metrics["train_loss"] = str(train_loss)
        if val_loss is not None:
            metrics["val_loss"] = str(val_loss)
        if metric is not None:
            metrics["metric"] = str(metric)
        if lr_value is not None:
            metrics["lr"] = str(lr_value)

    # ── Pre-flight validation ──────────────────────────────────────────────────
    preflight_errors: list[str] = []
    if not data_path or not str(data_path).strip():
        preflight_errors.append("❌ **No data path provided.** Upload a dataset or enter a path.")
    if not model_name:
        preflight_errors.append("❌ **No model selected.** Choose an architecture in the Model tab.")
    if task == "anomaly" and model_name != "Autoencoder":
        preflight_errors.append(
            "❌ **Anomaly detection requires the Autoencoder model.** "
            f"Selected model '{model_name}' does not support anomaly detection. "
            "Switch to Autoencoder or change the task."
        )
    if preflight_errors:
        yield _emit("\n".join(preflight_errors))
        return

    # Sanitize bundle name — allow only letters, digits, underscores, hyphens
    safe_bundle_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(bundle_name).strip()) or "my_model"

    log_lines: list[str] = []
    _LOG_CHAR_BUDGET = 40_000  # ~40 KB of visible text; enough for 200+ epochs

    def log(msg: str) -> str:
        nonlocal log_lines
        log_lines.append(msg)
        # Cap by character budget rather than line count — some messages (e.g.
        # cleaning reports, tracebacks) contain many embedded newlines and would
        # blow through a per-call cap without ever being trimmed.
        combined = "\n".join(log_lines)
        if len(combined) > _LOG_CHAR_BUDGET:
            # Drop the oldest half and rebuild
            log_lines = log_lines[len(log_lines) // 2 :]
            combined  = "\n".join(log_lines)
        telemetry["status_log"] = combined
        return combined

    status = log("Initialising…")
    _set_phase("initialising", eta_text="ETA: estimating…")
    yield _emit(status)

    save_project_state({
        **load_project_state(),
        "preprocessing": {
            "augmentation": augmentation,
            "val_split": val_split,
            "tabular_missing_strategy": tabular_missing_strategy,
            "tabular_clip_outliers": bool(tabular_clip_outliers),
            "tabular_scaling": tabular_scaling,
            "text_lowercase": bool(text_lowercase),
            "text_strip_urls": bool(text_strip_urls),
            "text_strip_punctuation": bool(text_strip_punctuation),
            "text_remove_stopwords": bool(text_remove_stopwords),
            "text_deduplicate": bool(text_deduplicate),
            "text_apply_stemming": bool(text_apply_stemming),
            "text_apply_lemmatization": bool(text_apply_lemmatization),
            "text_use_ngrams": bool(text_use_ngrams),
            "image_size": int(image_size),
            "image_normalization": image_normalization,
            "image_force_grayscale": bool(image_force_grayscale),
        },
    })

    extra_data: dict = {}
    history:    list = []
    final_model      = None
    prep:       dict = {}
    classes:    list = []
    clustering_result = None
    eval_summary_text = ""
    metrics_payload: dict = {}
    stopped_early_flag = False

    try:
        # ── 1. Load data ──────────────────────────────────────────────────────
        _set_phase("loading data", eta_text="ETA: loading data")
        progress(0.05, desc="Loading data…")
        status = log(f"Loading {modality} data…")
        if modality in {"tabular", "text", "timeseries"} and bool(use_random_subset) and float(subset_percent) < 100:
            status = log(
                f"Using a random subset of {float(subset_percent):.2f}% of the structured dataset "
                f"(seed {int(subset_seed)})."
            )
        yield _emit(status)

        bs = int(batch_size)

        _val_split = float(val_split)

        if modality == "image":
            from modalities.image import load_image_data
            image_aug_options = {
                "horizontal_flip": bool(image_aug_flip),
                "vertical_flip": bool(image_aug_vertical),
                "rotation": bool(image_aug_rotation),
                "color_jitter": bool(image_aug_color),
                "grayscale": bool(image_aug_gray),
                "perspective": bool(image_aug_perspective),
            }
            train_loader, val_loader, classes, prep, val_samples = load_image_data(
                data_path, mode=training_mode, batch_size=bs,
                val_split=_val_split, augmentation=augmentation,
                verify_files=bool(image_verify_files),
                image_size=int(image_size),
                augmentation_options=image_aug_options,
                normalization_preset=str(image_normalization),
                force_grayscale=bool(image_force_grayscale),
            )
            extra_data["val_samples"] = val_samples

        elif modality == "text":
            from modalities.text import load_text_data
            text_cleaning = {
                "lowercase": bool(text_lowercase),
                "strip_urls": bool(text_strip_urls),
                "strip_punctuation": bool(text_strip_punctuation),
                "remove_stopwords": bool(text_remove_stopwords),
                "deduplicate": bool(text_deduplicate),
                "apply_stemming": bool(text_apply_stemming),
                "apply_lemmatization": bool(text_apply_lemmatization),
                "use_ngrams": bool(text_use_ngrams),
            }
            if any(text_cleaning.values()):
                from data_pipeline.cleaning import clean_text_dataframe
                from data_pipeline.io_utils import read_structured_file

                raw_df = read_structured_file(data_path)
                if bool(use_random_subset) and float(subset_percent) < 100:
                    from data_pipeline.io_utils import apply_random_subset
                    raw_df, sampled_rows = apply_random_subset(
                        raw_df,
                        enabled=True,
                        subset_percent=float(subset_percent),
                        subset_seed=int(subset_seed),
                    )
                    status = log(f"  ✓ Sampled {sampled_rows} text rows before cleaning")
                    yield _emit(status)
                cleaned_df, clean_report = clean_text_dataframe(
                    raw_df,
                    text_col=text_col,
                    lowercase=bool(text_lowercase),
                    strip_urls=bool(text_strip_urls),
                    strip_punctuation=bool(text_strip_punctuation),
                    remove_stopwords=bool(text_remove_stopwords),
                    deduplicate=bool(text_deduplicate),
                    apply_stemming=bool(text_apply_stemming),
                    apply_lemmatization=bool(text_apply_lemmatization),
                )
                status = log("  ✓ Text cleaning:\n" + clean_report["summary"])
                yield _emit(status)
                _cleaning_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                cleaned_df.to_csv(_cleaning_tmp.name, index=False)
                _cleaning_tmp.close()
                atexit.register(os.unlink, _cleaning_tmp.name)
                data_path = _cleaning_tmp.name

            train_loader, val_loader, classes, prep, vocab_size, val_texts = load_text_data(
                data_path, text_col=text_col, label_col=label_col,
                model_name=model_name, batch_size=bs,
                max_length=TEXT_DEFAULTS["max_length"],
                val_split=_val_split,
                cleaning_options=text_cleaning,
                subset_percent=float(subset_percent) if bool(use_random_subset) else 100.0,
                subset_seed=int(subset_seed))
            extra_data.update({"vocab_size": vocab_size, "val_texts": val_texts})

        elif modality == "tabular":
            from modalities.tabular import load_tabular_data
            # Optional data cleaning
            _cleaning_tmp = None
            if use_data_cleaning:
                from data_pipeline.cleaning import clean_dataframe, cleaning_report_markdown
                from data_pipeline.io_utils import read_structured_file
                raw_df = read_structured_file(data_path)
                if bool(use_random_subset) and float(subset_percent) < 100:
                    from data_pipeline.io_utils import apply_random_subset
                    raw_df, sampled_rows = apply_random_subset(
                        raw_df,
                        enabled=True,
                        subset_percent=float(subset_percent),
                        subset_seed=int(subset_seed),
                    )
                    status = log(f"  ✓ Sampled {sampled_rows} tabular rows before cleaning")
                    yield _emit(status)
                cleaned_df, clean_report = clean_dataframe(
                    raw_df,
                    label_col,
                    strategy=str(tabular_missing_strategy),
                    clip_outliers=bool(tabular_clip_outliers),
                )
                status = log("  ✓ Data cleaning:\n" + cleaning_report_markdown(clean_report))
                yield _emit(status)
                _cleaning_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                cleaned_df.to_csv(_cleaning_tmp.name, index=False)
                _cleaning_tmp.close()
                atexit.register(os.unlink, _cleaning_tmp.name)
                data_path = _cleaning_tmp.name

            train_loader, val_loader, classes, prep, input_size, \
                X_train, y_train, X_val, y_val = load_tabular_data(
                    data_path, label_col=label_col, feature_cols=feature_cols, batch_size=bs,
                    val_split=_val_split, augmentation=augmentation,
                    scaling_strategy=str(tabular_scaling),
                    subset_percent=float(subset_percent) if bool(use_random_subset) else 100.0,
                    subset_seed=int(subset_seed))
            extra_data.update({"input_size": input_size,
                                "X_train": X_train, "y_train": y_train,
                                "X_val":   X_val,   "y_val":   y_val})

            # Column type inference summary
            try:
                from data_pipeline.type_inference import infer_column_types, suggest_features_and_label
                from data_pipeline.io_utils import read_structured_file
                df_peek = read_structured_file(data_path, nrows=500)
                col_types    = infer_column_types(df_peek)
                suggestions  = suggest_features_and_label(df_peek)
                if suggestions.get("warnings"):
                    for w in suggestions["warnings"]:
                        status = log(f"  ⚠️ {w}")
                    yield _emit(status)
            except Exception:
                pass

        elif modality == "timeseries":
            from modalities.timeseries import load_timeseries_data
            if bool(timeseries_sort_by_time) or str(timeseries_fill_strategy) != "none":
                from data_pipeline.cleaning import clean_timeseries_dataframe
                from data_pipeline.io_utils import read_structured_file

                raw_df = read_structured_file(data_path)
                cleaned_df, clean_report = clean_timeseries_dataframe(
                    raw_df,
                    label_col=label_col,
                    time_col=time_col if time_col else None,
                    sort_by_time=bool(timeseries_sort_by_time),
                    fill_strategy=str(timeseries_fill_strategy),
                )
                status = log("  ✓ Time-series cleaning:\n" + clean_report["summary"])
                yield _emit(status)
                _cleaning_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                cleaned_df.to_csv(_cleaning_tmp.name, index=False)
                _cleaning_tmp.close()
                atexit.register(os.unlink, _cleaning_tmp.name)
                data_path = _cleaning_tmp.name

            train_loader, val_loader, classes, prep, input_size, \
                X_train, y_train, X_val, y_val = load_timeseries_data(
                data_path, label_col=label_col,
                feature_cols=feature_cols,
                time_col=time_col if time_col else None,
                window_size=int(window_size), batch_size=bs,
                val_split=_val_split, augmentation=augmentation, task=task,
                subset_percent=float(subset_percent) if bool(use_random_subset) else 100.0,
                subset_seed=int(subset_seed))
            extra_data.update({
                "input_size": input_size,
                "sklearn_input_size": prep.get("sklearn_input_size", X_train.shape[1] if len(X_train.shape) > 1 else input_size),
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
            })

        elif modality == "audio":
            from modalities.audio import load_audio_data
            audio_aug_options = {
                "noise": bool(audio_aug_noise),
                "time_shift": bool(audio_aug_shift),
                "gain_jitter": bool(audio_aug_gain),
                "time_mask": bool(audio_aug_time_mask),
                "freq_mask": bool(audio_aug_freq_mask),
            }
            train_loader, val_loader, classes, prep = load_audio_data(
                data_path, batch_size=bs, sample_rate=int(sample_rate),
                val_split=_val_split, augmentation=augmentation,
                verify_files=bool(audio_verify_files),
                normalize_waveform=bool(audio_normalize_waveform),
                image_size=int(audio_image_size),
                n_mels=int(audio_n_mels),
                augmentation_options=audio_aug_options,
            )

        elif modality == "video":
            from modalities.video import load_video_data
            train_loader, val_loader, classes, prep = load_video_data(
                data_path, n_frames=int(n_frames),
                batch_size=VIDEO_DEFAULTS["batch_size"],
                val_split=_val_split,
                verify_files=bool(video_verify_files))

        num_classes = len(classes)
        status = log(f"  ✓ {num_classes} classes: {', '.join(classes)}")
        yield _emit(status)

        # ── 2. Build model ────────────────────────────────────────────────────
        _set_phase("building model", eta_text="ETA: building model")
        progress(0.12, desc="Building model…")
        status = log(f"Building {model_name} ({training_mode})…")
        yield _emit(status)

        sklearn_model = is_sklearn(model_name)

        if model_name == "Autoencoder":
            from models.autoencoder import get_autoencoder
            ae_modality = modality if modality in ("image", "audio", "tabular") else (
                "tabular" if modality == "timeseries" else None
            )
            if ae_modality is None:
                raise ValueError(
                    f"Autoencoder is not supported for the '{modality}' modality. "
                    "Supported: image, audio, tabular, timeseries."
                )
            model = get_autoencoder(
                modality=ae_modality,
                input_size=extra_data.get("input_size"),
                latent_dim=128,
            )

        elif sklearn_model:
            from models.tabular_models import get_tabular_model
            model = get_tabular_model(
                model_name, num_classes=num_classes,
                input_size=extra_data.get("sklearn_input_size", extra_data.get("input_size", 10)),
                task=task,
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                C=float(C_param), max_iter=int(max_iter),
                learning_rate=float(lr_xgb),
            )

        elif is_vit(model_name):
            from models.vit_models import get_vit_model
            model = get_vit_model(model_name, num_classes=num_classes,
                                   pretrained=(training_mode == "fine-tune"))

        elif is_whisper(model_name):
            from models.whisper_model import get_whisper_model
            size = "tiny" if "Tiny" in model_name else "base"
            model = get_whisper_model(num_classes=num_classes, model_size=size,
                                       freeze_encoder=(training_mode == "fine-tune"))

        elif modality in ("image", "audio"):
            from models.image_models import get_image_model
            model = get_image_model(model_name, num_classes=num_classes,
                                    mode=training_mode, dropout=float(dropout))

        elif modality == "text":
            from models.text_models import get_text_model
            model = get_text_model(
                model_name, num_classes=num_classes, mode=training_mode,
                vocab_size=extra_data.get("vocab_size", 30522),
                hidden_size=int(hidden_size), num_layers=int(num_layers),
                dropout=float(dropout))

        elif modality == "tabular":
            from models.tabular_models import get_tabular_model
            model = get_tabular_model(
                model_name, num_classes=num_classes,
                input_size=extra_data["input_size"], dropout=float(dropout))

        elif modality == "timeseries":
            from models.timeseries_models import get_timeseries_model
            model = get_timeseries_model(
                model_name, num_classes=num_classes,
                input_size=extra_data["input_size"],
                hidden_size=int(hidden_size), num_layers=int(num_layers),
                dropout=float(dropout))

        elif modality == "video":
            from models.video_models import get_video_model
            model = get_video_model(model_name, num_classes=num_classes, mode=training_mode)

        status = log(f"  ✓ Device: {DEVICE}")
        yield _emit(status)

        # ── 3. Train ──────────────────────────────────────────────────────────
        _set_phase("training", eta_text="ETA: estimating…")
        ckpt_dir = None
        if int(checkpoint_every) > 0:
            ckpt_dir = os.path.join("outputs", "checkpoints", safe_bundle_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            status = log(f"  Checkpoints → {ckpt_dir}")
            yield _emit(status)

        if model_name == "Autoencoder":
            import torch, torch.nn as nn
            model = model.to(DEVICE)
            ae_optim      = torch.optim.Adam(model.parameters(), lr=float(lr))
            ae_criterion  = nn.MSELoss()
            ae_history    = []
            ae_epoch_times = []
            status = log("Training — autoencoder optimisation started")
            yield _emit(status)
            for epoch in range(1, int(epochs) + 1):
                epoch_t0 = time.time()
                # ── Train pass ──────────────────────────────────────
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    inputs, _ = batch
                    inputs    = inputs.to(DEVICE)
                    ae_optim.zero_grad()
                    recon = model(inputs)
                    loss  = ae_criterion(recon, inputs)
                    loss.backward()
                    ae_optim.step()
                    train_loss += loss.item()
                train_loss /= max(len(train_loader), 1)

                # ── Validation pass ─────────────────────────────────
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, _ = batch
                        inputs    = inputs.to(DEVICE)
                        recon     = model(inputs)
                        val_loss += ae_criterion(recon, inputs).item()
                val_loss /= max(len(val_loader), 1)

                rec = {"epoch": epoch, "train_loss": round(train_loss, 4),
                       "val_loss": round(val_loss, 4), "val_acc": 0.0,
                       "eta_seconds": 0, "lr": float(lr)}
                epoch_elapsed = time.time() - epoch_t0
                ae_epoch_times.append(epoch_elapsed)
                avg_epoch_time = sum(ae_epoch_times) / len(ae_epoch_times)
                eta_seconds = avg_epoch_time * (int(epochs) - epoch)
                rec["eta_seconds"] = round(eta_seconds, 1)
                ae_history.append(rec)
                pct = 0.15 + 0.65 * epoch / int(epochs)
                progress(pct, desc=f"Autoencoder epoch {epoch}/{int(epochs)}")
                eta_str = _fmt_eta(eta_seconds) if epoch < int(epochs) else "ETA: completing training"
                telemetry["eta_text"] = eta_str
                _update_epoch_metrics(
                    epoch_text=f"{epoch}/{int(epochs)}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    metric="reconstruction",
                    lr_value=f"{float(lr):.6f}",
                )
                line = (f"Epoch {epoch}/{int(epochs)} — "
                        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {eta_str}")
                status = log(line)
                live_plot = _loss_plot(ae_history) if (
                    epoch == int(epochs) or epoch % _LIVE_PLOT_UPDATE_EVERY == 0
                ) else _NONE
                yield _emit(status, loss_fig=live_plot, eta=eta_str)
            final_model = model
            history     = ae_history

        elif sklearn_model:
            from training.trainer import train_sklearn
            X_train = extra_data["X_train"]; y_train = extra_data["y_train"]
            X_val   = extra_data.get("X_val"); y_val   = extra_data.get("y_val")
            _set_phase("training", eta_text="ETA: fitting baseline")
            _update_epoch_metrics(
                epoch_text="Baseline",
                train_loss="—",
                val_loss="—",
                metric="fitting",
                lr_value="—",
            )
            status = log("Training — fitting baseline model")
            yield _emit(status)
            for prog in train_sklearn(model, X_train, y_train, X_val, y_val):
                if prog["done"]:
                    final_model = prog["model"]
                    history     = prog["history"]
                    _set_phase("training", eta_text="ETA: finalizing results")
                    if history:
                        last = history[-1]
                        _update_epoch_metrics(
                            epoch_text=str(last.get("epoch", "1/1")),
                            train_loss=last.get("train_loss"),
                            val_loss=last.get("val_loss"),
                            metric=last.get("val_acc"),
                            lr_value=last.get("lr"),
                        )
                    status = log("Training — baseline fit complete, finalizing metrics")
                    yield _emit(status)
                else:
                    _set_phase("training", eta_text="ETA: fitting baseline")
                    status = log(prog.get("status", "Fitting model…"))
                    yield _emit(status, eta=telemetry["eta_text"])

        else:
            from training.trainer import train_pytorch, compute_class_weights
            # Optional class weighting for imbalanced datasets
            cw = None
            if use_class_weights and task in ("classification",):
                try:
                    # Fast path: pull labels directly from the dataset without
                    # iterating the full DataLoader (avoids an extra O(n) pass).
                    ds = train_loader.dataset
                    if hasattr(ds, "tensors"):
                        # TensorDataset — labels are the second tensor
                        all_labels = ds.tensors[1].tolist()
                    elif hasattr(ds, "samples"):
                        # ImageDataset / AudioDataset style
                        all_labels = [s[1] for s in ds.samples]
                    else:
                        # Fallback: iterate (safe but slower)
                        all_labels = [int(lbl) for _, lbls in train_loader
                                      for lbl in lbls.tolist()]
                    cw = compute_class_weights(all_labels)
                    status = log("  ✓ Class weights applied (imbalance correction)")
                    yield _emit(status)
                except Exception as e:
                    status = log(f"  ⚠️ Class weights skipped: {e}")
                    yield _emit(status)

            # Accumulate epoch records so the live loss plot shows the full curve,
            # not just the most-recent single epoch.
            _running_history: list[dict] = []
            status = log("Training — optimiser warmup complete, starting epochs")
            yield _emit(status)

            for prog in train_pytorch(
                model, train_loader, val_loader,
                epochs=int(epochs), lr=float(lr),
                optimizer_name=optimizer,
                patience=DEFAULTS["early_stopping_patience"],
                scheduler_name=scheduler_name,
                use_amp=bool(use_amp),
                task=task,
                class_weights=cw,
                checkpoint_dir=ckpt_dir,
                checkpoint_every=int(checkpoint_every),
            ):
                if prog["done"]:
                    final_model = prog["model"]
                    history     = prog["history"]
                else:
                    stopped_early_flag = prog.get("stopped_early", False)
                    pct = 0.15 + 0.65 * prog["epoch"] / prog["epochs"]
                    progress(pct, desc=f"Epoch {prog['epoch']}/{prog['epochs']}")
                    eta_str = _fmt_eta(prog.get("eta_seconds", 0))
                    telemetry["eta_text"] = eta_str
                    metric_label = "acc" if task == "classification" else (
                        "exact-match" if task == "multi-label" else "MAE")
                    _update_epoch_metrics(
                        epoch_text=f"{prog['epoch']}/{prog['epochs']}",
                        train_loss=prog["train_loss"],
                        val_loss=prog["val_loss"],
                        metric=f"{metric_label}={prog['val_acc']}",
                        lr_value=prog.get("lr", ""),
                    )
                    line = (
                        f"Epoch {prog['epoch']}/{prog['epochs']}  "
                        f"train={prog['train_loss']}  val={prog['val_loss']}  "
                        f"{metric_label}={prog['val_acc']}  "
                        f"lr={prog.get('lr','')}  {eta_str}"
                        + ("  ⏹ early stop" if prog["stopped_early"] else "")
                    )
                    status = log(line)
                    # Append to running history so the plot shows the full curve
                    _running_history.append({k: prog[k] for k in
                                             ("epoch", "train_loss", "val_loss", "val_acc")})
                    live_plot = _loss_plot(_running_history) if (
                        prog["epoch"] == prog["epochs"]
                        or prog["stopped_early"]
                        or prog["epoch"] % _LIVE_PLOT_UPDATE_EVERY == 0
                    ) else _NONE
                    yield _emit(status, loss_fig=live_plot,
                                eta=eta_str if not prog["stopped_early"] else "ETA: completing training")

        _set_phase("complete", eta_text="ETA: Done")
        if stopped_early_flag:
            status = log("Training — early stopping triggered, best checkpoint restored")
            yield _emit(status)
        status = log("Training complete ✓")
        loss_fig_final = _loss_plot(history) if history else _NONE
        yield _emit(status, loss_fig=loss_fig_final, eta="ETA: Done")

        # ── 3b. Plain-English training summary ───────────────────────────────
        training_summary_text = ""
        if history and not sklearn_model and model_name != "Autoencoder":
            try:
                from ui.training_summary import summarise_run
                training_summary_text = summarise_run(
                    history, task, model_name, modality,
                    stopped_early=stopped_early_flag)
                yield _emit(status, loss_fig=loss_fig_final, eta="ETA: Done",
                            train_summary=training_summary_text)
            except Exception as e:
                logger.warning("Training summary generation failed: %s", e)

        # ── 4. Architecture visualisation ─────────────────────────────────────
        arch_fig = None
        if final_model is not None and not sklearn_model and model_name != "Autoencoder":
            try:
                from ui.architecture_viz import summarise_model
                _, arch_fig = summarise_model(final_model)
            except Exception as e:
                logger.warning("Architecture visualisation failed: %s", e)

        # ── 5. Clustering ─────────────────────────────────────────────────────
        tsne_fig = None
        if task in ("clustering",) or model_name == "Autoencoder":
            _set_phase("clustering", eta_text="ETA: clustering")
            progress(0.84, desc="Clustering embeddings…")
            if sklearn_model:
                status = log("ℹ️  sklearn models don't support embedding-based clustering.\n"
                             "   Use an MLP or Autoencoder for full clustering support.")
                yield _emit(status)
            else:
                status = log("Extracting embeddings → KMeans…")
                yield _emit(status)
                try:
                    from training.clustering import run_clustering, tsne_plot
                    k = int(n_clusters) if int(n_clusters) > 1 else num_classes
                    clustering_result = run_clustering(final_model, train_loader, n_clusters=k)
                    tsne_fig = tsne_plot(clustering_result["features"],
                                        clustering_result["cluster_labels"])
                    status = log(f"  Silhouette score: {clustering_result['silhouette_score']}")
                    yield _emit(status, tsne=gr.update(value=tsne_fig))
                except Exception as e:
                    status = log(f"  Clustering skipped: {e}")
                    yield _emit(status)

        # ── 6. Evaluation ─────────────────────────────────────────────────────
        cm_fig = roc_fig = gcam_fig = shap_fig = None
        mis_img_fig = None
        mis_df_val  = None
        reg_fig     = None
        anomaly_fig = None

        # Standard classification / multi-label evaluation
        if task in ("classification", "multi-label") and model_name != "Autoencoder":
            _set_phase("evaluating", eta_text="ETA: evaluating")
            progress(0.88, desc="Evaluating…")
            status = log("Running evaluation…")
            yield _emit(status)

            try:
                from eval.metrics import (
                    compute_confusion_matrix,
                    compute_roc_curves,
                    _get_preds_and_probs,
                    classification_metric_summary,
                )
                cm_fig, eval_summary_text = compute_confusion_matrix(
                    final_model, val_loader, classes, modality, task,
                    is_sklearn=sklearn_model,
                    X_val=extra_data.get("X_val"), y_val=extra_data.get("y_val"))
                if task in ("classification", "multi-label"):
                    roc_fig = compute_roc_curves(
                        final_model, val_loader, classes, modality, task,
                        is_sklearn=sklearn_model,
                        X_val=extra_data.get("X_val"), y_val=extra_data.get("y_val"))
                if task == "classification":
                    y_true, y_pred, y_prob = _get_preds_and_probs(
                        final_model, val_loader, classes, sklearn_model,
                        extra_data.get("X_val"), extra_data.get("y_val"), DEVICE
                    )
                    metrics_payload = classification_metric_summary(
                        y_true, y_pred, y_prob, n_classes=len(classes)
                    )
                status = log("  ✓ Metrics computed")
            except Exception as e:
                status = log(f"  Metrics error: {e}")
            yield _emit(status)

            if modality in ("image", "audio", "video") and not sklearn_model:
                try:
                    from eval.gradcam import compute_gradcam
                    gcam_fig = compute_gradcam(final_model, val_loader, modality,
                                               model_name, prep, n_samples=4)
                    if gcam_fig:
                        status = log("  ✓ Grad-CAM generated")
                except Exception as e:
                    status = log(f"  Grad-CAM skipped: {e}")
                yield _emit(status)

            if modality == "tabular":
                try:
                    from eval.shap_explain import compute_shap
                    shap_fig = compute_shap(final_model, extra_data, modality, prep)
                    if shap_fig:
                        status = log("  ✓ SHAP computed")
                except Exception as e:
                    status = log(f"  SHAP skipped: {e}")
                yield _emit(status)

            text_shap_html = None
            if modality == "text" and not sklearn_model:
                try:
                    from eval.shap_explain import compute_text_shap
                    text_shap_html = compute_text_shap(
                        final_model, val_loader, classes, prep, n_samples=3)
                    if text_shap_html:
                        status = log("  ✓ Token importance computed")
                except Exception as e:
                    status = log(f"  Token importance skipped: {e}")
                yield _emit(status)

            try:
                from eval.misclassified import find_misclassified
                mis_img_fig, mis_df_val = find_misclassified(
                    final_model, val_loader, classes, modality, prep,
                    task=task,
                    val_samples=extra_data.get("val_samples"),
                    val_texts=extra_data.get("val_texts"),
                    X_val=extra_data.get("X_val"), y_val=extra_data.get("y_val"),
                    is_sklearn=sklearn_model,
                )
                status = log("  ✓ Misclassified samples collected")
            except Exception as e:
                status = log(f"  Misclassified skipped: {e}")
            yield _emit(status)

        # Regression evaluation
        if task == "regression" and model_name != "Autoencoder":
            _set_phase("evaluating", eta_text="ETA: evaluating")
            progress(0.88, desc="Evaluating…")
            status = log("Running regression evaluation…")
            yield _emit(status)

            try:
                import numpy as np
                import torch
                from eval.regression_metrics import (
                    compute_regression_metrics,
                    residual_plot,
                    format_regression_report,
                )

                if sklearn_model:
                    X_val = extra_data.get("X_val")
                    y_val = extra_data.get("y_val")
                    if X_val is None or y_val is None:
                        raise ValueError("Validation data is unavailable for regression evaluation.")
                    all_true = np.asarray(y_val, dtype=float).ravel().tolist()
                    all_pred = np.asarray(final_model.predict(X_val), dtype=float).ravel().tolist()
                else:
                    all_true, all_pred = [], []
                    final_model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            inputs, labels = batch
                            logits = final_model(inputs.to(DEVICE))
                            all_true.extend(np.asarray(labels.cpu().numpy(), dtype=float).ravel().tolist())
                            all_pred.extend(np.asarray(logits.detach().cpu().numpy(), dtype=float).ravel().tolist())

                reg_metrics = compute_regression_metrics(all_true, all_pred)
                metrics_payload = reg_metrics
                eval_summary_text = format_regression_report(reg_metrics)
                reg_fig = residual_plot(all_true, all_pred)
                status = log("  ✓ Regression metrics (R², RMSE, MAE)")
            except Exception as e:
                status = log(f"  Regression metrics skipped: {e}")
            yield _emit(status)

        # Anomaly detection evaluation (Autoencoder)
        if model_name == "Autoencoder" or task == "anomaly":
            _set_phase("computing anomaly scores", eta_text="ETA: computing anomaly scores")
            progress(0.88, desc="Computing anomaly scores…")
            try:
                from eval.anomaly_detection import (
                    compute_reconstruction_errors, find_anomalies,
                    reconstruction_error_plot)
                errors       = compute_reconstruction_errors(final_model, val_loader, DEVICE)
                anom_result  = find_anomalies(errors)
                anomaly_fig  = reconstruction_error_plot(errors)
                n_anom       = anom_result["n_anomalies"]
                ratio        = anom_result["anomaly_ratio"] * 100
                thr          = anom_result["threshold"]
                eval_summary_text += (
                    f"\n\n**Anomaly Detection**\n"
                    f"Threshold: {thr:.4f}  |  "
                    f"Anomalies: {n_anom} ({ratio:.1f}%)"
                )
                status = log(f"  ✓ Anomaly detection: {n_anom} anomalies ({ratio:.1f}%)")
            except Exception as e:
                status = log(f"  Anomaly detection skipped: {e}")
            yield _emit(status)

        # ── 7. Export ─────────────────────────────────────────────────────────
        _set_phase("exporting", eta_text="ETA: exporting")
        progress(0.95, desc="Exporting bundle…")
        status = log("Exporting ONNX bundle…")
        yield _emit(status)

        sample_batch = None
        if not sklearn_model:
            try:
                sample_inputs, _ = next(iter(val_loader))
                sample_batch = ({k: v[:1] for k, v in sample_inputs.items()}
                                if isinstance(sample_inputs, dict)
                                else sample_inputs[:1])
            except StopIteration:
                pass

        from export.exporter import export_bundle
        bundle_path, export_warnings = export_bundle(
            model=final_model,
            preprocessing_config=prep,
            classes=classes,
            bundle_name=safe_bundle_name,
            output_dir=str(_OUTPUTS_ROOT),
            sample_batch=sample_batch,
            is_sklearn=sklearn_model,
            clustering_result=clustering_result,
        )
        for w in export_warnings:
            status = log(w)
            yield _emit(status)

        class_note_text, explanation_note_text, diagnostics_note_text, sample_review_note_text = (
            _build_evaluation_notes(
                task=task,
                modality=modality,
                model_name=model_name,
                sklearn_model=sklearn_model,
                cm_fig=cm_fig,
                roc_fig=roc_fig,
                gcam_fig=gcam_fig,
                shap_fig=shap_fig,
                reg_fig=reg_fig,
                anomaly_fig=anomaly_fig,
                mis_img_fig=mis_img_fig,
                mis_df_val=mis_df_val,
                tsne_fig=tsne_fig,
            )
        )
        eval_kpi_html = _format_eval_kpis(eval_summary_text, history, task)

        # ── 8. Save run history ───────────────────────────────────────────────
        try:
            from training.run_comparison import save_run
            save_run(
                model_name=model_name, modality=modality,
                task=task, training_mode=training_mode,
                hyperparams={"lr": lr, "batch_size": batch_size, "epochs": epochs,
                             "optimizer": optimizer, "scheduler": scheduler_name},
                history=history, bundle_path=bundle_path,
                metrics=metrics_payload,
            )
        except Exception as e:
            logger.warning("Could not save run history: %s", e)

        _set_phase("complete", eta_text="ETA: Done")
        progress(1.0, desc="Done!")
        status = log(f"\n✅ Done!  Bundle saved to:\n  {bundle_path}")
        yield _emit(
            status,
            loss_fig=gr.update(value=_loss_plot(history) if history else None),
            eta="ETA: Done",
            eval_sum=_format_eval_summary(eval_summary_text),
            cm=gr.update(value=cm_fig),
            roc=gr.update(value=roc_fig),
            tsne=gr.update(value=tsne_fig),
            gcam=gr.update(value=gcam_fig),
            shap=gr.update(value=shap_fig),
            mis_img=gr.update(value=mis_img_fig),
            mis_df=gr.update(value=mis_df_val),
            bundle=bundle_path,
            arch=gr.update(value=arch_fig),
            train_summary=training_summary_text,
            reg_plot=gr.update(value=reg_fig),
            anomaly_plot=gr.update(value=anomaly_fig),
            eval_kpis=eval_kpi_html,
            class_note=class_note_text,
            explanation_note_val=explanation_note_text,
            diagnostics_note_val=diagnostics_note_text,
            sample_review_note_val=sample_review_note_text,
            text_shap=gr.update(value=text_shap_html or ""),
            dash_kpi=_build_dashboard_kpi(task, modality, history, metrics_payload,
                                          model_name, training_mode),
            dash_primary=gr.update(value=_pick_primary_chart(
                task, cm_fig, reg_fig, tsne_fig, anomaly_fig)),
            dash_explain_plot=gr.update(value=shap_fig or gcam_fig),
            dash_explain_html=gr.update(value=text_shap_html or ""),
            dash_loss=gr.update(value=_loss_plot(history) if history else None),
            dash_actions=_build_action_items(task, history, metrics_payload, model_name),
        )

    except Exception as exc:
        _set_phase("error", eta_text="ETA: halted")
        try:
            from ui.error_formatter import format_error
            friendly = format_error(exc)
        except Exception:
            friendly = f"❌ **Error:** {exc}"
        tb = traceback.format_exc()
        yield _emit(log(
            f"{friendly}\n\n"
            f"{'─' * 60}\n"
            f"Traceback (most recent call last):\n{tb}"
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="NoCode-DL", theme=APP_THEME, css=APP_CSS) as demo:
    with gr.Column(elem_classes=["studio-shell"]):
        gr.HTML('<a href="#studio-main-content" class="studio-skip-link">Skip to main content</a>')
        gr.HTML(
            f"""
<section class="studio-hero" aria-label="Welcome">
  <div class="studio-hero__grid">
    <div class="studio-hero__content">
      <div class="studio-hero__eyebrow">NoCode Deep Learning Studio</div>
      <h1 class="studio-hero__title">Build confident deep learning projects.</h1>
      <p class="studio-hero__subtitle">
        A local-first studio: drop in a dataset, let the workspace infer the setup,
        experiment with modern model families, and learn the workflow visually instead of wrestling with boilerplate.
      </p>
      <div class="studio-hero__pills">
        <span class="studio-hero__pill">Guided no-code workflow</span>
        <span class="studio-hero__pill">Pretrained and from-scratch models</span>
        <span class="studio-hero__pill">Runs locally on your machine</span>
      </div>
      <div class="studio-hero__meta">
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Runtime</span>
          <span class="studio-hero__meta-value">Local-first on your machine</span>
        </div>
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Hardware</span>
          <span class="studio-hero__meta-value">Accelerator detected: {DEVICE.upper()}</span>
        </div>
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Workflow</span>
          <span class="studio-hero__meta-value">Data → Model → Train → Evaluate → Export</span>
        </div>
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Privacy</span>
          <span class="studio-hero__meta-value">🔒 Your data never leaves this machine</span>
        </div>
      </div>
    </div>
  </div>
</section>
"""
        )

        project_mode_state = gr.State(value=_project_mode_default())
        recommendation_state = gr.State(value={})
        quality_report_state = gr.State(value={})

        with gr.Row():
            with gr.Group(elem_classes=["studio-card", "studio-card--soft"]) as why_group:
                project_mode = gr.Radio(
                    PROJECT_MODES,
                    value=_project_mode_default(),
                    label="Workspace mode",
                    info=tip("project_mode"),
                )
            with gr.Group(elem_classes=["studio-card"]) as explanation_group_top:
                why_box = gr.Markdown(
                    _why_this_matters_markdown(_project_mode_default(), "image", "", "classification", {})
                )

        with gr.Row(elem_classes=["studio-cockpit"]):
            with gr.Group(elem_classes=["studio-card"]):
                workspace_snapshot = gr.Markdown(
                    "### Workspace cockpit\n\n- **Dataset**: Waiting\n- **Model**: Waiting\n- **Training**: Blocked"
                )
            with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                next_step_box = gr.Markdown(
                    "### Next step\n\nUpload a dataset or paste a local path to get started."
                )

        with gr.Row():
            with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                with gr.Row():
                    tutorial_modality_dropdown = gr.Dropdown(
                        choices=_TUTORIAL_MODALITIES,
                        value="image",
                        label="Tutorial modality",
                        scale=1,
                        min_width=160,
                    )
                    load_tutorial_btn = gr.Button("✨ Load Tutorial", variant="secondary", scale=3)
                tutorial_box = gr.Markdown(_tutorial_markdown("image"))

        with gr.Tabs(elem_classes=["studio-panel"], elem_id="studio-main-content"):

            # ── Tab 1: Data ───────────────────────────────────────────────────────
            with gr.Tab("🗂 Data", elem_classes=["studio-tab-panel"]):
                gr.HTML(
                    """
<section class="studio-banner" aria-label="Step 1: Data setup">
  <div class="studio-banner__grid">
    <div>
      <div class="studio-banner__eyebrow">Step 1</div>
      <h2>Turn raw files into a confident starting point.</h2>
      <p>Upload a structured file or folder-based dataset and let the workspace detect the shape of the problem. The goal is to reduce manual setup so you can focus on reasoning about the data.</p>
    </div>
    <div class="studio-showcase">
      <div class="studio-showcase__card"><strong>Auto-detected schema</strong><span>Columns and likely label fields appear as selectors instead of requiring memorized names.</span></div>
      <div class="studio-showcase__card"><strong>Class balance preview</strong><span>Students can quickly see skew, missing fields, or modality mismatches before training.</span></div>
    </div>
  </div>
</section>
"""
                )
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"]) as model_selection_group:
                            gr.HTML(_section_intro("Dataset setup", "Point the workspace at your source data, then let the app infer the columns and prepare the right controls for the modality."))
                            modality = gr.Dropdown(MODALITIES, value="image", label="Modality",
                                                   info=tip("modality"))

                            with gr.Accordion("📂 Upload Dataset", open=True):
                                gr.Markdown(
                                    "Upload a **ZIP** (image/audio/video class folders) "
                                    "or **CSV/TSV/JSON** (tabular/text/timeseries). "
                                    "The path below will be filled automatically."
                                )
                                data_upload = gr.File(
                                    label="Drop file here",
                                    file_types=[".zip", ".csv", ".tsv", ".json"],
                                    file_count="single",
                                )

                            data_path = gr.Textbox(
                                label="Data path",
                                placeholder="/path/to/dataset  or  /path/to/data.csv",
                                info=tip("data_path"))

                        with gr.Group(elem_classes=["studio-card", "studio-card--compact"]) as column_mapping_group:
                            gr.HTML(_section_intro("Column mapping", "For structured data, selectors populate from detected fields so you can choose from the actual dataset schema."))
                            label_col   = gr.Dropdown(
                                choices=[],
                                value="label",
                                label="Label column",
                                info=tip("label_col"),
                                allow_custom_value=True,
                                visible=True,
                            )
                            text_col    = gr.Dropdown(
                                choices=[],
                                value="text",
                                label="Input text column (X)",
                                info=tip("text_col"),
                                allow_custom_value=True,
                                visible=False,
                            )
                            feature_cols = gr.Dropdown(
                                choices=[],
                                value=[],
                                multiselect=True,
                                label="Feature columns",
                                info=tip("feature_cols"),
                                visible=False,
                            )
                            time_col    = gr.Dropdown(
                                choices=[],
                                value="",
                                label="Timestamp column (optional)",
                                info=tip("time_col"),
                                allow_custom_value=True,
                                visible=False,
                            )
                            window_size = gr.Slider(10, 500, value=50,    step=10,
                                                    label="Window size (timeseries)", visible=False,
                                                    info=tip("window_size"))
                            n_frames    = gr.Slider(4,  32,  value=8,     step=4,
                                                    label="Frames to sample (video)", visible=False,
                                                    info=tip("n_frames"))
                            image_size  = gr.Slider(64, 384, value=160, step=16,
                                                    label="Image size", visible=False,
                                                    info=tip("image_size"))
                            image_size_hint = gr.Markdown(
                                value="",
                                visible=False,
                            )
                            sample_rate = gr.Slider(8000, 44100, value=22050, step=1000,
                                                    label="Sample rate Hz (audio)", visible=False,
                                                    info=tip("sample_rate"))
                            audio_image_size = gr.Slider(96, 384, value=224, step=32,
                                                         label="Spectrogram image size", visible=False,
                                                         info=tip("audio_image_size"))
                            audio_n_mels = gr.Slider(32, 256, value=64, step=16,
                                                     label="Mel bins", visible=False,
                                                     info=tip("audio_n_mels"))
                            augmentation = gr.Dropdown(AUG_LEVELS, value="light",
                                                       label="Augmentation level",
                                                       info=tip("augmentation"))

                        with gr.Group(elem_classes=["studio-card", "studio-card--compact"]) as split_cleaning_group:
                            with gr.Accordion("⚙️ Split, Subset & Cleaning", open=True):
                                val_split = gr.Slider(0.1, 0.4, value=0.2, step=0.05,
                                                      label="Validation split",
                                                      info=tip("val_split"))
                                use_random_subset = gr.Checkbox(
                                    value=False,
                                    label="Use random subset for quick experiments",
                                    info=tip("use_random_subset"),
                                    visible=False,
                                )
                                subset_percent = gr.Slider(
                                    0.1, 100.0, value=1.0, step=0.1,
                                    label="Subset percentage",
                                    info=tip("subset_percent"),
                                    visible=False,
                                )
                                subset_seed = gr.Number(
                                    value=42,
                                    precision=0,
                                    label="Subset seed",
                                    info=tip("subset_seed"),
                                    visible=False,
                                )
                                use_data_cleaning = gr.Checkbox(value=False,
                                                                 label="Auto-clean tabular data",
                                                                 info=tip("use_data_cleaning"),
                                                                 visible=False)
                                tabular_missing_strategy = gr.Dropdown(
                                    ["auto", "median", "mean", "drop"],
                                    value="auto",
                                    label="Tabular missing-value strategy",
                                    info=tip("tabular_missing_strategy"),
                                    visible=False,
                                )
                                tabular_clip_outliers = gr.Checkbox(
                                    value=False,
                                    label="Clip extreme tabular values",
                                    info=tip("tabular_clip_outliers"),
                                    visible=False,
                                )
                                text_lowercase = gr.Checkbox(
                                    value=True,
                                    label="Lowercase text",
                                    info=tip("text_lowercase"),
                                    visible=False,
                                )
                                text_strip_urls = gr.Checkbox(
                                    value=True,
                                    label="Remove URLs from text",
                                    info=tip("text_strip_urls"),
                                    visible=False,
                                )
                                text_strip_punctuation = gr.Checkbox(
                                    value=False,
                                    label="Remove punctuation from text",
                                    info=tip("text_strip_punctuation"),
                                    visible=False,
                                )
                                text_remove_stopwords = gr.Checkbox(
                                    value=False,
                                    label="Remove common stop words",
                                    info=tip("text_remove_stopwords"),
                                    visible=False,
                                )
                                text_deduplicate = gr.Checkbox(
                                    value=False,
                                    label="Remove duplicate text rows",
                                    info=tip("text_deduplicate"),
                                    visible=False,
                                )
                                text_apply_stemming = gr.Checkbox(
                                    value=False,
                                    label="Apply stemming",
                                    info=tip("text_apply_stemming"),
                                    visible=False,
                                )
                                text_apply_lemmatization = gr.Checkbox(
                                    value=False,
                                    label="Apply lightweight lemmatization",
                                    info=tip("text_apply_lemmatization"),
                                    visible=False,
                                )
                                text_use_ngrams = gr.Checkbox(
                                    value=False,
                                    label="Use n-gram baseline features for classical text models",
                                    info=tip("text_use_ngrams"),
                                    visible=False,
                                )
                                timeseries_sort_by_time = gr.Checkbox(
                                    value=True,
                                    label="Sort rows by timestamp",
                                    info=tip("timeseries_sort_by_time"),
                                    visible=False,
                                )
                                timeseries_fill_strategy = gr.Dropdown(
                                    ["none", "forward_fill", "interpolate"],
                                    value="none",
                                    label="Time-series missing-value strategy",
                                    info=tip("timeseries_fill_strategy"),
                                    visible=False,
                                )
                                image_verify_files = gr.Checkbox(
                                    value=False,
                                    label="Skip unreadable image files",
                                    info=tip("image_verify_files"),
                                    visible=False,
                                )
                                image_aug_flip = gr.Checkbox(
                                    value=True,
                                    label="Image: horizontal flip",
                                    info=tip("image_aug_flip"),
                                    visible=False,
                                )
                                image_aug_vertical = gr.Checkbox(
                                    value=False,
                                    label="Image: vertical flip",
                                    info=tip("image_aug_vertical"),
                                    visible=False,
                                )
                                image_aug_rotation = gr.Checkbox(
                                    value=False,
                                    label="Image: rotation",
                                    info=tip("image_aug_rotation"),
                                    visible=False,
                                )
                                image_aug_color = gr.Checkbox(
                                    value=False,
                                    label="Image: color jitter",
                                    info=tip("image_aug_color"),
                                    visible=False,
                                )
                                image_aug_gray = gr.Checkbox(
                                    value=False,
                                    label="Image: grayscale",
                                    info=tip("image_aug_gray"),
                                    visible=False,
                                )
                                image_aug_perspective = gr.Checkbox(
                                    value=False,
                                    label="Image: perspective warp",
                                    info=tip("image_aug_perspective"),
                                    visible=False,
                                )
                                image_normalization = gr.Dropdown(
                                    ["imagenet", "simple_0_1", "none"],
                                    value="imagenet",
                                    label="Image normalization preset",
                                    info=tip("image_normalization"),
                                    visible=False,
                                )
                                image_force_grayscale = gr.Checkbox(
                                    value=False,
                                    label="Convert all images to grayscale before training",
                                    info=tip("image_force_grayscale"),
                                    visible=False,
                                )
                                audio_verify_files = gr.Checkbox(
                                    value=False,
                                    label="Skip unreadable audio files",
                                    info=tip("audio_verify_files"),
                                    visible=False,
                                )
                                audio_normalize_waveform = gr.Checkbox(
                                    value=True,
                                    label="Normalize audio loudness",
                                    info=tip("audio_normalize_waveform"),
                                    visible=False,
                                )
                                audio_aug_noise = gr.Checkbox(
                                    value=True,
                                    label="Audio: add noise",
                                    info=tip("audio_aug_noise"),
                                    visible=False,
                                )
                                audio_aug_shift = gr.Checkbox(
                                    value=False,
                                    label="Audio: time shift",
                                    info=tip("audio_aug_shift"),
                                    visible=False,
                                )
                                audio_aug_gain = gr.Checkbox(
                                    value=False,
                                    label="Audio: gain jitter",
                                    info=tip("audio_aug_gain"),
                                    visible=False,
                                )
                                audio_aug_time_mask = gr.Checkbox(
                                    value=False,
                                    label="Audio: time masking",
                                    info=tip("audio_aug_time_mask"),
                                    visible=False,
                                )
                                audio_aug_freq_mask = gr.Checkbox(
                                    value=False,
                                    label="Audio: frequency masking",
                                    info=tip("audio_aug_freq_mask"),
                                    visible=False,
                                )
                                video_verify_files = gr.Checkbox(
                                    value=False,
                                    label="Skip unreadable video files",
                                    info=tip("video_verify_files"),
                                    visible=False,
                                )
                                tabular_scaling = gr.Dropdown(
                                    ["standard", "minmax", "robust", "none"],
                                    value="standard",
                                    label="Tabular scaling strategy",
                                    info=tip("tabular_scaling"),
                                    visible=False,
                                )

                        preview_btn = gr.Button("🔍 Preview Dataset", variant="secondary")

                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"]) as evaluation_report_group:
                            gr.HTML(_section_intro("Dataset intelligence", "Preview the upload, review validation feedback, and sanity-check class balance before you commit to training."))
                            stats_summary_box = gr.Textbox(label="Dataset summary", lines=10,
                                                            interactive=False)
                            dist_plot         = gr.Plot(label="Class distribution")
                            validation_box    = gr.Markdown(
                                value="Review validation feedback and inferred columns here after previewing the dataset."
                            )
                            quality_report_box = gr.Markdown(
                                value="The preflight Data Quality Report will appear here after preview."
                            )

            # ── Tab 2: Model ──────────────────────────────────────────────────────
            with gr.Tab("🧠 Model", elem_classes=["studio-tab-panel"]):
                gr.HTML(
                    """
<section class="studio-banner" aria-label="Step 2: Model selection">
  <div class="studio-banner__grid">
    <div>
      <div class="studio-banner__eyebrow">Step 2</div>
      <h2>Choose a model path that informs, not overwhelms.</h2>
      <p>Stay in a guided lane with sensible defaults, while still having room to explore pretrained models, deeper architectures, and training strategy choices when you want them.</p>
    </div>
    <div class="studio-badge-list">
      <span class="studio-badge">Fine-tune pretrained backbones</span>
      <span class="studio-badge">Train from scratch</span>
      <span class="studio-badge">Recommended defaults</span>
      <span class="studio-badge">Advanced knobs when needed</span>
    </div>
  </div>
</section>
"""
                )
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"]) as export_utilities_group:
                            gr.HTML(_section_intro("Architecture", "Choose how much of the model should be pretrained, select the task, and let the app guide the setup toward safe defaults."))
                            training_mode = gr.Radio(["fine-tune", "from_scratch"], value="fine-tune",
                                                     label="Training mode",
                                                     info=tip("training_mode"))
                            model_name    = gr.Dropdown([], label="Model architecture",
                                                        info=tip("model_name"))
                            task          = gr.Dropdown(TASKS, value="classification", label="Task",
                                                        info=tip("task"))
                            n_clusters    = gr.Slider(2, 32, value=8, step=1,
                                                      label="Number of clusters (clustering only)",
                                                      info=tip("n_clusters"),
                                                      visible=False)
                            use_class_weights = gr.Checkbox(
                                value=False, label="Auto-balance class weights",
                                info=tip("use_class_weights"))
                            recommend_btn       = gr.Button("✨ Get Guided Recommendations",
                                                            variant="secondary")
                            apply_recommendations_btn = gr.Button(
                                "Apply recommendations",
                                variant="primary"
                            )
                            recommendations_box = gr.Markdown("*Load a dataset first, then click above.*")
                            recommendation_rationale_box = gr.Markdown("*The rationale for the recommendation will appear here.*")

                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"]) as hyperparams_group:
                            gr.HTML(_section_intro("Hyperparameters", "Keep the primary settings visible and tuck specialist controls away until they’re actually relevant."))
                            lr         = gr.Slider(1e-5, 1e-1, value=1e-3, step=1e-5,
                                                   label="Learning rate",
                                                   info=tip("learning_rate"))
                            batch_size = gr.Slider(4, 64, value=16, step=4, label="Batch size",
                                                   info=tip("batch_size"))
                            epochs     = gr.Slider(1, 100, value=10, step=1, label="Epochs",
                                                   info=tip("epochs"))
                            dropout    = gr.Slider(0.0, 0.8, value=0.3, step=0.05, label="Dropout",
                                                   info=tip("dropout"))
                            optimizer  = gr.Dropdown(OPTIMIZERS, value="adam", label="Optimizer",
                                                     info=tip("optimizer"))
                            scheduler_name = gr.Dropdown(SCHEDULERS, value="cosine",
                                                         label="LR scheduler",
                                                         info=tip("scheduler"))
                            use_amp    = gr.Checkbox(value=False, label="Mixed precision (CUDA only)",
                                                     info=tip("use_amp"))

                            with gr.Accordion("Sequence model options", open=False) as sequence_options_group:
                                hidden_size = gr.Slider(32, 512, value=128, step=32,
                                                        label="Hidden size",
                                                        info=tip("hidden_size"))
                                num_layers  = gr.Slider(1, 4, value=1, step=1,
                                                        label="Num layers",
                                                        info=tip("num_layers"))

                            with gr.Accordion("sklearn / XGBoost options", open=False) as sklearn_options_group:
                                n_estimators = gr.Slider(10,  500, value=100, step=10,
                                                          label="n_estimators",
                                                          info=tip("n_estimators"))
                                max_depth    = gr.Slider(0,   50,  value=0,   step=1,
                                                          label="Max depth (0 = unlimited)",
                                                          info=tip("max_depth"))
                                C_param      = gr.Slider(0.01, 10, value=1.0, step=0.01,
                                                          label="C (LogisticRegression)",
                                                          info=tip("C_param"))
                                max_iter     = gr.Slider(100, 2000, value=1000, step=100,
                                                          label="Max iter (LogReg)",
                                                          info=tip("max_iter"))
                                lr_xgb       = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                                          label="Learning rate (XGBoost)",
                                                          info=tip("lr_xgb"))

            # ── Tab 3: Train ──────────────────────────────────────────────────────
            with gr.Tab("🚀 Train", elem_classes=["studio-tab-panel"]):
                gr.HTML(
                    """
<div class="studio-kpi-grid">
  <div class="studio-kpi"><strong>Launch point</strong><span>Run the core training path first</span></div>
  <div class="studio-kpi"><strong>Outputs</strong><span>Curves, summaries, exports, and bundle path</span></div>
  <div class="studio-kpi"><strong>Advanced mode</strong><span>Cross-validation and search after baseline success</span></div>
</div>
"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML(_section_intro("Run configuration", "Name the bundle, control checkpoint behavior, and start or stop the active training session from one place."))
                            bundle_name = gr.Textbox(value="my_model", label="Bundle name",
                                                      info=tip("bundle_name"))
                            checkpoint_every = gr.Slider(0, 20, value=0, step=1,
                                                          label="Save checkpoint every N epochs",
                                                          info=tip("checkpoint_every"))
                            train_btn   = gr.Button("▶ Train & Export", variant="primary")
                            stop_btn    = gr.Button("⏹ Stop", variant="stop")
                            gr.HTML("<p class='studio-inline-note'>Tip: start with a short run to verify your dataset and model pairing before longer experiments.</p>")

                        with gr.Group(elem_classes=["studio-card", "studio-card--compact", "studio-card--soft"]) as advanced_training_group:
                            gr.HTML(_section_intro("Advanced training", "Use these after the baseline workflow is stable. They reuse your Data and Model settings."))
                            with gr.Accordion("🔬 Cross-validation and search", open=False):
                                with gr.Tabs():
                                    with gr.Tab("🔁 K-Fold Cross-Validation"):
                                        gr.Markdown(
                                            "Evaluate model stability across multiple train/val splits. "
                                            "Supports up to 10 folds and reports a richer metric suite for classification."
                                        )
                                        cv_k = gr.Slider(2, 10, value=10, step=1,
                                                          label="Number of folds (k)",
                                                          info=tip("cv_k"))
                                        cv_run_btn = gr.Button("▶ Run Cross-Validation",
                                                               variant="secondary")
                                        cv_significance_models = gr.Dropdown(
                                            choices=[],
                                            value=[],
                                            multiselect=True,
                                            label="Compare against additional model families",
                                            info=tip("cv_significance_models"),
                                        )
                                        cv_significance_metric = gr.Dropdown(
                                            choices=["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_ovr", "mcc"],
                                            value="accuracy",
                                            label="Significance metric",
                                            info=tip("cv_significance_metric"),
                                        )
                                        cv_significance_btn = gr.Button(
                                            "🧪 Test Significance Between Families",
                                            variant="secondary",
                                        )
                                        cv_significance_box = gr.Markdown(
                                            "*Select one or more additional families and use 10-fold CV to compare them against the active model.*"
                                        )
                                        cv_significance_plot = gr.Plot(
                                            label="Fold-by-fold family comparison"
                                        )
                                        cv_results_box = gr.Textbox(
                                            label="Cross-validation results", lines=10,
                                            interactive=False,
                                            value="*Configure Data & Model tabs, then click Run.*")

                                    with gr.Tab("🎲 Random Hyperparameter Search"):
                                        gr.Markdown(
                                            "Try multiple learning rate and dropout combinations "
                                            "and report which performs best on the validation set."
                                        )
                                        hparam_lr_vals = gr.Textbox(
                                            value="1e-4, 1e-3, 1e-2",
                                            label="Learning rate candidates (comma-separated)",
                                            info=tip("hparam_lr_vals"))
                                        hparam_dropout_vals = gr.Textbox(
                                            value="0.1, 0.3, 0.5",
                                            label="Dropout candidates (comma-separated)",
                                            info=tip("hparam_dropout_vals"))
                                        hparam_n_trials     = gr.Slider(2, 20, value=6, step=1,
                                                                         label="Max trials (random subset of grid)",
                                                                         info=tip("hparam_n_trials"))
                                        hparam_epochs       = gr.Slider(1, 20, value=3, step=1,
                                                                         label="Epochs per trial",
                                                                         info=tip("hparam_epochs"))
                                        hparam_run_btn      = gr.Button("▶ Run Hyperparameter Search",
                                                                         variant="secondary")
                                        hparam_results_box  = gr.Textbox(
                                            label="Search results", lines=12,
                                            interactive=False,
                                            value="*Configure Data & Model tabs, then click Run.*")

                                    with gr.Tab("🧭 Model Sweep"):
                                        gr.Markdown(
                                            "Train multiple selected model families with the current Data and Hyperparameter settings, then rank them in one sortable table."
                                        )
                                        sweep_models = gr.Dropdown(
                                            choices=[],
                                            value=[],
                                            multiselect=True,
                                            label="Models to train and compare",
                                            info=tip("sweep_models"),
                                        )
                                        sweep_metrics = gr.Dropdown(
                                            choices=CLASSIFICATION_METRIC_OPTIONS,
                                            value=["accuracy", "f1_macro", "auc_ovr"],
                                            multiselect=True,
                                            label="Metrics to show in the table",
                                            info=tip("sweep_metrics"),
                                        )
                                        with gr.Row():
                                            sweep_sort_metric = gr.Dropdown(
                                                choices=CLASSIFICATION_METRIC_OPTIONS,
                                                value="accuracy",
                                                label="Sort by metric",
                                                info=tip("sweep_sort_metric"),
                                            )
                                            sweep_sort_order = gr.Dropdown(
                                                choices=["descending", "ascending"],
                                                value="descending",
                                                label="Sort order",
                                                info=tip("sweep_sort_order"),
                                            )
                                        sweep_run_btn = gr.Button("▶ Train Selected Models", variant="secondary")
                                        sweep_status_box = gr.Markdown(
                                            "*Choose models and metrics, then run the sweep to build a comparison table.*"
                                        )
                                        sweep_table = gr.DataFrame(
                                            label="Model comparison table",
                                            interactive=False,
                                            wrap=True,
                                            line_breaks=True,
                                            height=320,
                                        )

                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML(_section_intro("Live training telemetry", "Watch progress, inspect the training trace, and monitor the learning curves while the job runs."))
                            eta_box       = gr.Textbox(label="ETA", interactive=False, max_lines=1)
                            live_metrics_card = gr.HTML(
                                value=_format_live_training_metrics_card()
                            )
                            status_box    = gr.Textbox(label="Training log", lines=16, interactive=False)
                            loss_plot     = gr.Plot(label="Loss & accuracy curves")

                        with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                            gr.HTML(_section_intro("Run summary", "When training completes, this area consolidates the narrative summary and architecture overview into one executive view."))
                            training_summary_out = gr.Markdown(
                                value="> ℹ️ Training summary will appear here once training is complete.")
                            arch_viz_out  = gr.Plot(label="Model architecture (parameter counts)")

            # ── Tab 4: Evaluate ───────────────────────────────────────────────────
            with gr.Tab("📊 Evaluate", elem_classes=["studio-tab-panel"]):
                gr.HTML(
                    """
<div class="studio-kpi-grid">
  <div class="studio-kpi"><strong>Core report</strong><span>Overall metrics and narrative summary</span></div>
  <div class="studio-kpi"><strong>Diagnostics</strong><span>Task-specific plots for classification, regression, clustering, and anomaly runs</span></div>
  <div class="studio-kpi"><strong>Failure analysis</strong><span>Misclassifications, explanations, and saliency tools</span></div>
</div>
"""
                )
                with gr.Group(elem_classes=["studio-card"]) as history_group:
                    gr.HTML(_section_intro("Evaluation report", "High-level metrics land here first so you have a readable summary before digging into technical plots."))
                    eval_kpi_bar = gr.HTML(
                        value=_format_eval_kpis("", [], "classification")
                    )
                    eval_summary_box = gr.Markdown(
                        value="> ℹ️ Evaluation metrics will appear here after training completes."
                    )

                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"], visible=True) as classification_views_group:
                            gr.HTML(_section_intro("Classification views", "Use these to inspect class-level performance and threshold behavior."))
                            cm_plot  = gr.Plot(label="Confusion Matrix")
                            roc_plot = gr.Plot(label="ROC / AUC Curves")
                            classification_note = gr.Markdown(_DEFAULT_CLASSIFICATION_NOTE)
                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"], visible=True) as explanation_group:
                            gr.HTML(_section_intro("Explanations", "Surface why the model behaved the way it did across visual, audio, and tabular workflows."))
                            gradcam_out = gr.Plot(label="Grad-CAM heatmap (image / audio / video)")
                            shap_out    = gr.Plot(label="SHAP feature importance (tabular)")
                            text_shap_out = gr.HTML(
                                value="<p style='color: var(--color-text-muted, #6a857c); font-style: italic;'>"
                                      "Token importance highlights will appear here after training a text model.</p>")
                            explanation_note = gr.Markdown(_DEFAULT_EXPLANATION_NOTE)

                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"], visible=False) as diagnostics_group:
                            gr.HTML(_section_intro("Prediction diagnostics", "Regression and anomaly plots help you read error shape, spread, and outlier behavior quickly."))
                            regression_plot_out = gr.Plot(label="Regression diagnostics (Predicted vs Actual · Residuals)")
                            anomaly_plot_out = gr.Plot(label="Anomaly detection — reconstruction errors")
                            diagnostics_note = gr.Markdown(_DEFAULT_DIAGNOSTICS_NOTE)
                    with gr.Column():
                        with gr.Group(elem_classes=["studio-card"], visible=True) as sample_review_group:
                            gr.HTML(_section_intro("Sample review", "Look at the model’s mistakes and cluster structure to understand where learning is breaking down."))
                            misclass_img = gr.Plot(label="Misclassified samples — visual")
                            misclass_df  = gr.DataFrame(
                                label="Misclassified samples — table",
                                wrap=True,
                                line_breaks=True,
                                height=320,
                                column_widths=["80px", "120px", "140px", "140px", "140px", "160px", "140px"],
                            )
                            sample_review_note = gr.Markdown(_DEFAULT_SAMPLE_REVIEW_NOTE)
                        with gr.Group(elem_classes=["studio-card"], visible=False) as clustering_group:
                            gr.HTML(_section_intro("Clustering review", "Inspect the learned structure and separation between clusters for clustering workflows."))
                            tsne_plot_out   = gr.Plot(label="t-SNE cluster visualisation")
                            clustering_note = gr.Markdown(
                                "*Clustering structure will appear here for clustering runs.*"
                            )

            # ── Tab 5: Dashboard ──────────────────────────────────────────────────
            with gr.Tab("📈 Dashboard", elem_classes=["studio-tab-panel"]):
                with gr.Group(elem_classes=["studio-card"]):
                    gr.HTML(_section_intro(
                        "Executive Dashboard",
                        "Consolidated view of your training run. Key metrics, the most relevant diagnostic, explainability, and actionable recommendations — all in one place."))
                    dashboard_kpi = gr.HTML(
                        value="<p style='color: var(--color-text-muted, #6a857c); font-style: italic;'>"
                              "Train a model to populate the dashboard.</p>")

                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML(_section_intro("Primary Diagnostic", "The most relevant chart for your task type."))
                            dashboard_primary_chart = gr.Plot(label="Diagnostic chart")
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML(_section_intro("Explainability", "Why the model made its decisions."))
                            dashboard_explain_plot = gr.Plot(label="Explanation chart")
                            dashboard_explain_html = gr.HTML(value="")
                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML(_section_intro("Training Curves", "Loss and metric progression across epochs."))
                            dashboard_loss_chart = gr.Plot(label="Loss & accuracy")
                        with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                            dashboard_actions = gr.Markdown(
                                "### Recommendations\n\n*Train a model to see recommendations.*")
                        with gr.Group(elem_classes=["studio-card"]):
                            streamlit_btn = gr.Button("🚀 Export Streamlit Dashboard", variant="secondary")
                            streamlit_status = gr.Markdown("")

            # ── Tab 6: Try Your Model ─────────────────────────────────────────────
            with gr.Tab("🧪 Try Your Model", elem_classes=["studio-tab-panel"]):
                with gr.Group(elem_classes=["studio-card"]):
                    gr.HTML(_section_intro("Interactive inference", "Load the latest bundle, try new examples, and inspect confidence scores without leaving the workspace."))
                    infer_bundle_path = gr.Dropdown(
                        choices=[],
                        label="Bundle path (select a saved bundle or paste a path)",
                        info=tip("infer_bundle_path"),
                        allow_custom_value=True,
                    )
                    infer_refresh_btn = gr.Button("🔄 Refresh bundles", variant="secondary", size="sm")

                with gr.Tabs():
                    with gr.Tab("🖼 Image"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card"]):
                                    gr.HTML(_section_intro("Image prediction", "Upload a single image and review the ranked class predictions."))
                                    infer_img_input = gr.Image(type="filepath", label="Upload image",
                                                                sources=["upload", "clipboard"])
                                    infer_img_btn   = gr.Button("🔮 Predict", variant="primary")
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                    infer_img_md    = gr.Markdown("*Upload an image and click Predict.*")
                                    infer_img_chart = gr.Plot(label="Confidence scores")

                    with gr.Tab("✍️ Text"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card"]):
                                    gr.HTML(_section_intro("Text prediction", "Paste a sample sentence or paragraph and run the trained classifier against it."))
                                    infer_text_input = gr.Textbox(label="Enter text", lines=4,
                                                                   placeholder="Type or paste text here…",
                                                                   info=tip("infer_text_input"))
                                    infer_text_btn   = gr.Button("🔮 Predict", variant="primary")
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                    infer_text_md    = gr.Markdown("*Enter text and click Predict.*")
                                    infer_text_chart = gr.Plot(label="Confidence scores")
                                    infer_text_explain = gr.HTML(value="")

                    with gr.Tab("📊 Tabular"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card"]):
                                    gr.HTML(_section_intro("Tabular prediction", "Provide feature values as JSON so the model can score a single record."))
                                    gr.Markdown(
                                        "Enter feature values as JSON, e.g. "
                                        "`{\"age\": 25, \"income\": 50000, \"score\": 0.8}`"
                                    )
                                    infer_tab_fill_btn = gr.Button("✨ Use detected sample", variant="secondary")
                                    infer_tab_input = gr.Textbox(label="Feature values (JSON)", lines=5,
                                                                  placeholder='{"feature1": value, …}',
                                                                  info=tip("infer_tab_input"))
                                    infer_tab_btn   = gr.Button("🔮 Predict", variant="primary")
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                    infer_tab_md    = gr.Markdown("*Enter features and click Predict.*")
                                    infer_tab_chart = gr.Plot(label="Confidence scores")

                    with gr.Tab("📈 Time-series"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card"]):
                                    gr.HTML(_section_intro("Time-series prediction", "Provide a short JSON window of sequential feature rows and score it with the current bundle."))
                                    gr.Markdown(
                                        "Enter a JSON array like "
                                        "`[{\"sensor_a\": 0.12, \"sensor_b\": 1.4}, {\"sensor_a\": 0.18, \"sensor_b\": 1.3}]`"
                                    )
                                    infer_ts_fill_btn = gr.Button("✨ Use detected window sample", variant="secondary")
                                    infer_ts_input = gr.Textbox(label="Time-series window (JSON)", lines=7,
                                                                placeholder='[{"feature1": value, "feature2": value}, …]',
                                                                info=tip("infer_ts_input"))
                                    infer_ts_btn = gr.Button("🔮 Predict", variant="primary")
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                    infer_ts_md = gr.Markdown("*Enter a window and click Predict.*")
                                    infer_ts_chart = gr.Plot(label="Prediction summary")

                    with gr.Tab("🔊 Audio"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card"]):
                                    gr.HTML(_section_intro("Audio prediction", "Upload a single clip to test the current bundle on new audio input."))
                                    infer_aud_input = gr.Audio(type="filepath", label="Upload audio")
                                    infer_aud_btn   = gr.Button("🔮 Predict", variant="primary")
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                    infer_aud_md    = gr.Markdown("*Upload audio and click Predict.*")
                                    infer_aud_chart = gr.Plot(label="Confidence scores")

                    with gr.Tab("📁 Batch"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card"]):
                                    gr.HTML(_section_intro(
                                        "Batch prediction",
                                        "Point at a folder of files — the app detects the modality from the "
                                        "loaded bundle and scores every matching file, then lets you download the results as CSV.",
                                    ))
                                    batch_folder = gr.Textbox(
                                        label="Input folder path",
                                        placeholder="/path/to/folder of images (or audio, text, video…)",
                                        info="Paste an absolute path to a folder containing files to classify.",
                                    )
                                    batch_run_btn = gr.Button("🔮 Run Batch Prediction", variant="primary")
                            with gr.Column():
                                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                    batch_status = gr.Markdown("*Set a bundle and folder path, then click Run.*")
                                    batch_df = gr.Dataframe(
                                        headers=["file", "prediction", "confidence"],
                                        datatype=["str", "str", "str"],
                                        row_count=(1, "dynamic"),
                                        col_count=(3, "fixed"),
                                        interactive=False,
                                        wrap=True,
                                        label="Predictions",
                                    )
                                    batch_save_btn = gr.Button("⬇️ Save CSV to outputs/", variant="secondary")
                                    batch_save_status = gr.Markdown("")

            # ── Tab 6: Export & History ───────────────────────────────────────────
            with gr.Tab("📦 Export & History", elem_classes=["studio-tab-panel"]):
                with gr.Group(elem_classes=["studio-card"]):
                    gr.HTML(_section_intro("Bundle outputs", "Generate deployment artifacts, inspect the saved bundle path, and keep track of previous runs from one place."))
                    bundle_out = gr.Textbox(label="Saved bundle path", interactive=False)
                    with gr.Row():
                        fastapi_btn    = gr.Button("⚡ Generate FastAPI Server", variant="secondary")
                        model_card_btn = gr.Button("📄 Generate Model Card",    variant="secondary")
                        docker_btn     = gr.Button("🐳 Generate Docker Bundle", variant="secondary")
                    with gr.Row():
                        fastapi_status_box    = gr.Markdown("*FastAPI export status will appear here.*")
                        model_card_status_box = gr.Markdown("*Model card generation status will appear here.*")
                        docker_status_box     = gr.Markdown("*Docker bundle status will appear here.*")

                with gr.Group(elem_classes=["studio-card"]):
                    gr.HTML(_section_intro("Run history", "Review earlier training runs and compare selected experiments side by side."))
                    refresh_btn  = gr.Button("🔄 Refresh History")
                    history_df   = gr.Dataframe(
                        headers=HISTORY_COLUMNS,
                        datatype=["number", "str", "str", "str", "str", "str", "number", "number", "str"],
                        row_count=(1, "dynamic"),
                        col_count=(len(HISTORY_COLUMNS), "fixed"),
                        interactive=False,
                        wrap=True,
                        line_breaks=True,
                        height=320,
                        column_widths=["70px", "150px", "120px", "110px", "130px", "120px", "90px", "90px", "360px"],
                        label="All training runs",
                    )
                    compare_ids  = gr.Textbox(value="",
                                               label="Compare run IDs (comma-separated, e.g. 0,1,2)",
                                               info=tip("compare_ids"))
                    compare_btn  = gr.Button("📈 Compare Selected Runs")
                    compare_status_box = gr.Markdown("*Select one or more run IDs to generate a comparison plot.*")
                    compare_plot = gr.Plot(label="Run comparison")

            # ── Tab 7: Object Detection ───────────────────────────────────────────
            with gr.Tab("🎯 Object Detection"):
                gr.Markdown(
                    """
    ### YOLO Object Detection
    Upload an **image** or **video** to detect objects using a pretrained YOLOv8 model,
    or **train a custom YOLO classifier** on your own image dataset.
    No extra setup — weights are downloaded automatically on first use.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        det_model = gr.Dropdown(
                            choices=list(YOLO_MODELS.keys()),
                            value=list(YOLO_MODELS.keys())[0],
                            label="Model",
                            info=tip("det_model"),
                        )
                        det_conf  = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                              label="Confidence threshold",
                                              info=tip("det_conf"))
                        det_iou   = gr.Slider(0.05, 0.95, value=0.45, step=0.05,
                                              label="IoU threshold (NMS)",
                                              info=tip("det_iou"))

                with gr.Tabs():

                    # ── Inference: Image ──────────────────────────────────────────
                    with gr.Tab("🖼 Image"):
                        with gr.Row():
                            with gr.Column():
                                det_img_input  = gr.Image(type="filepath",
                                                           label="Upload image",
                                                           sources=["upload", "clipboard"])
                                det_img_btn    = gr.Button("🔍 Detect Objects", variant="primary")
                            with gr.Column():
                                det_img_output = gr.Image(label="Detected objects",
                                                           interactive=False)
                        det_img_stats_md = gr.Markdown("*Upload an image and click Detect.*")
                        det_img_chart    = gr.Plot(label="Detection breakdown")

                    # ── Inference: Video ──────────────────────────────────────────
                    with gr.Tab("🎬 Video"):
                        with gr.Row():
                            with gr.Column():
                                det_vid_input  = gr.Video(label="Upload video",
                                                           sources=["upload"],
                                                           format="mp4")
                                det_vid_btn    = gr.Button("▶ Detect Objects in Video",
                                                            variant="primary")
                            with gr.Column():
                                det_vid_output = gr.Video(label="Annotated video",
                                                           interactive=False)
                        det_vid_status   = gr.Textbox(label="Progress", lines=3,
                                                       interactive=False)
                        det_vid_stats_md = gr.Markdown("*Upload a video and click Detect.*")
                        det_vid_chart    = gr.Plot(label="Detection breakdown")

                    # ── Custom Training ───────────────────────────────────────────
                    with gr.Tab("🏋 Train Custom Classifier"):
                        gr.Markdown(
                            """
    Train YOLOv8 as an **image classifier** on your own classes.
    Organise your images as `dataset_folder/class_name/image.jpg` (same format as the Data tab),
    then point to the root folder below.
                            """
                        )
                        with gr.Row():
                            with gr.Column():
                                yolo_data_path = gr.Textbox(
                                    label="Dataset folder",
                                    placeholder="/path/to/image_class_folders",
                                    info=tip("yolo_data_dir"))
                                yolo_model_size = gr.Dropdown(
                                    choices=["YOLOv8 Nano", "YOLOv8 Small", "YOLOv8 Medium"],
                                    value="YOLOv8 Nano",
                                    label="Model size",
                                    info=tip("yolo_model_size"))
                                yolo_epochs   = gr.Slider(1, 100, value=20, step=1,
                                                           label="Epochs",
                                                           info=tip("yolo_epochs"))
                                yolo_batch    = gr.Slider(4, 64, value=16, step=4,
                                                           label="Batch size",
                                                           info=tip("yolo_batch"))
                                yolo_train_btn = gr.Button("▶ Train", variant="primary")
                            with gr.Column():
                                yolo_status_box = gr.Textbox(label="Training log", lines=12,
                                                              interactive=False)
                        yolo_result_md = gr.Markdown("*Start training to see results.*")

    # ── Dynamic callbacks ─────────────────────────────────────────────────────

    def update_modality_controls(mod):
        is_structured = mod in {"tabular", "text", "timeseries"}
        return (
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod in {"tabular", "timeseries"})),
            gr.update(visible=(mod == "timeseries")),
            gr.update(visible=(mod == "timeseries")),
            gr.update(visible=(mod == "video")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod not in ("image", "audio", "video"))),
            gr.update(visible=is_structured),
            gr.update(visible=is_structured),
            gr.update(visible=is_structured),
            gr.update(visible=(mod == "tabular")),
            gr.update(visible=(mod == "tabular")),
            gr.update(visible=(mod == "tabular")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "text")),
            gr.update(visible=(mod == "timeseries")),
            gr.update(visible=(mod == "timeseries")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "image")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "audio")),
            gr.update(visible=(mod == "video")),
            gr.update(visible=(mod == "tabular")),
        )

    @lru_cache(maxsize=32)
    def _read_structured_preview_cached(path_sig):
        if not path_sig[0]:
            return None

        candidate = path_sig[0]
        if path_sig[2]:
            structured = []
            for root, _dirs, files in os.walk(candidate):
                for name in files:
                    if name.lower().endswith((".csv", ".tsv", ".json")):
                        structured.append(os.path.join(root, name))
            if len(structured) == 1:
                candidate = structured[0]
            else:
                return None

        from data_pipeline.io_utils import read_structured_file

        lower = candidate.lower()
        if lower.endswith((".csv", ".tsv", ".json")):
            return read_structured_file(candidate, nrows=500)
        return None

    def _read_structured_preview(dpath):
        return _read_structured_preview_cached(_path_signature(dpath))

    @lru_cache(maxsize=32)
    def _compute_stats_cached(path_sig, modality_name, label_name, subset_enabled, subset_percent, subset_seed):
        from data_pipeline.stats import compute_stats
        return compute_stats(
            path_sig[0],
            modality_name,
            label_name,
            subset_enabled=bool(subset_enabled),
            subset_percent=float(subset_percent),
            subset_seed=int(subset_seed),
        )

    @lru_cache(maxsize=32)
    def _validate_dataset_cached(path_sig, modality_name, label_name, subset_enabled, subset_percent, subset_seed):
        from data_pipeline.validation import validate_dataset
        return validate_dataset(
            path_sig[0],
            modality_name,
            label_name,
            subset_enabled=bool(subset_enabled),
            subset_percent=float(subset_percent),
            subset_seed=int(subset_seed),
        )

    def _get_dataset_stats(dpath, mod, lcol, subset_enabled=False, subset_percent=100.0, subset_seed=42):
        return _compute_stats_cached(_path_signature(dpath), mod, lcol, bool(subset_enabled), float(subset_percent), int(subset_seed))

    def _get_dataset_validation(dpath, mod, lcol, subset_enabled=False, subset_percent=100.0, subset_seed=42):
        return _validate_dataset_cached(_path_signature(dpath), mod, lcol, bool(subset_enabled), float(subset_percent), int(subset_seed))

    def _column_choice_updates(dpath, mod, current_label, current_text, current_features, current_time):
        structured_modalities = {"tabular", "text", "timeseries"}
        if mod not in structured_modalities:
            return (
                gr.update(choices=[], value=current_label or "label"),
                gr.update(choices=[], value=current_text or "text"),
                gr.update(choices=[], value=current_features or []),
                gr.update(choices=[], value=current_time or ""),
            )

        try:
            df = _read_structured_preview(dpath)
        except Exception:
            df = None

        if df is None or df.empty:
            return (
                gr.update(choices=[], value=current_label or "label"),
                gr.update(choices=[], value=current_text or "text"),
                gr.update(choices=[], value=current_features or []),
                gr.update(choices=[], value=current_time or ""),
            )

        from data_pipeline.type_inference import infer_column_types, suggest_features_and_label

        choices = df.columns.tolist()
        suggestions = suggest_features_and_label(df)
        col_types = infer_column_types(df)
        datetime_cols = [col for col, inferred in col_types.items() if inferred == "datetime"]

        label_value = current_label if current_label in choices else None
        if not label_value:
            label_value = suggestions.get("suggested_label") or (choices[0] if choices else "label")

        text_value = current_text if current_text in choices else None
        if not text_value:
            text_value = suggestions.get("text_col") or ("text" if "text" in choices else (choices[0] if choices else "text"))

        time_value = current_time if current_time in choices else None
        if not time_value:
            time_value = datetime_cols[0] if datetime_cols else ""

        allowed_feature_choices = [c for c in choices if c not in {label_value, time_value, text_value}]
        if current_features:
            feature_value = [col for col in current_features if col in allowed_feature_choices]
        else:
            suggested_feature_cols = suggestions.get("feature_cols", []) or []
            feature_value = [col for col in suggested_feature_cols if col in allowed_feature_choices]
        if mod == "timeseries":
            feature_value = [col for col in feature_value if col != time_value]

        return (
            gr.update(choices=choices, value=label_value),
            gr.update(choices=choices, value=text_value),
            gr.update(choices=allowed_feature_choices, value=feature_value),
            gr.update(choices=[""] + choices, value=time_value),
        )

    def update_model_choices(mod, mode):
        choices = get_models(mod, mode)
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def update_task_choices(mod):
        choices = get_compatible_tasks(mod)
        if mod in ("tabular", "timeseries") and "regression" not in choices:
            choices = choices + ["regression"]
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def update_training_mode(mod, current_mode):
        available = get_modes(mod)
        if not available:
            return gr.update(choices=["fine-tune", "from_scratch"], value=current_mode)
        if current_mode in available and mod not in {"tabular", "timeseries"}:
            return gr.update(choices=available, value=current_mode)
        preferred = None
        if mod in {"tabular", "timeseries"} and "from_scratch" in available:
            preferred = "from_scratch"
        elif "fine-tune" in available:
            preferred = "fine-tune"
        else:
            preferred = available[0]
        return gr.update(choices=available, value=preferred)

    def update_clusters_visible(tsk):
        return gr.update(visible=(tsk == "clustering"))

    def update_class_weights_visible(tsk):
        """Class weights only apply to classification — hide for other tasks."""
        return gr.update(visible=(tsk == "classification"))

    def update_cv_significance_choices(mod, mode, current_model, task_name):
        if mod not in {"tabular", "timeseries"} or task_name not in {"classification", "regression"}:
            return gr.update(choices=[], value=[]), gr.update(
                choices=CLASSIFICATION_METRIC_OPTIONS,
                value="accuracy",
            )
        available = [
            name for name in get_models(mod, mode)
            if name not in {current_model, "Autoencoder"}
        ]
        metric_choices = (
            CLASSIFICATION_METRIC_OPTIONS
            if task_name == "classification"
            else REGRESSION_METRIC_OPTIONS
        )
        return (
            gr.update(choices=available, value=[]),
            gr.update(choices=metric_choices, value=metric_choices[0]),
        )

    def update_model_sweep_controls(mod, mode, current_model, task_name):
        if task_name not in {"classification", "regression"}:
            return (
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=None),
            )
        available_models = [
            name for name in get_models(mod, mode)
            if name != "Autoencoder"
        ]
        metric_choices = (
            CLASSIFICATION_METRIC_OPTIONS if task_name == "classification" else REGRESSION_METRIC_OPTIONS
        )
        default_metrics = (
            ["accuracy", "f1_macro", "auc_ovr"] if task_name == "classification" else ["mae", "rmse", "r2"]
        )
        sort_metric = default_metrics[0]
        return (
            gr.update(choices=available_models, value=[]),
            gr.update(choices=metric_choices, value=default_metrics),
            gr.update(choices=metric_choices, value=sort_metric),
        )

    def update_image_size_guidance(mod, model, current_size):
        if mod != "image":
            return gr.update(visible=False), gr.update(value="", visible=False)

        current = int(current_size or 160)
        if is_vit(model):
            return (
                gr.update(value=224, minimum=224, maximum=224, step=16, visible=True),
                gr.update(
                    value=(
                        "### Image size guidance\n\n"
                        "`ViT` models in this app require **224 × 224** inputs, so the size is locked automatically."
                    ),
                    visible=True,
                ),
            )

        suggested = current if current not in {224} else 160
        return (
            gr.update(value=suggested, minimum=64, maximum=384, step=16, visible=True),
            gr.update(
                value=(
                    "### Image size guidance\n\n"
                    "Smaller image sizes like **96–160** train faster and use less memory. "
                    "Larger sizes like **224–320** preserve more visual detail. "
                    "The selected size is saved into preprocessing and reused during inference and export."
                ),
                visible=True,
            ),
        )

    def update_modality_augmentation_defaults(mod, level):
        image_defaults = {
            "none":   (False, False, False, False, False, False),
            "light":  (True,  False, False, False, False, False),
            "medium": (True,  False, True,  True,  False, False),
            "heavy":  (True,  True,  True,  True,  True,  True),
        }
        audio_defaults = {
            "none":   (False, False, False, False, False),
            "light":  (True,  False, False, False, False),
            "medium": (True,  True,  True,  True,  False),
            "heavy":  (True,  True,  True,  True,  True),
        }
        img_vals = image_defaults.get(level, image_defaults["light"]) if mod == "image" else image_defaults["none"]
        aud_vals = audio_defaults.get(level, audio_defaults["light"]) if mod == "audio" else audio_defaults["none"]
        return tuple(gr.update(value=value) for value in (*img_vals, *aud_vals))

    def update_workspace_cockpit(dpath, mod, lcol, model, tsk, bname, latest_bundle):
        return _workspace_status_markdown(dpath, mod, lcol, model, tsk, bname, latest_bundle)

    def load_builtin_walkthrough(tutorial_modality: str):
        from ui.tutorials import (
            prepare_mnist_tutorial, prepare_iris_tutorial, prepare_newsgroups_tutorial,
            prepare_audio_tutorial, prepare_timeseries_tutorial, prepare_video_tutorial,
        )

        mod = tutorial_modality or "image"

        _PREPARE = {
            "image":      (prepare_mnist_tutorial,      _BUILTIN_IMAGE_TUTORIAL_PATH),
            "tabular":    (prepare_iris_tutorial,        _BUILTIN_TABULAR_TUTORIAL_PATH),
            "text":       (prepare_newsgroups_tutorial,  _BUILTIN_TEXT_TUTORIAL_PATH),
            "audio":      (prepare_audio_tutorial,       _BUILTIN_AUDIO_TUTORIAL_PATH),
            "timeseries": (prepare_timeseries_tutorial,  _BUILTIN_TIMESERIES_TUTORIAL_PATH),
            "video":      (prepare_video_tutorial,       _BUILTIN_VIDEO_TUTORIAL_PATH),
        }

        prepare_fn, dest = _PREPARE[mod]
        if not dest.exists() or not any(dest.rglob("*")):
            print(f"[tutorial] Preparing {mod} tutorial dataset …", flush=True)
            try:
                prepare_fn(dest)
                print(f"[tutorial] {mod} tutorial ready.", flush=True)
            except Exception as exc:
                raise gr.Error(f"Could not prepare {mod} tutorial: {exc}") from exc

        # Modality-specific configs: (data_path, image_size, augmentation, val_split,
        #   training_mode, lr, batch_size, epochs, dropout, optimizer, scheduler, class_weights, bundle_name)
        _CONFIGS: dict[str, tuple] = {
            "image":      (str(dest.resolve()),                          64, "none",  0.15, "from_scratch", 1e-3, 16, 5,  0.2, "adam", "none", False, "mnist_tutorial"),
            "tabular":    (str((dest / "iris.csv").resolve()),           64, "none",  0.20, "from_scratch", 1e-3, 32, 20, 0.0, "adam", "none", False, "iris_tutorial"),
            "text":       (str((dest / "newsgroups.csv").resolve()),     64, "none",  0.15, "from_scratch", 2e-4, 32, 5,  0.1, "adam", "none", False, "newsgroups_tutorial"),
            "audio":      (str(dest.resolve()),                          64, "none",  0.15, "from_scratch", 1e-3, 16, 10, 0.2, "adam", "none", False, "speechcommands_tutorial"),
            "timeseries": (str((dest / "timeseries.csv").resolve()),     64, "none",  0.20, "from_scratch", 1e-3, 32, 15, 0.1, "adam", "none", False, "timeseries_tutorial"),
            "video":      (str(dest.resolve()),                          64, "none",  0.20, "from_scratch", 1e-3, 8,  10, 0.2, "adam", "none", False, "video_tutorial"),
        }

        cfg = _CONFIGS[mod]
        return (mod, *cfg, _tutorial_markdown(mod, loaded=True))

    def apply_builtin_walkthrough_model(mod, mode):
        model_choices = get_models(mod, mode)
        chosen_model = "TinyCNN" if "TinyCNN" in model_choices else (model_choices[0] if model_choices else None)
        task_choices = get_compatible_tasks(mod) if chosen_model else TASKS
        chosen_task = "classification" if "classification" in task_choices else (task_choices[0] if task_choices else None)
        return (
            gr.update(choices=model_choices, value=chosen_model),
            gr.update(choices=task_choices, value=chosen_task),
        )

    def update_evaluate_visibility(task_name, model_choice):
        is_classification = task_name in {"classification", "multi-label"}
        is_regression = task_name == "regression"
        is_clustering = task_name == "clustering"
        is_anomaly = task_name == "anomaly" or model_choice == "Autoencoder"
        return (
            gr.update(visible=is_classification),
            gr.update(visible=is_classification),
            gr.update(visible=is_regression or is_anomaly),
            gr.update(visible=is_classification),
            gr.update(visible=is_clustering),
        )

    def persist_project_mode(project_mode_val):
        save_project_state({
            **load_project_state(),
            "project_mode": project_mode_val,
        })
        return project_mode_val

    project_mode.change(
        persist_project_mode,
        inputs=project_mode,
        outputs=project_mode_state,
    )
    project_mode.change(
        _workflow_mode_updates,
        inputs=project_mode,
        outputs=[
            split_cleaning_group,
            advanced_training_group,
            sequence_options_group,
            sklearn_options_group,
            column_mapping_group,
            hyperparams_group,
            model_selection_group,
            evaluation_report_group,
            export_utilities_group,
            history_group,
            explanation_group_top,
        ],
    )
    project_mode.change(
        _why_this_matters_markdown,
        inputs=[project_mode, modality, model_name, task, recommendation_state],
        outputs=why_box,
    )

    modality.change(update_modality_controls, modality,
                    [
                        text_col, feature_cols, time_col, window_size, n_frames, image_size, image_size_hint, sample_rate, audio_image_size, label_col,
                        use_random_subset, subset_percent, subset_seed,
                        use_data_cleaning, tabular_missing_strategy, tabular_clip_outliers,
                        text_lowercase, text_strip_urls, text_strip_punctuation, text_remove_stopwords,
                        text_deduplicate, text_apply_stemming, text_apply_lemmatization, text_use_ngrams,
                        timeseries_sort_by_time, timeseries_fill_strategy,
                        image_verify_files, image_aug_flip, image_aug_vertical, image_aug_rotation,
                        image_aug_color, image_aug_gray, image_aug_perspective, image_normalization, image_force_grayscale,
                        audio_verify_files, audio_normalize_waveform, audio_n_mels,
                        audio_aug_noise, audio_aug_shift, audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask,
                        video_verify_files, tabular_scaling,
                    ])
    modality.change(update_training_mode, [modality, training_mode], training_mode)
    modality.change(
        _column_choice_updates,
        [data_path, modality, label_col, text_col, feature_cols, time_col],
        [label_col, text_col, feature_cols, time_col],
    )
    modality.change(update_model_choices, [modality, training_mode], model_name)
    modality.change(update_task_choices, modality, task)
    modality.change(
        update_cv_significance_choices,
        [modality, training_mode, model_name, task],
        [cv_significance_models, cv_significance_metric],
    )
    modality.change(
        update_model_sweep_controls,
        [modality, training_mode, model_name, task],
        [sweep_models, sweep_metrics, sweep_sort_metric],
    )
    modality.change(
        update_image_size_guidance,
        [modality, model_name, image_size],
        [image_size, image_size_hint],
    )
    modality.change(
        update_modality_augmentation_defaults,
        [modality, augmentation],
        [
            image_aug_flip, image_aug_vertical, image_aug_rotation,
            image_aug_color, image_aug_gray, image_aug_perspective,
            audio_aug_noise, audio_aug_shift, audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask,
        ],
    )
    modality.change(
        _why_this_matters_markdown,
        [project_mode, modality, model_name, task, recommendation_state],
        why_box,
    )
    training_mode.change(update_model_choices, [modality, training_mode], model_name)
    training_mode.change(
        update_cv_significance_choices,
        [modality, training_mode, model_name, task],
        [cv_significance_models, cv_significance_metric],
    )
    training_mode.change(
        update_model_sweep_controls,
        [modality, training_mode, model_name, task],
        [sweep_models, sweep_metrics, sweep_sort_metric],
    )
    model_name.change(
        update_cv_significance_choices,
        [modality, training_mode, model_name, task],
        [cv_significance_models, cv_significance_metric],
    )
    model_name.change(
        update_image_size_guidance,
        [modality, model_name, image_size],
        [image_size, image_size_hint],
    )
    model_name.change(
        update_model_sweep_controls,
        [modality, training_mode, model_name, task],
        [sweep_models, sweep_metrics, sweep_sort_metric],
    )
    model_name.change(
        update_evaluate_visibility,
        [task, model_name],
        [classification_views_group, explanation_group, diagnostics_group, sample_review_group, clustering_group],
    )
    model_name.change(
        _why_this_matters_markdown,
        [project_mode, modality, model_name, task, recommendation_state],
        why_box,
    )
    augmentation.change(
        update_modality_augmentation_defaults,
        [modality, augmentation],
        [
            image_aug_flip, image_aug_vertical, image_aug_rotation,
            image_aug_color, image_aug_gray, image_aug_perspective,
            audio_aug_noise, audio_aug_shift, audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask,
        ],
    )
    task.change(update_clusters_visible, task, n_clusters)
    task.change(update_class_weights_visible, task, use_class_weights)
    task.change(
        update_cv_significance_choices,
        [modality, training_mode, model_name, task],
        [cv_significance_models, cv_significance_metric],
    )
    task.change(
        update_model_sweep_controls,
        [modality, training_mode, model_name, task],
        [sweep_models, sweep_metrics, sweep_sort_metric],
    )
    task.change(
        update_evaluate_visibility,
        [task, model_name],
        [classification_views_group, explanation_group, diagnostics_group, sample_review_group, clustering_group],
    )
    task.change(
        _why_this_matters_markdown,
        [project_mode, modality, model_name, task, recommendation_state],
        why_box,
    )

    demo.load(update_modality_controls, modality,
                    [
                        text_col, feature_cols, time_col, window_size, n_frames, image_size, image_size_hint, sample_rate, audio_image_size, label_col,
                        use_random_subset, subset_percent, subset_seed,
                        use_data_cleaning, tabular_missing_strategy, tabular_clip_outliers,
                        text_lowercase, text_strip_urls, text_strip_punctuation, text_remove_stopwords,
                        text_deduplicate, text_apply_stemming, text_apply_lemmatization, text_use_ngrams,
                        timeseries_sort_by_time, timeseries_fill_strategy,
                        image_verify_files, image_aug_flip, image_aug_vertical, image_aug_rotation,
                        image_aug_color, image_aug_gray, image_aug_perspective, image_normalization, image_force_grayscale,
                        audio_verify_files, audio_normalize_waveform, audio_n_mels,
                        audio_aug_noise, audio_aug_shift, audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask,
                        video_verify_files, tabular_scaling,
                    ])
    demo.load(update_training_mode, [modality, training_mode], training_mode)
    demo.load(
        update_cv_significance_choices,
        [modality, training_mode, model_name, task],
        [cv_significance_models, cv_significance_metric],
    )
    demo.load(
        update_model_sweep_controls,
        [modality, training_mode, model_name, task],
        [sweep_models, sweep_metrics, sweep_sort_metric],
    )
    demo.load(
        update_image_size_guidance,
        [modality, model_name, image_size],
        [image_size, image_size_hint],
    )
    demo.load(
        update_modality_augmentation_defaults,
        [modality, augmentation],
        [
            image_aug_flip, image_aug_vertical, image_aug_rotation,
            image_aug_color, image_aug_gray, image_aug_perspective,
            audio_aug_noise, audio_aug_shift, audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask,
        ],
    )
    demo.load(update_model_choices, [modality, training_mode], model_name)
    demo.load(update_task_choices, modality, task)
    demo.load(
        update_workspace_cockpit,
        [data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        [workspace_snapshot, next_step_box],
    )
    demo.load(
        update_evaluate_visibility,
        [task, model_name],
        [classification_views_group, explanation_group, diagnostics_group, sample_review_group, clustering_group],
    )
    demo.load(
        _workflow_mode_updates,
        [project_mode],
        [
            split_cleaning_group,
            advanced_training_group,
            sequence_options_group,
            sklearn_options_group,
            column_mapping_group,
            hyperparams_group,
            model_selection_group,
            evaluation_report_group,
            export_utilities_group,
            history_group,
            explanation_group_top,
        ],
    )
    demo.load(
        _why_this_matters_markdown,
        [project_mode, modality, model_name, task, recommendation_state],
        why_box,
    )

    # ── File upload → auto-fill data path ────────────────────────────────────

    def handle_upload(uploaded_file, mod, current_label, current_text, current_features, current_time):
        """Extract zip → temp dir or use CSV path directly."""
        if uploaded_file is None:
            updates = _column_choice_updates("", mod, current_label, current_text, current_features, current_time)
            return "", *updates
        path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
        if path.lower().endswith(".zip"):
            tmp_dir = tempfile.mkdtemp(prefix="nocode_dl_")
            # Register cleanup so the temp directory is removed on process exit
            atexit.register(shutil.rmtree, tmp_dir, True)
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(tmp_dir)
            except zipfile.BadZipFile:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                updates = _column_choice_updates("", mod, current_label, current_text, current_features, current_time)
                return "", *updates
            structured = []
            for root, _dirs, files in os.walk(tmp_dir):
                for name in files:
                    if name.lower().endswith((".csv", ".tsv", ".json")):
                        structured.append(os.path.join(root, name))
            if len(structured) == 1:
                resolved_path = structured[0]
                updates = _column_choice_updates(resolved_path, mod, current_label, current_text, current_features, current_time)
                return resolved_path, *updates
            # If zip contained a single top-level folder, use that
            entries = os.listdir(tmp_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
                resolved_path = os.path.join(tmp_dir, entries[0])
                updates = _column_choice_updates(resolved_path, mod, current_label, current_text, current_features, current_time)
                return resolved_path, *updates
            updates = _column_choice_updates(tmp_dir, mod, current_label, current_text, current_features, current_time)
            return tmp_dir, *updates
        updates = _column_choice_updates(path, mod, current_label, current_text, current_features, current_time)
        return path, *updates  # CSV / TSV / JSON: use the path directly

    data_upload.change(
        handle_upload,
        inputs=[data_upload, modality, label_col, text_col, feature_cols, time_col],
        outputs=[data_path, label_col, text_col, feature_cols, time_col],
    )
    data_path.change(
        _column_choice_updates,
        inputs=[data_path, modality, label_col, text_col, feature_cols, time_col],
        outputs=[label_col, text_col, feature_cols, time_col],
    )
    data_path.change(
        suggest_bundle_name,
        inputs=[data_path, modality, model_name, task, bundle_name],
        outputs=bundle_name,
    )
    data_path.change(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )
    label_col.change(
        _column_choice_updates,
        inputs=[data_path, modality, label_col, text_col, feature_cols, time_col],
        outputs=[label_col, text_col, feature_cols, time_col],
    )
    label_col.change(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )
    text_col.change(
        _column_choice_updates,
        inputs=[data_path, modality, label_col, text_col, feature_cols, time_col],
        outputs=[label_col, text_col, feature_cols, time_col],
    )
    time_col.change(
        _column_choice_updates,
        inputs=[data_path, modality, label_col, text_col, feature_cols, time_col],
        outputs=[label_col, text_col, feature_cols, time_col],
    )
    model_name.change(
        suggest_bundle_name,
        inputs=[data_path, modality, model_name, task, bundle_name],
        outputs=bundle_name,
    )
    model_name.change(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )
    task.change(
        suggest_bundle_name,
        inputs=[data_path, modality, model_name, task, bundle_name],
        outputs=bundle_name,
    )
    task.change(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )
    bundle_name.change(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )

    # ── Data preview ──────────────────────────────────────────────────────────

    def preview_dataset(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val, project_mode_val, model_val, task_val):
        if not dpath:
            empty_report = {
                "status": "blocked",
                "blocking_issues": ["No data path provided."],
                "warnings": [],
                "info": [],
                "suggested_actions": ["Upload a dataset or paste a local path before previewing."],
            }
            return "No path provided.", None, "Enter a data path above.", "### Data Quality Report\n\n- No data path provided.", empty_report, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})
        try:
            path = os.path.abspath(str(dpath))

            structured_suffixes = (".csv", ".tsv", ".json")
            is_structured_file = os.path.isfile(path) and path.lower().endswith(structured_suffixes)
            is_directory = os.path.isdir(path)

            if mod in {"image", "audio", "video"} and is_structured_file:
                pretty_mod = mod.capitalize()
                return (
                    "Dataset preview unavailable for the current modality.",
                    None,
                    (
                        f"### {pretty_mod} dataset mismatch\n\n"
                        f"You selected **{mod}** mode, but the path points to a structured file:\n\n"
                        f"`{path}`\n\n"
                        "Switch the modality to **tabular**, **text**, or **timeseries**, "
                            "or provide a folder of class-organised media files instead."
                    ),
                    "### Data Quality Report\n\n- Preview is blocked until the modality and data source match.",
                    {"status": "blocked", "blocking_issues": ["Modality and data source do not match."], "warnings": [], "info": [], "suggested_actions": ["Switch modality or point to the expected dataset format."]},
                    _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
                )

            if mod in {"tabular", "text", "timeseries"} and is_directory:
                structured_preview = _read_structured_preview(path)
                if structured_preview is None:
                    return (
                        "Structured dataset preview unavailable.",
                        None,
                        (
                            "### Structured data not found\n\n"
                            "The selected path is a folder, but I could not find a single CSV, TSV, "
                            "or JSON file to inspect inside it.\n\n"
                            "Point to a structured data file directly, or upload a ZIP containing one."
                        ),
                        "### Data Quality Report\n\n- A structured file could not be located inside this folder.",
                        {"status": "blocked", "blocking_issues": ["No single structured data file was found."], "warnings": [], "info": [], "suggested_actions": ["Point to a CSV, TSV, or JSON file directly."]},
                        _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
                    )

            from data_pipeline.stats import plot_class_distribution
            from data_pipeline.quality_report import build_quality_report, quality_report_markdown
            stats    = _get_dataset_stats(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val)
            warnings = _get_dataset_validation(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val)
            fig      = plot_class_distribution(stats)
            warn_txt = "\n".join(warnings) if warnings else "✅  No issues detected."
            preview_df = _read_structured_preview(dpath) if mod in {"tabular", "text", "timeseries"} else None
            if preview_df is not None and bool(subset_enabled) and float(subset_pct) < 100:
                from data_pipeline.io_utils import apply_random_subset
                preview_df, _ = apply_random_subset(
                    preview_df,
                    enabled=True,
                    subset_percent=float(subset_pct),
                    subset_seed=int(subset_seed_val),
                )
            quality_report = build_quality_report(path, mod, lcol, stats, warnings, preview_df)

            # Append column type inference for tabular
            if mod == "tabular" and not stats.get("error"):
                try:
                    from data_pipeline.type_inference import (
                        infer_column_types, suggest_features_and_label,
                        type_inference_markdown)
                    from data_pipeline.io_utils import read_structured_file
                    df         = read_structured_file(dpath, nrows=500)
                    col_types  = infer_column_types(df)
                    suggestions = suggest_features_and_label(df)
                    warn_txt   += "\n\n" + type_inference_markdown(col_types, suggestions)
                except Exception:
                    pass
            save_project_state({
                "project_mode": project_mode_val,
                "dataset_path": path,
                "modality": mod,
                "label_col": lcol,
                "random_subset": {
                    "enabled": bool(subset_enabled),
                    "percent": float(subset_pct),
                    "seed": int(subset_seed_val),
                },
                "model_name": model_val,
                "task": task_val,
                "quality_report": quality_report,
            })
            return (
                stats.get("summary", str(stats.get("error", ""))),
                fig,
                warn_txt,
                quality_report_markdown(quality_report),
                quality_report,
                _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
            )
        except Exception as e:
            pretty_mod = mod.capitalize() if isinstance(mod, str) else "Dataset"
            return (
                f"{pretty_mod} preview could not be generated.",
                None,
                (
                    f"### Preview error\n\n"
                    f"`{e}`\n\n"
                    "Check that the path exists, matches the selected modality, and contains "
                    "the expected files."
                ),
                f"### Data Quality Report\n\n- Preview failed: `{e}`",
                {"status": "blocked", "blocking_issues": [str(e)], "warnings": [], "info": [], "suggested_actions": ["Fix the preview error before training."]},
                _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
            )

    preview_event = preview_btn.click(preview_dataset,
                                      inputs=[data_path, modality, label_col, use_random_subset, subset_percent, subset_seed, project_mode, model_name, task],
                                      outputs=[stats_summary_box, dist_plot, validation_box, quality_report_box, quality_report_state, why_box])

    # ── Guided recommendations ────────────────────────────────────────────────

    def get_recommendations(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val, project_mode_val, model_val, task_val, quality_report_val):
        if not dpath:
            return "*Provide a data path first.*", "*Preview a dataset to explain the recommendation.*", {}, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})
        try:
            from ui.guided_mode import (
                recommend,
                format_recommendation_summary,
                format_recommendation_rationale,
            )
            stats = _get_dataset_stats(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val)
            if "error" in stats:
                return f"Dataset error: {stats['error']}", "*Recommendation unavailable while dataset preview has errors.*", {}, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})
            rec = recommend(mod, stats, DEVICE)
            if isinstance(quality_report_val, dict) and quality_report_val.get("warnings"):
                rec.setdefault("warnings", [])
                rec["warnings"].extend(quality_report_val.get("warnings", [])[:2])
            save_project_state({
                **load_project_state(),
                "project_mode": project_mode_val,
                "recommendation": rec,
            })
            return (
                format_recommendation_summary(rec),
                format_recommendation_rationale(rec),
                rec,
                _why_this_matters_markdown(project_mode_val, mod, model_val or rec.get("model_name", ""), task_val or rec.get("task", "classification"), rec),
            )
        except Exception as e:
            return f"Error: {e}", "*Recommendation engine failed unexpectedly.*", {}, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})

    recommend_btn.click(get_recommendations,
                        inputs=[data_path, modality, label_col, use_random_subset, subset_percent, subset_seed, project_mode, model_name, task, quality_report_state],
                        outputs=[recommendations_box, recommendation_rationale_box, recommendation_state, why_box])
    preview_event.then(
        get_recommendations,
        inputs=[data_path, modality, label_col, use_random_subset, subset_percent, subset_seed, project_mode, model_name, task, quality_report_state],
        outputs=[recommendations_box, recommendation_rationale_box, recommendation_state, why_box],
    )
    preview_event.then(
        _suggest_tabular_example,
        inputs=[data_path, modality, label_col, infer_tab_input],
        outputs=infer_tab_input,
    )
    preview_event.then(
        suggest_bundle_name,
        inputs=[data_path, modality, model_name, task, bundle_name],
        outputs=bundle_name,
    )
    preview_event.then(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )

    def apply_recommendations(modality_val, rec, project_mode_val, data_path_val, label_col_val, bundle_name_val, bundle_path_val):
        updates = _apply_recommendation_payload(modality_val, rec)
        selected_model = rec.get("model_name", "")
        selected_task = rec.get("task", "classification")
        save_project_state({
            **load_project_state(),
            "project_mode": project_mode_val,
            "recommendation": rec,
            "recommendation_applied": True,
        })
        snapshot, next_step = _workspace_status_markdown(
            data_path_val,
            modality_val,
            label_col_val,
            selected_model,
            selected_task,
            bundle_name_val,
            bundle_path_val,
        )
        why_text = _why_this_matters_markdown(project_mode_val, modality_val, selected_model, selected_task, rec)
        return (*updates, snapshot, next_step, why_text)

    apply_recommendations_btn.click(
        apply_recommendations,
        inputs=[modality, recommendation_state, project_mode, data_path, label_col, bundle_name, bundle_out],
        outputs=[
            training_mode, model_name, task, augmentation, image_size, batch_size, epochs, dropout, scheduler_name, use_class_weights,
            workspace_snapshot, next_step_box, why_box,
        ],
    )

    tutorial_modality_dropdown.change(
        lambda m: _tutorial_markdown(m),
        inputs=[tutorial_modality_dropdown],
        outputs=[tutorial_box],
    )

    tutorial_event = load_tutorial_btn.click(
        load_builtin_walkthrough,
        inputs=[tutorial_modality_dropdown],
        outputs=[
            modality,
            data_path,
            image_size,
            augmentation,
            val_split,
            training_mode,
            lr,
            batch_size,
            epochs,
            dropout,
            optimizer,
            scheduler_name,
            use_class_weights,
            bundle_name,
            tutorial_box,
        ],
    )
    tutorial_event.then(
        apply_builtin_walkthrough_model,
        inputs=[modality, training_mode],
        outputs=[model_name, task],
    )
    tutorial_event.then(
        preview_dataset,
        inputs=[data_path, modality, label_col, use_random_subset, subset_percent, subset_seed, project_mode, model_name, task],
        outputs=[stats_summary_box, dist_plot, validation_box, quality_report_box, quality_report_state, why_box],
    )
    tutorial_event.then(
        get_recommendations,
        inputs=[data_path, modality, label_col, use_random_subset, subset_percent, subset_seed, project_mode, model_name, task, quality_report_state],
        outputs=[recommendations_box, recommendation_rationale_box, recommendation_state, why_box],
    )
    tutorial_event.then(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )

    # ── Train button ──────────────────────────────────────────────────────────

    train_event = train_btn.click(
        run_pipeline,
        inputs=[
            modality, data_path, text_col, label_col, feature_cols, time_col,
            window_size, n_frames, image_size, sample_rate, audio_image_size, audio_n_mels, augmentation,
            val_split, use_random_subset, subset_percent, subset_seed, use_data_cleaning, tabular_missing_strategy, tabular_clip_outliers, tabular_scaling,
            text_lowercase, text_strip_urls, text_strip_punctuation, text_remove_stopwords,
            text_deduplicate, text_apply_stemming, text_apply_lemmatization, text_use_ngrams,
            timeseries_sort_by_time, timeseries_fill_strategy,
            image_verify_files, image_aug_flip, image_aug_vertical, image_aug_rotation,
            image_aug_color, image_aug_gray, image_aug_perspective, image_normalization, image_force_grayscale,
            audio_verify_files, audio_normalize_waveform, audio_aug_noise, audio_aug_shift,
            audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask, video_verify_files,
            training_mode, model_name, task,
            epochs, lr, batch_size, dropout, optimizer, scheduler_name, use_amp,
            hidden_size, num_layers,
            n_estimators, max_depth, C_param, max_iter, lr_xgb,
            n_clusters,
            use_class_weights, checkpoint_every,
            bundle_name,
        ],
        outputs=[
            status_box, loss_plot, eta_box, live_metrics_card,
            eval_summary_box, cm_plot, roc_plot, tsne_plot_out,
            gradcam_out, shap_out, misclass_img, misclass_df,
            bundle_out, arch_viz_out,
            training_summary_out, regression_plot_out, anomaly_plot_out, eval_kpi_bar,
            classification_note, explanation_note, diagnostics_note, sample_review_note,
            text_shap_out,
            dashboard_kpi, dashboard_primary_chart, dashboard_explain_plot,
            dashboard_explain_html, dashboard_loss_chart, dashboard_actions,
        ],
    )

    # Stop button cancels the running training generator
    stop_btn.click(None, cancels=[train_event])

    # Auto-fill inference bundle path from training result
    bundle_out.change(
        lambda p: gr.update(choices=_list_bundles(), value=p),
        inputs=bundle_out, outputs=infer_bundle_path)
    infer_refresh_btn.click(
        lambda: gr.update(choices=_list_bundles()),
        outputs=infer_bundle_path)
    bundle_out.change(
        update_workspace_cockpit,
        inputs=[data_path, modality, label_col, model_name, task, bundle_name, bundle_out],
        outputs=[workspace_snapshot, next_step_box],
    )
    bundle_out.change(
        _suggest_tabular_example,
        inputs=[data_path, modality, label_col, infer_tab_input],
        outputs=infer_tab_input,
    )

    # ── Cross-validation callback ─────────────────────────────────────────────

    def run_cv(modality_val, data_path_val, text_col_val, label_col_val,
               time_col_val, window_size_val, n_frames_val, sample_rate_val,
               augmentation_val, val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val, model_name_val,
               task_val, epochs_val, lr_val, batch_size_val, dropout_val,
               optimizer_val, scheduler_val, use_amp_val,
               hidden_size_val, num_layers_val, cv_k_val):
        if not data_path_val or not model_name_val:
            yield "❌ Set Data path and Model before running cross-validation."
            return

        yield f"Starting {int(cv_k_val)}-fold cross-validation…\n"
        lines = []

        try:
            from training.cross_val import cross_validate, format_cv_results

            # Build a model factory that captures the current UI settings
            def _model_factory(num_classes: int, input_size: int = 1):
                if is_sklearn(model_name_val):
                    from models.tabular_models import get_tabular_model
                    return get_tabular_model(model_name_val, num_classes=num_classes,
                                             input_size=input_size,
                                             task=task_val,
                                             dropout=float(dropout_val))
                elif modality_val == "text":
                    from models.text_models import get_text_model
                    return get_text_model(model_name_val, num_classes=num_classes,
                                          mode=training_mode_val,
                                          hidden_size=int(hidden_size_val),
                                          num_layers=int(num_layers_val),
                                          dropout=float(dropout_val))
                elif modality_val == "tabular":
                    from models.tabular_models import get_tabular_model
                    return get_tabular_model(model_name_val, num_classes=num_classes,
                                             input_size=input_size,
                                             dropout=float(dropout_val))
                elif modality_val == "timeseries":
                    from models.timeseries_models import get_timeseries_model
                    return get_timeseries_model(
                        model_name_val,
                        num_classes=num_classes,
                        input_size=input_size,
                        hidden_size=int(hidden_size_val),
                        num_layers=int(num_layers_val),
                        dropout=float(dropout_val),
                    )
                else:
                    from models.image_models import get_image_model
                    return get_image_model(model_name_val, num_classes=num_classes,
                                           mode=training_mode_val,
                                           dropout=float(dropout_val))

            # Load the full dataset for cross-validation
            if modality_val == "tabular":
                import pandas as pd, numpy as np
                import torch
                from torch.utils.data import TensorDataset
                from modalities.tabular import load_tabular_data
                # We need the full dataset; load with a tiny val_split and combine
                tl, vl, classes, prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
                    data_path_val, label_col=label_col_val, batch_size=int(batch_size_val),
                    val_split=0.01,
                    subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                    subset_seed=int(subset_seed_val))
                X_all = np.concatenate([Xtr, Xv])
                y_all = np.concatenate([ytr, yv])
                full_dataset = TensorDataset(
                    torch.tensor(X_all), torch.tensor(y_all.astype("int64")))
                num_classes  = len(classes)

                for prog in cross_validate(
                    full_dataset,
                    model_factory=lambda: _model_factory(num_classes, input_size),
                    k=int(cv_k_val),
                    epochs=int(epochs_val), lr=float(lr_val),
                    optimizer_name=optimizer_val, task=task_val,
                    batch_size=int(batch_size_val),
                ):
                    if prog.get("done"):
                        result = format_cv_results(
                            prog["fold_accs"],
                            prog["mean_acc"],
                            prog["std_acc"],
                            fold_metrics=prog.get("fold_metrics"),
                            aggregate_metrics=prog.get("aggregate_metrics"),
                            task=task_val,
                        )
                        if prog.get("warning"):
                            result = f"> ⚠️ {prog['warning']}\n\n" + result
                        lines.append(result)
                    else:
                        fold_line = (
                            f"Fold {prog['fold']}/{prog['total_folds']}  "
                            f"epoch {prog['epoch']}  val_metric={prog.get('val_acc', 0):.2f}"
                            + ("%" if task_val == "classification" else "")
                        )
                        lines.append(fold_line)
                    yield "\n".join(lines)
            elif modality_val == "timeseries":
                import numpy as np
                import torch
                from torch.utils.data import TensorDataset
                from modalities.timeseries import prepare_timeseries_windows

                windows, labels, classes, prep, input_size = prepare_timeseries_windows(
                    data_path=data_path_val,
                    label_col=label_col_val,
                    time_col=time_col_val if time_col_val else None,
                    window_size=int(window_size_val),
                    task=task_val,
                )
                num_classes = len(classes) if task_val == "classification" else 1

                if is_sklearn(model_name_val):
                    X_all = windows.reshape(len(windows), -1).astype(np.float32)
                    y_all = labels
                    from sklearn.model_selection import KFold, StratifiedKFold
                    from eval.metrics import classification_metric_summary, regression_metric_summary

                    split_warning = None
                    if task_val == "classification":
                        _, counts = np.unique(y_all, return_counts=True)
                        min_class_size = int(counts.min()) if len(counts) else 0
                        if min_class_size >= int(cv_k_val):
                            splitter = StratifiedKFold(n_splits=int(cv_k_val), shuffle=True, random_state=42)
                            splits = list(splitter.split(X_all, y_all))
                        else:
                            splitter = KFold(n_splits=int(cv_k_val), shuffle=True, random_state=42)
                            splits = list(splitter.split(X_all))
                            split_warning = (
                                f"Using plain K-Fold instead of StratifiedKFold because the smallest class has only {min_class_size} samples."
                            )
                    else:
                        splitter = KFold(n_splits=int(cv_k_val), shuffle=True, random_state=42)
                        splits = list(splitter.split(X_all))

                    fold_scores = []
                    fold_metrics = []
                    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
                        model = _model_factory(num_classes=num_classes, input_size=X_all.shape[1])
                        X_train = X_all[train_idx]
                        y_train = y_all[train_idx]
                        X_val = X_all[val_idx]
                        y_val = y_all[val_idx]
                        lines.append(f"Fold {fold_idx}/{int(cv_k_val)}  fitting classical baseline…")
                        yield "\n".join(lines)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                        if task_val == "classification":
                            metric_summary = classification_metric_summary(y_val, y_pred, y_prob, n_classes=num_classes)
                            score = float(metric_summary["accuracy"]) * 100.0
                        else:
                            metric_summary = regression_metric_summary(y_val, y_pred)
                            score = float(metric_summary["mae"])
                        fold_scores.append(round(score, 4))
                        fold_metrics.append(metric_summary)
                    from training.cross_val import _aggregate_metric_dicts, format_cv_results
                    result = format_cv_results(
                        fold_scores,
                        round(float(np.mean(fold_scores)), 4),
                        round(float(np.std(fold_scores)), 4),
                        fold_metrics=fold_metrics,
                        aggregate_metrics=_aggregate_metric_dicts(fold_metrics),
                        task=task_val,
                    )
                    if split_warning:
                        result = f"> ⚠️ {split_warning}\n\n" + result
                    lines.append(result)
                    yield "\n".join(lines)
                else:
                    label_dtype = torch.int64 if task_val == "classification" else torch.float32
                    full_dataset = TensorDataset(
                        torch.tensor(windows, dtype=torch.float32),
                        torch.tensor(labels, dtype=label_dtype),
                    )
                    for prog in cross_validate(
                        full_dataset,
                        model_factory=lambda: _model_factory(num_classes, input_size),
                        k=int(cv_k_val),
                        epochs=int(epochs_val), lr=float(lr_val),
                        optimizer_name=optimizer_val, task=task_val,
                        batch_size=int(batch_size_val),
                    ):
                        if prog.get("done"):
                            result = format_cv_results(
                                prog["fold_accs"],
                                prog["mean_acc"],
                                prog["std_acc"],
                                fold_metrics=prog.get("fold_metrics"),
                                aggregate_metrics=prog.get("aggregate_metrics"),
                                task=task_val,
                            )
                            if prog.get("warning"):
                                result = f"> ⚠️ {prog['warning']}\n\n" + result
                            lines.append(result)
                        else:
                            fold_line = (
                                f"Fold {prog['fold']}/{prog['total_folds']}  "
                                f"epoch {prog['epoch']}  val_metric={prog.get('val_acc', 0):.2f}"
                                + ("%" if task_val == "classification" else "")
                            )
                            lines.append(fold_line)
                        yield "\n".join(lines)
            else:
                yield "ℹ️ Cross-validation is currently supported for **tabular** modality only."
                return

        except Exception as exc:
            yield f"❌ Cross-validation failed: {exc}"

    cv_run_btn.click(
        run_cv,
        inputs=[
            modality, data_path, text_col, label_col, time_col,
            window_size, n_frames, sample_rate, augmentation,
            val_split, use_random_subset, subset_percent, subset_seed, training_mode, model_name, task,
            epochs, lr, batch_size, dropout, optimizer, scheduler_name, use_amp,
            hidden_size, num_layers, cv_k,
        ],
        outputs=cv_results_box,
    )

    def run_cv_significance(modality_val, data_path_val, label_col_val, time_col_val, window_size_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val,
                            base_model_name, task_val, batch_size_val, dropout_val,
                            optimizer_val, hidden_size_val, num_layers_val,
                            epochs_val, lr_val, n_estimators_val, max_depth_val,
                            c_param_val, max_iter_val, lr_xgb_val, cv_k_val,
                            metric_name, compare_models):
        if modality_val not in {"tabular", "timeseries"}:
            return "### ⚠️ Significance testing is currently available for tabular and time-series models only.", None
        if task_val not in {"classification", "regression"}:
            return "### ⚠️ Significance testing currently supports classification or regression tasks.", None
        if int(cv_k_val) != 10:
            return "### ⚠️ Set cross-validation to **10 folds** before running the significance test.", None
        selected_models = [m for m in (compare_models or []) if m and m != base_model_name]
        if not selected_models:
            return "### ⚠️ Choose at least one additional model family to compare against the active model.", None

        import numpy as np
        import torch
        from scipy.stats import ttest_rel, wilcoxon
        from sklearn.model_selection import KFold, StratifiedKFold
        from torch.utils.data import DataLoader, TensorDataset

        from eval.metrics import classification_metric_summary, regression_metric_summary
        from models.registry import is_sklearn
        from models.tabular_models import get_tabular_model
        from modalities.tabular import load_tabular_data
        from modalities.timeseries import prepare_timeseries_windows
        from training.cross_val import _collect_fold_predictions, _train_one_fold

        if modality_val == "tabular":
            _tl, _vl, classes, _prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
                data_path_val,
                label_col=label_col_val,
                batch_size=int(batch_size_val),
                val_split=0.01,
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val),
            )
            X_all = np.concatenate([Xtr, Xv])
            y_all = np.concatenate([ytr, yv])
            ts_input_size = None
            windows_all = None
        else:
            windows_all, y_all, classes, prep_ts, ts_input_size = prepare_timeseries_windows(
                data_path=data_path_val,
                label_col=label_col_val,
                time_col=time_col_val if time_col_val else None,
                window_size=int(window_size_val),
                task=task_val,
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val),
            )
            X_all = windows_all.reshape(len(windows_all), -1).astype(np.float32)
            input_size = int(prep_ts.get("sklearn_input_size", X_all.shape[1]))

        split_warning = None
        if task_val == "classification":
            _, counts = np.unique(y_all, return_counts=True)
            min_class_size = int(counts.min()) if len(counts) else 0
            if min_class_size >= 10:
                splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                splits = list(splitter.split(X_all, y_all))
            else:
                splitter = KFold(n_splits=10, shuffle=True, random_state=42)
                splits = list(splitter.split(X_all))
                split_warning = (
                    f"Used plain K-Fold because the smallest class has only {min_class_size} samples."
                )
            higher_is_better = metric_name not in {"mae", "rmse"}
        else:
            splitter = KFold(n_splits=10, shuffle=True, random_state=42)
            splits = list(splitter.split(X_all))
            higher_is_better = metric_name == "r2"

        def build_family(model_name_local):
            if modality_val == "timeseries" and not is_sklearn(model_name_local):
                from models.timeseries_models import get_timeseries_model

                return get_timeseries_model(
                    model_name_local,
                    num_classes=(len(classes) if task_val == "classification" else 1),
                    input_size=ts_input_size,
                    hidden_size=int(hidden_size_val),
                    num_layers=int(num_layers_val),
                    dropout=float(dropout_val),
                )
            return get_tabular_model(
                model_name_local,
                num_classes=len(classes) if task_val == "classification" else 1,
                input_size=input_size,
                task=task_val,
                dropout=float(dropout_val),
                n_estimators=int(n_estimators_val),
                max_depth=int(max_depth_val),
                C=float(c_param_val),
                max_iter=int(max_iter_val),
                learning_rate=float(lr_xgb_val),
            )

        def score_family(model_name_local):
            fold_scores = []
            for train_idx, val_idx in splits:
                if is_sklearn(model_name_local):
                    X_train = X_all[train_idx]
                    y_train = y_all[train_idx]
                    X_val = X_all[val_idx]
                    y_val = y_all[val_idx]
                    model = build_family(model_name_local)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                else:
                    model = build_family(model_name_local)
                    if modality_val == "timeseries":
                        label_dtype = torch.int64 if task_val == "classification" else torch.float32
                        train_loader = DataLoader(
                            TensorDataset(
                                torch.tensor(windows_all[train_idx], dtype=torch.float32),
                                torch.tensor(y_all[train_idx], dtype=label_dtype),
                            ),
                            batch_size=int(batch_size_val),
                            shuffle=True,
                        )
                        val_loader = DataLoader(
                            TensorDataset(
                                torch.tensor(windows_all[val_idx], dtype=torch.float32),
                                torch.tensor(y_all[val_idx], dtype=label_dtype),
                            ),
                            batch_size=int(batch_size_val),
                            shuffle=False,
                        )
                    else:
                        X_train = X_all[train_idx]
                        y_train = y_all[train_idx]
                        X_val = X_all[val_idx]
                        y_val = y_all[val_idx]
                        label_dtype = torch.int64 if task_val == "classification" else torch.float32
                        train_loader = DataLoader(
                            TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=label_dtype)),
                            batch_size=int(batch_size_val),
                            shuffle=True,
                        )
                        val_loader = DataLoader(
                            TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=label_dtype)),
                            batch_size=int(batch_size_val),
                            shuffle=False,
                        )
                    for _ in _train_one_fold(
                        model,
                        train_loader,
                        val_loader,
                        epochs=int(epochs_val),
                        lr=float(lr_val),
                        optimizer_name=optimizer_val,
                        task=task_val,
                    ):
                        pass
                    y_true, y_pred, y_prob = _collect_fold_predictions(model, val_loader, task_val)
                    y_val = y_true

                if task_val == "classification":
                    metric_summary = classification_metric_summary(y_val, y_pred, y_prob, n_classes=len(classes))
                else:
                    metric_summary = regression_metric_summary(y_val, y_pred)
                score = metric_summary.get(metric_name)
                if score is None:
                    raise ValueError(f"Metric '{metric_name}' is unavailable for model '{model_name_local}'.")
                fold_scores.append(float(score))
            return np.asarray(fold_scores, dtype=float)

        baseline_scores = score_family(base_model_name)
        challenger_curves = {}
        lines = [
            "### Significance testing",
            "",
            f"- **Baseline family**: `{base_model_name}`",
            f"- **Metric**: `{metric_name}`",
            f"- **Folds**: `10` paired folds",
            f"- **Split strategy**: `{'StratifiedKFold' if split_warning is None and task_val == 'classification' else 'KFold'}`",
            "",
            "| Challenger | Baseline mean | Challenger mean | Delta | Paired t-test p | Wilcoxon p | Result |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        if split_warning:
            lines.insert(5, f"- **Note**: {split_warning}")

        for challenger in selected_models:
            challenger_scores = score_family(challenger)
            challenger_curves[challenger] = challenger_scores
            if higher_is_better:
                delta = challenger_scores.mean() - baseline_scores.mean()
            else:
                delta = baseline_scores.mean() - challenger_scores.mean()
            ttest_p = float(ttest_rel(challenger_scores, baseline_scores).pvalue)
            try:
                wilcoxon_p = float(wilcoxon(challenger_scores, baseline_scores).pvalue)
            except ValueError:
                wilcoxon_p = float("nan")
            result = "Different" if (ttest_p < 0.05 or (not np.isnan(wilcoxon_p) and wilcoxon_p < 0.05)) else "No clear difference"
            lines.append(
                f"| {challenger} | {baseline_scores.mean():.4f} | {challenger_scores.mean():.4f} | {delta:+.4f} | {ttest_p:.4f} | "
                f"{('n/a' if np.isnan(wilcoxon_p) else f'{wilcoxon_p:.4f}')} | {result} |"
            )

        lines.extend([
            "",
            "Paired tests use the same 10 folds for each family. Positive delta means the challenger performed better on the selected metric.",
        ])
        plot = _plot_cv_significance_scores(metric_name, base_model_name, baseline_scores, challenger_curves)
        return "\n".join(lines), plot

    cv_significance_btn.click(
        run_cv_significance,
        inputs=[
            modality, data_path, label_col, time_col, window_size, use_random_subset, subset_percent, subset_seed, training_mode,
            model_name, task, batch_size, dropout,
            optimizer, hidden_size, num_layers,
            epochs, lr, n_estimators, max_depth,
            C_param, max_iter, lr_xgb, cv_k,
            cv_significance_metric, cv_significance_models,
        ],
        outputs=[cv_significance_box, cv_significance_plot],
    )

    # ── Hyperparameter search callback ────────────────────────────────────────

    def run_hparam_search(modality_val, data_path_val, text_col_val, label_col_val,
                          time_col_val, window_size_val, n_frames_val, sample_rate_val,
                          augmentation_val, val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val,
                          model_name_val, task_val, batch_size_val, dropout_val,
                          optimizer_val, hidden_size_val, num_layers_val,
                          lr_vals_str, dropout_vals_str, n_trials_val, hparam_epochs_val):
        if not data_path_val or not model_name_val:
            yield "❌ Set Data path and Model before running hyperparameter search."
            return

        # Parse candidate values
        try:
            lr_candidates      = [float(v.strip()) for v in lr_vals_str.split(",") if v.strip()]
            dropout_candidates = [float(v.strip()) for v in dropout_vals_str.split(",") if v.strip()]
        except ValueError as e:
            yield f"❌ Could not parse candidates: {e}"
            return

        param_space = {"lr": lr_candidates, "dropout": dropout_candidates}
        yield f"Starting random hyperparameter search over {len(lr_candidates)}×{len(dropout_candidates)} grid…\n"
        lines = []

        try:
            if modality_val != "tabular":
                yield "ℹ️ Hyperparameter search is currently supported for **tabular** modality only."
                return

            import pandas as pd, numpy as np, torch
            from torch.utils.data import TensorDataset, DataLoader
            from modalities.tabular import load_tabular_data
            from training.hparam_search import random_search, format_search_results

            tl, vl, classes, prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
                data_path_val, label_col=label_col_val, batch_size=int(batch_size_val),
                val_split=float(val_split_val),
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val))
            num_classes = len(classes)

            def _factory(params):
                from models.tabular_models import get_tabular_model
                return get_tabular_model(
                    model_name_val, num_classes=num_classes,
                    input_size=input_size,
                    dropout=float(params.get("dropout", dropout_val)))

            all_results = []
            for prog in random_search(
                _factory, tl, vl,
                param_space=param_space,
                n_trials=int(n_trials_val),
                epochs_per_trial=int(hparam_epochs_val),
                task=task_val,
            ):
                if prog.get("done"):
                    lines.append("\n" + format_search_results(prog["all_results"]))
                    lines.append(f"\n✅ Best params: {prog['best_params']}  "
                                 f"val_acc={prog['best_val_acc']:.2f}%")
                else:
                    lines.append(f"Trial {prog['trial']}/{prog['total_trials']}  "
                                 f"params={prog['params']}  val_acc={prog['val_acc']:.2f}%")
                yield "\n".join(lines)

        except Exception as exc:
            yield f"❌ Hyperparameter search failed: {exc}"

    hparam_run_btn.click(
        run_hparam_search,
        inputs=[
            modality, data_path, text_col, label_col, time_col,
            window_size, n_frames, sample_rate, augmentation,
            val_split, use_random_subset, subset_percent, subset_seed, training_mode, model_name, task,
            batch_size, dropout, optimizer,
            hidden_size, num_layers,
            hparam_lr_vals, hparam_dropout_vals, hparam_n_trials, hparam_epochs,
        ],
        outputs=hparam_results_box,
    )

    def run_model_sweep(
        modality_val, data_path_val, text_col_val, label_col_val, feature_cols_val, time_col_val,
        window_size_val, n_frames_val, image_size_val, sample_rate_val, audio_image_size_val, audio_n_mels_val, augmentation_val,
        val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, use_data_cleaning_val, tabular_missing_strategy_val, tabular_clip_outliers_val, tabular_scaling_val,
        text_lowercase_val, text_strip_urls_val, text_strip_punctuation_val, text_remove_stopwords_val,
        text_deduplicate_val, text_apply_stemming_val, text_apply_lemmatization_val, text_use_ngrams_val,
        timeseries_sort_by_time_val, timeseries_fill_strategy_val,
        image_verify_files_val, image_aug_flip_val, image_aug_vertical_val, image_aug_rotation_val,
        image_aug_color_val, image_aug_gray_val, image_aug_perspective_val, image_normalization_val, image_force_grayscale_val,
        audio_verify_files_val, audio_normalize_waveform_val, audio_aug_noise_val, audio_aug_shift_val,
        audio_aug_gain_val, audio_aug_time_mask_val, audio_aug_freq_mask_val, video_verify_files_val,
        training_mode_val, task_val,
        epochs_val, lr_val, batch_size_val, dropout_val, optimizer_val, scheduler_val, use_amp_val,
        hidden_size_val, num_layers_val,
        n_estimators_val, max_depth_val, c_param_val, max_iter_val, lr_xgb_val,
        n_clusters_val, use_class_weights_val, checkpoint_every_val,
        bundle_name_val, sweep_models_val, sweep_metrics_val, sweep_sort_metric_val, sweep_sort_order_val,
    ):
        import pandas as pd
        from training.run_comparison import load_history

        selected_models = [m for m in (sweep_models_val or []) if m]
        selected_metrics = [m for m in (sweep_metrics_val or []) if m]
        if len(selected_models) < 2:
            yield (
                "### ⚠️ Model sweep\n\nChoose at least two models to compare.",
                pd.DataFrame(),
            )
            return
        if task_val not in {"classification", "regression"}:
            yield (
                "### ⚠️ Model sweep\n\nModel sweep currently supports classification and regression tasks.",
                pd.DataFrame(),
            )
            return
        if not selected_metrics:
            yield (
                "### ⚠️ Model sweep\n\nChoose at least one metric to show in the table.",
                pd.DataFrame(),
            )
            return

        status_lines = ["### Model sweep", "", f"Preparing {len(selected_models)} runs for `{modality_val}` / `{task_val}`."]
        rows = []

        def _sorted_frame(items):
            if not items:
                return pd.DataFrame(columns=["model", *selected_metrics, "status", "bundle"])
            df = pd.DataFrame(items)
            sort_metric = sweep_sort_metric_val if sweep_sort_metric_val in df.columns else None
            if sort_metric:
                df = df.sort_values(
                    by=sort_metric,
                    ascending=(str(sweep_sort_order_val) == "ascending"),
                    na_position="last",
                    kind="stable",
                )
            visible_cols = ["model", *selected_metrics, "status", "bundle"]
            return df[[c for c in visible_cols if c in df.columns]]

        for idx, model_choice in enumerate(selected_models, start=1):
            status_lines.append(f"- Running `{model_choice}` ({idx}/{len(selected_models)})…")
            yield "\n".join(status_lines), _sorted_frame(rows)

            local_bundle = f"{bundle_name_val}_{_slug_part(model_choice)}"
            try:
                updates = list(run_pipeline(
                    modality_val, data_path_val, text_col_val, label_col_val, feature_cols_val, time_col_val,
                    window_size_val, n_frames_val, image_size_val, sample_rate_val, audio_image_size_val, audio_n_mels_val, augmentation_val,
                    val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, use_data_cleaning_val, tabular_missing_strategy_val, tabular_clip_outliers_val, tabular_scaling_val,
                    text_lowercase_val, text_strip_urls_val, text_strip_punctuation_val, text_remove_stopwords_val,
                    text_deduplicate_val, text_apply_stemming_val, text_apply_lemmatization_val, text_use_ngrams_val,
                    timeseries_sort_by_time_val, timeseries_fill_strategy_val,
                    image_verify_files_val, image_aug_flip_val, image_aug_vertical_val, image_aug_rotation_val,
                    image_aug_color_val, image_aug_gray_val, image_aug_perspective_val, image_normalization_val, image_force_grayscale_val,
                    audio_verify_files_val, audio_normalize_waveform_val, audio_aug_noise_val, audio_aug_shift_val,
                    audio_aug_gain_val, audio_aug_time_mask_val, audio_aug_freq_mask_val, video_verify_files_val,
                    training_mode_val, model_choice, task_val,
                    epochs_val, lr_val, batch_size_val, dropout_val, optimizer_val, scheduler_val, use_amp_val,
                    hidden_size_val, num_layers_val,
                    n_estimators_val, max_depth_val, c_param_val, max_iter_val, lr_xgb_val,
                    n_clusters_val, use_class_weights_val, checkpoint_every_val,
                    local_bundle,
                ))
                last = updates[-1]
                bundle_path_val = last[11]
                run_record = next((r for r in reversed(load_history()) if r.get("bundle_path") == bundle_path_val), None)
                metric_payload = (run_record or {}).get("metrics", {})
                row = {
                    "model": model_choice,
                    "status": "complete",
                    "bundle": bundle_path_val,
                }
                for metric_name in selected_metrics:
                    row[metric_name] = _metric_display_value(metric_name, metric_payload.get(metric_name))
                rows.append(row)
                status_lines.append(f"  ✓ `{model_choice}` complete")
            except Exception as exc:
                row = {
                    "model": model_choice,
                    "status": f"failed: {exc}",
                    "bundle": "",
                }
                for metric_name in selected_metrics:
                    row[metric_name] = None
                rows.append(row)
                status_lines.append(f"  ⚠️ `{model_choice}` failed")

            yield "\n".join(status_lines), _sorted_frame(rows)

        status_lines.extend([
            "",
            f"Sorted by `{sweep_sort_metric_val}` in **{sweep_sort_order_val}** order.",
        ])
        yield "\n".join(status_lines), _sorted_frame(rows)

    sweep_run_btn.click(
        run_model_sweep,
        inputs=[
            modality, data_path, text_col, label_col, feature_cols, time_col,
            window_size, n_frames, image_size, sample_rate, audio_image_size, audio_n_mels, augmentation,
            val_split, use_random_subset, subset_percent, subset_seed, use_data_cleaning, tabular_missing_strategy, tabular_clip_outliers, tabular_scaling,
            text_lowercase, text_strip_urls, text_strip_punctuation, text_remove_stopwords,
            text_deduplicate, text_apply_stemming, text_apply_lemmatization, text_use_ngrams,
            timeseries_sort_by_time, timeseries_fill_strategy,
            image_verify_files, image_aug_flip, image_aug_vertical, image_aug_rotation,
            image_aug_color, image_aug_gray, image_aug_perspective, image_normalization, image_force_grayscale,
            audio_verify_files, audio_normalize_waveform, audio_aug_noise, audio_aug_shift,
            audio_aug_gain, audio_aug_time_mask, audio_aug_freq_mask, video_verify_files,
            training_mode, task,
            epochs, lr, batch_size, dropout, optimizer, scheduler_name, use_amp,
            hidden_size, num_layers,
            n_estimators, max_depth, C_param, max_iter, lr_xgb,
            n_clusters, use_class_weights, checkpoint_every,
            bundle_name, sweep_models, sweep_metrics, sweep_sort_metric, sweep_sort_order,
        ],
        outputs=[sweep_status_box, sweep_table],
    )

    def _plot_cv_significance_scores(metric_name, baseline_name, baseline_scores, challenger_scores_by_name):
        plt = _get_pyplot()
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
        folds = np.arange(1, len(baseline_scores) + 1)
        ax.plot(folds, baseline_scores, marker="o", linewidth=2.2, label=baseline_name, color="#115d52")
        palette = ["#d97706", "#2563eb", "#7c3aed", "#dc2626", "#0f766e"]
        for idx, (name, scores) in enumerate(challenger_scores_by_name.items()):
            ax.plot(
                folds,
                scores,
                marker="o",
                linewidth=1.8,
                linestyle="--",
                label=name,
                color=palette[idx % len(palette)],
                alpha=0.95,
            )
        ax.set_title(f"Fold-by-fold comparison for {metric_name}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_xticks(folds)
        ax.grid(alpha=0.22, linestyle=":")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig

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
                 f"Start server:\n"
                 f"  cd {bundle_path}\n"
                 f"  pip install -r requirements_serve.txt\n"
                 f"  uvicorn main:app --host 0.0.0.0 --port 8000"),
            )
        except Exception as e:
            return _format_status_block("FastAPI export", f"Error: {e}", success=False)

    fastapi_btn.click(gen_fastapi, inputs=[bundle_out, modality], outputs=fastapi_status_box)

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
                 f"Build & run:\n"
                 f"  cd {bundle_path}\n"
                 f"  docker build -t nocode-dl-{mod} .\n"
                 f"  docker run -p 8000:8000 nocode-dl-{mod}\n\n"
                 f"Or with Compose:\n"
                 f"  docker-compose up"),
            )
        except Exception as e:
            return _format_status_block("Docker export", f"Error: {e}", success=False)

    docker_btn.click(gen_docker, inputs=[bundle_out, modality], outputs=docker_status_box)

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

    model_card_btn.click(gen_model_card,
                         inputs=[bundle_out, modality, model_name,
                                 training_mode, task, eval_summary_box],
                         outputs=model_card_status_box)

    # ── Streamlit dashboard generator ─────────────────────────────────────────

    def gen_streamlit(bundle_path_val, mod, mname, tmode, tsk):
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
            )
            return (
                f"### Streamlit dashboard exported\n\n"
                f"```\n{out_path}\n```\n\n"
                f"Run it with:\n\n"
                f"```bash\npip install streamlit pandas matplotlib\n"
                f"streamlit run \"{out_path}\"\n```"
            )
        except Exception as e:
            return f"### Export failed\n\n```\n{e}\n```"

    streamlit_btn.click(
        gen_streamlit,
        inputs=[bundle_out, modality, model_name, training_mode, task],
        outputs=streamlit_status,
    )

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

    refresh_btn.click(refresh_history, outputs=history_df)
    compare_btn.click(compare_runs, inputs=compare_ids, outputs=[compare_status_box, compare_plot])
    # Only auto-load history on page load if the history file already exists —
    # avoids a redundant disk read (and silent failure) on a fresh install.
    from training.run_comparison import HISTORY_FILE as _HIST_FILE
    if _HIST_FILE.exists():
        demo.load(refresh_history, outputs=history_df)

    demo.load(
        lambda: gr.update(choices=_list_bundles()),
        outputs=infer_bundle_path,
    )

    # ── Object Detection callbacks ────────────────────────────────────────────

    def detect_image(img_path, model_key, conf, iou):
        if img_path is None:
            return None, "*No image uploaded.*", None
        try:
            from detection.yolo_detector import (
                run_image_detection, stats_to_markdown, make_summary_chart)
            annotated_bgr, stats, n_det = run_image_detection(
                img_path, model_key, conf=conf, iou=iou)
            annotated_rgb = annotated_bgr[:, :, ::-1]
            md    = stats_to_markdown(stats, total_frames=1, n_detections=n_det)
            chart = make_summary_chart(stats)
            return annotated_rgb, md, chart
        except Exception as e:
            return None, f"❌ Error: {e}", None

    det_img_btn.click(
        detect_image,
        inputs=[det_img_input, det_model, det_conf, det_iou],
        outputs=[det_img_output, det_img_stats_md, det_img_chart],
    )

    def detect_video(vid_path, model_key, conf, iou):
        if vid_path is None:
            yield "No video uploaded.", None, "*Upload a video first.*", None
            return
        try:
            from detection.yolo_detector import (
                run_video_detection, stats_to_markdown, make_summary_chart)
            out_dir = str(_OUTPUTS_ROOT / "detections")
            os.makedirs(out_dir, exist_ok=True)

            # L2: prune annotated output files older than 1 hour to prevent unbounded disk growth
            try:
                import time as _time
                _cutoff = _time.time() - 3600
                for _f in Path(out_dir).glob("*_detected.mp4"):
                    if _f.stat().st_mtime < _cutoff:
                        _f.unlink(missing_ok=True)
            except Exception as _e:
                logger.warning("Video cleanup failed: %s", _e)
            for prog in run_video_detection(
                vid_path, model_key, conf=conf, iou=iou, output_dir=out_dir
            ):
                if prog["done"]:
                    stats        = prog["stats"]
                    total_frames = prog["total_frames"]
                    n_det        = prog["n_detections"]
                    md    = stats_to_markdown(stats, total_frames=total_frames,
                                              n_detections=n_det)
                    chart = make_summary_chart(stats)
                    yield (f"✅ Done — {total_frames} frames, {n_det} detections.",
                           prog["output_path"], md, chart)
                else:
                    pct = prog["progress"] * 100
                    eta = prog["eta_seconds"]
                    eta_str = (f"{int(eta)}s" if eta < 60
                               else f"{int(eta/60)}m {int(eta%60)}s")
                    yield (f"Frame {prog['frame']}/{prog['total']}  "
                           f"({pct:.0f}%)  ·  {prog['fps_proc']} fps  ·  ETA {eta_str}",
                           None, "", None)
        except Exception as e:
            yield f"❌ Error: {e}", None, f"```\n{e}\n```", None

    det_vid_btn.click(
        detect_video,
        inputs=[det_vid_input, det_model, det_conf, det_iou],
        outputs=[det_vid_status, det_vid_output, det_vid_stats_md, det_vid_chart],
    )

    # ── YOLO custom training ──────────────────────────────────────────────────

    def train_yolo_custom(data_dir, model_size_name, n_epochs, batch):
        if not data_dir:
            yield "No dataset folder provided.", "*Enter a dataset path first.*"
            return
        try:
            from detection.yolo_trainer import (
                YOLODatasetPreparer, train_yolo_classifier, YOLO_TRAIN_MODELS)
            log_lines = []
            def log(m):
                log_lines.append(m)
                return "\n".join(log_lines)

            status = log("Preparing dataset…")
            yield status, ""

            # M2: sanitize the basename from user input before using in a path
            _safe_base = re.sub(r"[^a-zA-Z0-9_\-]", "_",
                                os.path.basename(data_dir.rstrip("/"))) or "yolo_dataset"
            prep_out = str(_OUTPUTS_ROOT / "yolo_data" / _safe_base)
            preparer = YOLODatasetPreparer(data_dir, prep_out, val_split=0.2)
            info     = preparer.prepare()
            status   = log(f"✓ {info['n_train']} train / {info['n_val']} val  "
                           f"· Classes: {', '.join(info['classes'])}")
            yield status, ""

            size_key = YOLO_TRAIN_MODELS.get(model_size_name, "n")
            for prog in train_yolo_classifier(
                data_dir=prep_out, model_size=size_key,
                epochs=int(n_epochs), batch=int(batch),
            ):
                if prog["done"]:
                    best_acc = prog.get("best_val_acc", 0)
                    status   = log(f"\n✅ Training complete!  Best val acc: {best_acc:.1f}%")
                    result_md = (
                        f"### YOLO Training Complete\n"
                        f"**Best validation accuracy:** {best_acc:.1f}%\n\n"
                        f"**Model saved to:** `{prog.get('model_path', 'N/A')}`"
                    )
                    yield status, result_md
                else:
                    e   = prog.get("epoch", "?")
                    tot = prog.get("epochs", "?")
                    acc = prog.get("val_acc", 0)
                    status = log(f"Epoch {e}/{tot} — val_acc={acc:.1f}%")
                    yield status, ""
        except Exception as ex:
            yield f"❌ Error: {ex}", ""

    yolo_train_btn.click(
        train_yolo_custom,
        inputs=[yolo_data_path, yolo_model_size, yolo_epochs, yolo_batch],
        outputs=[yolo_status_box, yolo_result_md],
    )

    # ── Try Your Model — inference callbacks ──────────────────────────────────

    def _predict(bundle_path, predict_fn_name, inp):
        if not bundle_path:
            return "*Set the bundle path above first.*", None
        try:
            from ui.inference_helpers import (
                predict_image, predict_text, predict_tabular, predict_audio, predict_timeseries,
                predictions_to_markdown, predictions_to_chart)
            fn = {"image": predict_image, "text": predict_text,
                  "tabular": predict_tabular, "audio": predict_audio, "timeseries": predict_timeseries}[predict_fn_name]
            if predict_fn_name == "tabular":
                if not inp or not str(inp).strip():
                    return (
                        "### ⚠️ Missing tabular input\n\nPaste a JSON object with feature values before running prediction.",
                        None,
                    )
                import json
                try:
                    inp = json.loads(inp) if isinstance(inp, str) else inp
                except json.JSONDecodeError as exc:
                    return (
                        f"### ⚠️ Invalid JSON\n\n```text\n{exc}\n```\n\n"
                        "Use a JSON object like "
                        "`{\"study_hours\": 12, \"attendance_pct\": 96}`.",
                        None,
                    )
            elif predict_fn_name == "timeseries":
                if not inp or not str(inp).strip():
                    return (
                        "### ⚠️ Missing time-series input\n\nPaste a JSON array of sequential rows before running prediction.",
                        None,
                    )
                import json
                try:
                    inp = json.loads(inp) if isinstance(inp, str) else inp
                except json.JSONDecodeError as exc:
                    return (
                        f"### ⚠️ Invalid JSON\n\n```text\n{exc}\n```\n\n"
                        "Use a JSON array like "
                        "`[{\"sensor_a\": 0.12, \"sensor_b\": 1.4}, {\"sensor_a\": 0.18, \"sensor_b\": 1.3}]`.",
                        None,
                    )
            elif predict_fn_name == "text" and not str(inp or "").strip():
                return "### ⚠️ Missing text input\n\nEnter some text before running prediction.", None
            elif predict_fn_name in {"image", "audio"} and inp is None:
                return f"### ⚠️ Missing {predict_fn_name} input\n\nUpload a file before running prediction.", None
            preds = fn(bundle_path, inp)
            return predictions_to_markdown(preds), predictions_to_chart(preds)
        except Exception as e:
            return f"### ⚠️ Prediction error\n\n```text\n{e}\n```", None

    infer_img_btn.click(
        lambda bp, img: _predict(bp, "image", img),
        inputs=[infer_bundle_path, infer_img_input],
        outputs=[infer_img_md, infer_img_chart],
    )
    def _predict_text_with_explanation(bp, txt):
        md, chart = _predict(bp, "text", txt)
        explain_html = ""
        if bp and txt and txt.strip():
            try:
                from ui.inference_helpers import explain_text_prediction
                explain_html = explain_text_prediction(bp, txt)
            except Exception:
                pass
        return md, chart, explain_html

    infer_text_btn.click(
        _predict_text_with_explanation,
        inputs=[infer_bundle_path, infer_text_input],
        outputs=[infer_text_md, infer_text_chart, infer_text_explain],
    )
    infer_tab_btn.click(
        lambda bp, fv: _predict(bp, "tabular", fv),
        inputs=[infer_bundle_path, infer_tab_input],
        outputs=[infer_tab_md, infer_tab_chart],
    )
    infer_tab_fill_btn.click(
        _suggest_tabular_example,
        inputs=[data_path, modality, label_col, infer_tab_input],
        outputs=infer_tab_input,
    )
    infer_ts_fill_btn.click(
        _suggest_timeseries_example,
        inputs=[data_path, modality, label_col, time_col, infer_ts_input],
        outputs=infer_ts_input,
    )
    infer_aud_btn.click(
        lambda bp, aud: _predict(bp, "audio", aud),
        inputs=[infer_bundle_path, infer_aud_input],
        outputs=[infer_aud_md, infer_aud_chart],
    )
    infer_ts_btn.click(
        lambda bp, ts: _predict(bp, "timeseries", ts),
        inputs=[infer_bundle_path, infer_ts_input],
        outputs=[infer_ts_md, infer_ts_chart],
    )

    # ── Batch prediction ──────────────────────────────────────────────────────

    def _batch_predict(bundle_path: str, folder_path: str):
        if not bundle_path:
            return None, "⚠️ Select a bundle first."
        folder = Path(folder_path.strip()) if folder_path else None
        if not folder or not folder.is_dir():
            return None, f"⚠️ Folder not found: `{folder_path}`"

        try:
            from ui.inference_helpers import (
                _cached_bundle, predict_image, predict_audio, predict_text,
            )
            bundle = _cached_bundle(bundle_path)
            mod = bundle.get("preprocessing", {}).get("modality", "image")
        except Exception as exc:
            return None, f"⚠️ Could not load bundle: {exc}"

        _EXT_MAP = {
            "image":     {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"},
            "audio":     {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
            "text":      {".txt"},
            "video":     {".mp4", ".avi", ".mov", ".mkv", ".webm"},
        }
        _FN_MAP = {
            "image": predict_image,
            "audio": predict_audio,
            "text":  predict_text,
        }
        fn = _FN_MAP.get(mod)
        if fn is None:
            return None, f"⚠️ Batch prediction is not yet supported for **{mod}** modality."

        exts = _EXT_MAP.get(mod, set())
        files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        if not files:
            return None, f"⚠️ No {mod} files found in `{folder_path}`."

        rows = []
        for fpath in files:
            try:
                inp = fpath.read_text(encoding="utf-8", errors="replace") if mod == "text" else str(fpath)
                preds = fn(bundle_path, inp)
                top = preds[0] if preds else {}
                if top.get("error"):
                    rows.append([fpath.name, "ERROR", str(top["error"])])
                else:
                    rows.append([fpath.name, top.get("label", "?"), f"{top.get('confidence', 0)*100:.1f}%"])
            except Exception as exc:
                rows.append([fpath.name, "ERROR", str(exc)])

        import pandas as pd
        df = pd.DataFrame(rows, columns=["file", "prediction", "confidence"])
        status = f"✅ Scored **{len(rows)}** files."
        return df, status

    def _batch_save_csv(df):
        if df is None or (hasattr(df, "empty") and df.empty):
            return "⚠️ No results to save."
        import pandas as pd
        out_dir = _OUTPUTS_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"batch_predictions_{ts}.csv"
        pd.DataFrame(df).to_csv(path, index=False)
        return f"✅ Saved to `{path}`"

    batch_run_btn.click(
        _batch_predict,
        inputs=[infer_bundle_path, batch_folder],
        outputs=[batch_df, batch_status],
    )
    batch_save_btn.click(
        _batch_save_csv,
        inputs=[batch_df],
        outputs=[batch_save_status],
    )


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
        show_api=show_api,
        _frontend=frontend,
    )


if __name__ == "__main__":
    launch_app()
