"""Look of the Studio: Gradio theme, stylesheet path and the dark-mode shim."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The component tree is built in ui/layout.py; the callbacks below wire it up.
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

# The app stylesheet; launch_app() passes it to demo.launch(css=...).
_APP_CSS_PATH = _PROJECT_ROOT / "styles" / "app.css"

# Gradio 4 marked dark mode with a `dark` class on <html>; Gradio 6 puts it on
# <body>. styles/app.css defines its dark tokens and ~20 overrides under `html.dark`,
# so mirror the body class onto <html> (and keep it in sync if it changes).
_SYNC_DARK_CLASS_JS = """
() => {
  const sync = () => document.documentElement.classList.toggle(
    "dark", document.body.classList.contains("dark"));
  sync();
  new MutationObserver(sync).observe(document.body, {attributes: true, attributeFilter: ["class"]});
}
"""
