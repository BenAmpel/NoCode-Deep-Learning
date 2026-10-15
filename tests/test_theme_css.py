"""Static checks on styles/app.css dark mode.

Two bugs these guard against:
- ``@media (prefers-color-scheme: dark)`` blocks made the page half dark when
  the OS was dark but Gradio's theme was light (``?__theme=light``): the hero
  went dark while its headline stayed dark green.
- Gradio injects css= a second time with every selector prefixed by
  ``.gradio-container… .contain``. Those copies outrank ``html.dark X``, so a
  dark override needs a ``.studio-dark X`` twin, which still matches inside
  ``.contain`` (ui/theme.py sets ``studio-dark`` on ``.studio-shell``).
"""
import re

from ui.theme import _APP_CSS_PATH, _SYNC_DARK_CLASS_JS

CSS = _APP_CSS_PATH.read_text(encoding="utf-8")
PAGE_LEVEL = ("body", "gradio-app", ".gradio-container")


def _selector_lists(css: str):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for match in re.finditer(r"([^{}]+)\{", css):
        text = match.group(1).strip()
        if text and not text.startswith("@"):
            yield [s.strip() for s in text.split(",")]


def test_dark_mode_follows_gradio_theme_not_the_os():
    assert "prefers-color-scheme: dark" not in CSS


def test_every_in_shell_dark_rule_also_matches_inside_gradio_scope():
    missing = []
    for sels in _selector_lists(CSS):
        for sel in sels:
            if not sel.startswith("html.dark "):
                continue
            rest = sel[len("html.dark "):]
            if rest.split()[0].startswith(PAGE_LEVEL):
                continue
            twin = (".studio-shell.studio-dark" + rest[len(".studio-shell"):]
                    if rest.startswith(".studio-shell") else ".studio-dark " + rest)
            if twin not in sels:
                missing.append(sel)
    assert not missing, missing[:5]


def test_theme_script_marks_html_and_the_shell():
    assert 'classList.toggle("dark", dark)' in _SYNC_DARK_CLASS_JS
    assert '".studio-shell"' in _SYNC_DARK_CLASS_JS and '"studio-dark"' in _SYNC_DARK_CLASS_JS


def test_css_braces_balance():
    assert CSS.count("{") == CSS.count("}")
