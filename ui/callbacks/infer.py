"""Try Your Model tab: single and batch predictions."""
from __future__ import annotations

from pathlib import Path

from ui.helpers import _OUTPUTS_ROOT, _suggest_tabular_example, _suggest_timeseries_example
from ui.layout import Components


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


def register(c: Components) -> None:
    """Wire this area's events to the components in ``c``."""
    c.infer_img_btn.click(
        lambda bp, img: _predict(bp, "image", img),
        inputs=[c.infer_bundle_path, c.infer_img_input],
        outputs=[c.infer_img_md, c.infer_img_chart],
    )

    c.infer_text_btn.click(
        _predict_text_with_explanation,
        inputs=[c.infer_bundle_path, c.infer_text_input],
        outputs=[c.infer_text_md, c.infer_text_chart, c.infer_text_explain],
    )
    c.infer_tab_btn.click(
        lambda bp, fv: _predict(bp, "tabular", fv),
        inputs=[c.infer_bundle_path, c.infer_tab_input],
        outputs=[c.infer_tab_md, c.infer_tab_chart],
    )
    c.infer_tab_fill_btn.click(
        _suggest_tabular_example,
        inputs=[c.data_path, c.modality, c.label_col, c.infer_tab_input],
        outputs=c.infer_tab_input,
    )
    c.infer_ts_fill_btn.click(
        _suggest_timeseries_example,
        inputs=[c.data_path, c.modality, c.label_col, c.time_col, c.infer_ts_input],
        outputs=c.infer_ts_input,
    )
    c.infer_aud_btn.click(
        lambda bp, aud: _predict(bp, "audio", aud),
        inputs=[c.infer_bundle_path, c.infer_aud_input],
        outputs=[c.infer_aud_md, c.infer_aud_chart],
    )
    c.infer_ts_btn.click(
        lambda bp, ts: _predict(bp, "timeseries", ts),
        inputs=[c.infer_bundle_path, c.infer_ts_input],
        outputs=[c.infer_ts_md, c.infer_ts_chart],
    )

    c.batch_run_btn.click(
        _batch_predict,
        inputs=[c.infer_bundle_path, c.batch_folder],
        outputs=[c.batch_df, c.batch_status],
    )
    c.batch_save_btn.click(
        _batch_save_csv,
        inputs=[c.batch_df],
        outputs=[c.batch_save_status],
    )
