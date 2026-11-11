"""Language Model tab: connection test, zero-shot labeling, document Q&A.

The API key arrives as a plain callback argument and lives only in the
``LLMConfig`` built for one request. Nothing here writes it to project
state, run history, or logs, and every message shown is scrubbed of it.
"""
from __future__ import annotations

import logging
import time

import gradio as gr
import pandas as pd

from llm.client import LLMConfig, LLMError, test_connection
from llm.rag import ask
from llm.zero_shot import classify_folder, parse_labels
from ui.callbacks.infer import _batch_save_csv
from ui.layout import Components

logger = logging.getLogger("nocode_dl")


def _cfg(base_url, model, api_key) -> LLMConfig:
    return LLMConfig(base_url=base_url or "", model=model or "", api_key=api_key or "")


def on_test(base_url, model, api_key):
    cfg = _cfg(base_url, model, api_key)
    try:
        reply = test_connection(cfg)
        return f"✅ Connected to `{cfg.model}`. It replied: *{cfg.scrub(reply) or '(empty)'}*"
    except LLMError as exc:
        return f"❌ {cfg.scrub(exc)}"
    except Exception as exc:  # never surface a traceback that might carry the key
        logger.warning("LLM connection test failed: %s", type(exc).__name__)
        return f"❌ {cfg.scrub(exc)}"


def on_zero_shot(base_url, model, api_key, folder, labels_text, instructions, progress=gr.Progress()):
    cfg = _cfg(base_url, model, api_key)
    labels = parse_labels(labels_text)
    t0 = time.perf_counter()
    try:
        df = classify_folder(folder, labels, cfg, instructions or "",
                             progress=lambda i, n: progress((i, n), desc="Labeling files"))
    except LLMError as exc:
        return None, f"❌ {cfg.scrub(exc)}", None
    except Exception as exc:
        logger.warning("Zero-shot labeling failed: %s", type(exc).__name__)
        return None, f"❌ {cfg.scrub(exc)}", None
    elapsed = time.perf_counter() - t0
    n_err = int((df["prediction"] == "ERROR").sum())
    n_ok = len(df) - n_err
    counts = df.loc[df["prediction"] != "ERROR", "prediction"].value_counts()
    dist = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "none"
    def files(n):
        return f"{n:,} file" + ("" if n == 1 else "s")
    if n_ok == 0:
        status = (f"❌ No files were labeled ({files(n_err)} with `ERROR` rows). "
                  "The reason is in the confidence column.")
    else:
        status = f"✅ Labeled **{n_ok:,}** of {files(len(df))} ({dist})."
        if n_err:
            status += f" ⚠️ {files(n_err)} with `ERROR` rows; the reason is in the confidence column."
    status += f"\n\n⏱ {files(len(df))} in {elapsed:.1f} s ({1000 * elapsed / max(len(df), 1):.0f} ms each)."
    logger.info("Zero-shot labeled %d files in %.1f s", len(df), elapsed)
    return df, status, df


def on_rag(file_path, question, chunk, overlap, top_k, use_model, base_url, model, api_key):
    cfg = _cfg(base_url, model, api_key) if use_model else None
    try:
        answer, hits, n_chunks = ask(file_path, question or "", chunk, overlap, top_k, cfg)
    except LLMError as exc:
        msg = cfg.scrub(exc) if cfg else str(exc)
        return f"❌ {msg}", None
    except Exception as exc:
        logger.warning("Document Q&A failed: %s", type(exc).__name__)
        return f"❌ {cfg.scrub(exc) if cfg else exc}", None
    table = pd.DataFrame([{"rank": h.rank, "chunk": h.chunk_id, "similarity": round(h.score, 3),
                           "text": h.text} for h in hits])
    header = f"Retrieved {len(hits)} of {n_chunks} chunks."
    if answer is None:
        body = "*Retrieval only: the model was not called. Read the chunks below.*"
    else:
        body = answer or "*The model returned an empty answer.*"
    if hits and max(h.score for h in hits) == 0:
        header += " ⚠️ No chunk shares a word with the question, so these chunks are arbitrary."
    return f"{body}\n\n---\n{header}", table


def register(c: Components) -> None:
    server = [c.llm_base_url, c.llm_model, c.llm_api_key]
    c.llm_test_btn.click(on_test, inputs=server, outputs=[c.llm_test_status])
    c.llm_zs_run_btn.click(
        on_zero_shot,
        inputs=server + [c.llm_zs_folder, c.llm_zs_labels, c.llm_zs_instructions],
        outputs=[c.llm_zs_df, c.llm_zs_status, c.llm_zs_results],
    )
    c.llm_zs_save_btn.click(_batch_save_csv, inputs=[c.llm_zs_results], outputs=[c.llm_zs_save_status])
    c.llm_rag_btn.click(
        on_rag,
        inputs=[c.llm_rag_file, c.llm_rag_question, c.llm_rag_chunk, c.llm_rag_overlap,
                c.llm_rag_topk, c.llm_rag_use_model] + server,
        outputs=[c.llm_rag_answer, c.llm_rag_chunks],
    )


__all__ = ["register", "on_test", "on_zero_shot", "on_rag"]
