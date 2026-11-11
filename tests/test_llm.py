"""Language Model tab: client, zero-shot labeling, RAG. No network: every
request goes through a fake transport."""
import json
import logging
import urllib.error
from io import BytesIO

import pandas as pd
import pytest

from llm import rag, zero_shot
from llm.client import LLMConfig, LLMError, chat

SECRET = "sk-test-SECRET-1234567890"


def _reply(text):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()


class FakeServer:
    """Records requests; answers with a function of the user message."""

    def __init__(self, answer=lambda msgs: "positive", fail=None):
        self.answer, self.fail, self.calls = answer, list(fail or []), []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, json.loads(body), dict(headers)))
        if self.fail:
            exc = self.fail.pop(0)
            if exc is not None:
                raise exc
        return _reply(self.answer(json.loads(body)["messages"]))


def _http(code, body=b"err"):
    return urllib.error.HTTPError("http://x", code, "msg", {}, BytesIO(body))


# ── client ────────────────────────────────────────────────────────────────

def test_request_format_and_auth_header():
    srv = FakeServer(lambda m: "ready")
    cfg = LLMConfig(base_url=" http://localhost:11434/v1/ ", model="m", api_key=SECRET)
    assert chat(cfg, [{"role": "user", "content": "hi"}], transport=srv) == "ready"
    url, body, headers = srv.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert body["model"] == "m" and body["temperature"] == 0 and body["stream"] is False
    assert headers["Authorization"] == f"Bearer {SECRET}"
    # No key: no header. Full endpoint URL is not doubled.
    srv2 = FakeServer(lambda m: "ok")
    chat(LLMConfig(base_url="https://h/v1/chat/completions", model="m"), [], transport=srv2)
    assert srv2.calls[0][0] == "https://h/v1/chat/completions"
    assert "Authorization" not in srv2.calls[0][2]


def test_key_never_appears_in_repr_or_errors():
    cfg = LLMConfig(model="m", api_key=SECRET)
    assert SECRET not in repr(cfg)
    leaky = urllib.error.URLError(f"refused for key {SECRET}")
    with pytest.raises(LLMError) as ei:
        chat(cfg, [], transport=FakeServer(fail=[leaky] * 3), sleep=lambda s: None)
    assert SECRET not in str(ei.value) and "Cannot reach" in str(ei.value)
    with pytest.raises(LLMError) as ei:
        chat(cfg, [], transport=FakeServer(fail=[_http(400, SECRET.encode())]), sleep=lambda s: None)
    assert SECRET not in str(ei.value)
    with pytest.raises(LLMError, match="refused the API key"):
        chat(cfg, [], transport=FakeServer(fail=[_http(401)]))


@pytest.mark.parametrize("url,model,match", [
    ("", "m", "server URL"), ("localhost:11434", "m", "http"), ("ftp://x", "m", "http"),
    ("http://x", "  ", "model name"),
])
def test_config_validation(url, model, match):
    with pytest.raises(LLMError, match=match):
        chat(LLMConfig(base_url=url, model=model), [], transport=FakeServer())


def test_retries_transient_errors_then_succeeds():
    sleeps = []
    srv = FakeServer(lambda m: "fine", fail=[_http(429), _http(503), None])
    assert chat(LLMConfig(model="m"), [], transport=srv, sleep=sleeps.append) == "fine"
    assert len(srv.calls) == 3 and sleeps == [1, 2]


def test_no_retry_on_404_and_clear_message():
    srv = FakeServer(fail=[_http(404)])
    with pytest.raises(LLMError, match="404"):
        chat(LLMConfig(model="nope"), [], transport=srv, sleep=lambda s: None)
    assert len(srv.calls) == 1


@pytest.mark.parametrize("raw,match", [
    (b"<html>", "not JSON"), (b'{"error": {"message": "model not loaded"}}', "model not loaded"),
    (b'{"choices": []}', "no message"), (b"[]", "no message"),
])
def test_bad_replies_are_readable(raw, match):
    with pytest.raises(LLMError, match=match):
        chat(LLMConfig(model="m"), [], transport=lambda *a: raw)


def test_content_parts_and_null_content():
    parts = json.dumps({"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"text": "b"}]}}]})
    assert chat(LLMConfig(model="m"), [], transport=lambda *a: parts.encode()) == "ab"
    null = json.dumps({"choices": [{"message": {"content": None}}]})
    assert chat(LLMConfig(model="m"), [], transport=lambda *a: null.encode()) == ""


# ── zero-shot ─────────────────────────────────────────────────────────────

def test_parse_labels():
    assert zero_shot.parse_labels(" pos, Neg ,\n'neutral', pos, ,") == ["pos", "Neg", "neutral"]
    assert zero_shot.parse_labels("") == []


LABELS = ["positive", "negative", "very positive"]


@pytest.mark.parametrize("reply,expected", [
    ("positive", "positive"), ("Positive.", "positive"), ("**negative**", "negative"),
    ("Label: negative", "negative"), ('{"label": "negative"}', "negative"),
    ("negative\nBecause the customer is angry.", "negative"),
    ("I think this is very positive overall", "very positive"),
    ("<think>maybe negative?</think>positive", "positive"),
    ("positive or negative", None), ("non-negative", None), ("", None), ("unsure", None),
])
def test_parse_label(reply, expected):
    assert zero_shot.parse_label(reply, LABELS) == expected


def test_prompt_truncates_and_includes_definitions():
    msgs = zero_shot.build_messages("x" * 20_000, ["a", "b"], "a: good")
    assert "a: good" in msgs[0]["content"]
    assert len(msgs[1]["content"]) < zero_shot.MAX_CHARS + 200


def _folder(tmp_path):
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / "b.txt").write_text("I love it", encoding="utf-8")
    (d / "a.txt").write_text("I hate it", encoding="utf-8")
    (d / "sub" / "c.txt").write_bytes(b"caf\xe9 ok")           # not UTF-8
    (d / "empty.txt").write_text("   ")
    (d / ".hidden.txt").write_text("skip me")
    (d / "notes.md").write_text("not a txt")
    return d


def test_classify_folder_rows_and_errors(tmp_path):
    def answer(msgs):
        text = msgs[1]["content"]
        return "negative" if "hate" in text else ("positive" if "love" in text else "no idea")
    df = zero_shot.classify_folder(_folder(tmp_path), ["positive", "negative"], LLMConfig(model="m"),
                                   transport=FakeServer(answer))
    assert list(df.columns) == zero_shot.COLUMNS
    assert df["file"].tolist() == ["a.txt", "b.txt", "empty.txt", "c.txt"]
    got = dict(zip(df.file, df.prediction))
    assert got == {"a.txt": "negative", "b.txt": "positive", "empty.txt": "ERROR", "c.txt": "ERROR"}
    assert "Unclear reply" in df.set_index("file").loc["c.txt", "confidence"]


def test_classify_folder_stops_early_when_server_is_down(tmp_path):
    d = tmp_path / "many"
    d.mkdir()
    for i in range(10):
        (d / f"{i:02d}.txt").write_text("text")
    srv = FakeServer(fail=[urllib.error.URLError("refused")] * 100)
    df = zero_shot.classify_folder(d, ["a", "b"], LLMConfig(model="m", retries=0), transport=srv,
                                   sleep=lambda s: None)
    assert len(df) == 10 and (df.prediction == "ERROR").all()
    assert len(srv.calls) == 3
    assert df.confidence.iloc[-1].startswith("Skipped")


@pytest.mark.parametrize("folder,labels,match", [
    (None, ["a", "b"], "not found"), ("/no/such/dir", ["a", "b"], "not found"), ("ok", ["a"], "two labels"),
])
def test_classify_folder_input_errors(tmp_path, folder, labels, match):
    if folder == "ok":
        folder = _folder(tmp_path)
    with pytest.raises(LLMError, match=match):
        zero_shot.classify_folder(folder, labels, LLMConfig(model="m"), transport=FakeServer())
    empty = tmp_path / "nothing"
    empty.mkdir(exist_ok=True)
    with pytest.raises(LLMError, match="No .txt"):
        zero_shot.classify_folder(empty, ["a", "b"], LLMConfig(model="m"), transport=FakeServer())


# ── RAG ───────────────────────────────────────────────────────────────────

def test_chunking_edges():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = rag.chunk_text(words, 30, 10)
    assert chunks[0].split()[0] == "w0" and chunks[1].split()[0] == "w20"
    assert chunks[-1].split()[-1] == "w99"
    assert rag.chunk_text("", 30, 10) == []
    assert len(rag.chunk_text("one two", 30, 10)) == 1
    # Overlap >= size and absurd sizes are clamped, never an infinite loop.
    assert len(rag.chunk_text(words, 20, 500)) == 81
    assert len(rag.chunk_text(words, 0, -5)) == 5
    assert len(rag.chunk_text(words, None, None)) == 5
    assert len(rag.chunk_text(words, 10**9, 0)) == 1


def test_retrieval_ranks_relevant_chunk_first():
    chunks = ["The refund policy allows returns within 30 days.",
              "Shipping takes five business days.",
              "Our office is in Atlanta."]
    hits = rag.retrieve(chunks, "How many days do I have for a refund?", 2)
    assert hits[0].chunk_id == 1 and hits[0].rank == 1 and len(hits) == 2
    assert len(rag.retrieve(chunks, "refund", 99)) == 3
    assert len(rag.retrieve(chunks, "refund", 0)) == 1
    with pytest.raises(LLMError, match="question"):
        rag.retrieve(chunks, "  ", 2)
    # A stop-word-only document still retrieves (falls back to no stop list).
    assert rag.retrieve(["the and of", "it is"], "the", 1)[0].score >= 0


def test_ask_retrieval_only_and_with_model(tmp_path):
    doc = tmp_path / "policy.md"
    doc.write_text("Returns are accepted within 30 days. " * 5 + "Shipping is free over $50. " * 5)
    answer, hits, n = rag.ask(str(doc), "Is shipping free?", 20, 5, 2, None)
    assert answer is None and n >= 2 and "Shipping" in hits[0].text
    srv = FakeServer(lambda m: "Yes, over $50 [2].")
    answer, hits, _ = rag.ask(str(doc), "Is shipping free?", 20, 5, 2, LLMConfig(model="m"), transport=srv)
    assert answer == "Yes, over $50 [2]."
    assert "Shipping is free" in srv.calls[0][1]["messages"][1]["content"]


def test_read_document_errors(tmp_path):
    with pytest.raises(LLMError, match="Upload"):
        rag.read_document(None)
    with pytest.raises(LLMError, match="not found"):
        rag.read_document(tmp_path / "nope.txt")
    (tmp_path / "x.docx").write_text("x")
    with pytest.raises(LLMError, match="Unsupported"):
        rag.read_document(tmp_path / "x.docx")
    (tmp_path / "blank.txt").write_text("  \n ")
    with pytest.raises(LLMError, match="no readable text"):
        rag.read_document(tmp_path / "blank.txt")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    with pytest.raises(LLMError):
        rag.read_document(tmp_path / "bad.pdf")


# ── UI callbacks: the key never reaches disk or logs ──────────────────────

def test_callbacks_never_persist_or_log_the_key(tmp_path, monkeypatch, caplog):
    import ui.project_state as ps
    from ui.callbacks import llm as cb

    monkeypatch.setattr(ps, "STATE_FILE", tmp_path / "project_state.json")
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    def leaky(*a, **k):
        raise urllib.error.URLError(f"bad key {SECRET}")
    monkeypatch.setattr("llm.client._urllib_transport", leaky)
    monkeypatch.setattr("llm.client.time.sleep", lambda s: None)

    out = cb.on_test("http://127.0.0.1:9/v1", "m", SECRET)
    assert out.startswith("❌") and SECRET not in out
    df, status, state = cb.on_zero_shot("http://127.0.0.1:9/v1", "m", SECRET, str(_folder(tmp_path)),
                                        "positive, negative", "", progress=lambda *a, **k: None)
    assert SECRET not in status and SECRET not in df.to_csv()
    doc = tmp_path / "d.txt"
    doc.write_text("alpha beta gamma " * 50)
    md, table = cb.on_rag(str(doc), "beta?", 20, 5, 2, True, "http://127.0.0.1:9/v1", "m", SECRET)
    assert md.startswith("❌") and SECRET not in md
    md, table = cb.on_rag(str(doc), "beta?", 20, 5, 2, False, "", "", SECRET)
    assert "Retrieval only" in md and isinstance(table, pd.DataFrame)

    assert SECRET not in caplog.text
    for f in tmp_path.rglob("*"):
        if f.is_file() and f.suffix in {".json", ".log", ".csv", ".txt"}:
            assert SECRET not in f.read_text(errors="replace"), f


def test_zero_shot_save_uses_batch_filename(tmp_path, monkeypatch):
    from ui.callbacks import infer
    from ui.callbacks import llm as cb

    monkeypatch.setattr(infer, "_OUTPUTS_ROOT", tmp_path / "out")
    srv = FakeServer(lambda m: "positive")
    monkeypatch.setattr("llm.client._urllib_transport", srv)
    d = tmp_path / "t"
    d.mkdir()
    (d / "1.txt").write_text("good")
    df, status, state = cb.on_zero_shot("http://x/v1", "m", "", str(d), "positive, negative", "",
                                        progress=lambda *a, **k: None)
    assert status.startswith("✅") and "⏱" in status
    msg = infer._batch_save_csv(state)
    assert msg.startswith("✅")
    saved = next((tmp_path / "out").glob("batch_predictions_*.csv"))
    assert pd.read_csv(saved).columns.tolist() == zero_shot.COLUMNS


def test_retrieval_matches_word_forms():
    chunks = ["Returns are accepted within 30 days of purchase. Refunds go to the card.",
              "Shipping is free on orders over fifty dollars.",
              "Our store in Atlanta is open on weekdays."]
    hits = rag.retrieve(chunks, "How long do I have to return something?", 1)
    assert hits[0].chunk_id == 1 and hits[0].score > 0
    hits = rag.retrieve(chunks, "Is there a refund?", 1)
    assert hits[0].chunk_id == 1 and hits[0].score > 0
    hits = rag.retrieve(chunks, "shipped order", 1)
    assert hits[0].chunk_id == 2
    assert rag._stem("policies") == "policy" and rag._stem("class") == "class"
    assert rag._stem("is") == "is" and rag._stem("bus") == "bus"


def test_zero_shot_status_when_every_file_fails(tmp_path, monkeypatch):
    from ui.callbacks import llm as cb

    def down(*a, **k):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr("llm.client._urllib_transport", down)
    monkeypatch.setattr("llm.client.time.sleep", lambda s: None)
    d = tmp_path / "one"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    _, status, _ = cb.on_zero_shot("http://127.0.0.1:9/v1", "m", "", str(d), "a, b", "",
                                   progress=lambda *a, **k: None)
    assert status.startswith("❌") and "1 file " in status and "1 files" not in status
