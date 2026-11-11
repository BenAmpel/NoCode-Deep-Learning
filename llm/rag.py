"""Retrieval-augmented question answering over one document.

Chunks the document by words, ranks chunks with TF-IDF cosine similarity
(scikit-learn, no extra download), and asks the language model to answer
from the top chunks only. Without a model server it still shows the
retrieved chunks, which is the part students experiment with.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm.client import LLMConfig, LLMError, Transport, chat

TEXT_EXTS = {".txt", ".md", ".markdown", ".csv"}
MAX_DOC_CHARS = 5_000_000
MIN_CHUNK_WORDS = 20
MAX_CHUNK_WORDS = 2000


@dataclass
class Retrieved:
    rank: int
    score: float
    chunk_id: int
    text: str


def read_document(path: str | Path) -> str:
    if not path:
        raise LLMError("Upload a document first (.txt, .md, or .pdf).")
    p = Path(str(path))
    if not p.is_file():
        raise LLMError(f"Document not found: `{p}`")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise LLMError("Reading PDFs needs the `pypdf` package. Save the document as .txt instead.") from None
        try:
            text = "\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages)
        except Exception as exc:
            raise LLMError(f"Could not read the PDF: {exc}") from exc
    elif suffix in TEXT_EXTS:
        text = p.read_text(encoding="utf-8", errors="replace")
    else:
        raise LLMError(f"Unsupported document type `{suffix}`. Use .txt, .md, or .pdf.")
    if len(text) > MAX_DOC_CHARS:
        raise LLMError(f"The document is {len(text):,} characters; the limit is {MAX_DOC_CHARS:,}.")
    if not text.strip():
        raise LLMError("The document has no readable text (a scanned PDF needs OCR first).")
    return text


def chunk_text(text: str, chunk_words: int = 200, overlap_words: int = 40) -> list[str]:
    """Split into chunks of ``chunk_words`` words; consecutive chunks share
    ``overlap_words`` words so a sentence cut at a boundary appears whole once."""
    words = (text or "").split()
    if not words:
        return []
    size = int(min(max(int(chunk_words or 0), MIN_CHUNK_WORDS), MAX_CHUNK_WORDS))
    overlap = int(min(max(int(overlap_words or 0), 0), size - 1))
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(word: str) -> str:
    """Strip common English endings so "returns", "returned" and "return" match.

    Deliberately crude (no dictionary): retrieval only needs the question and
    the document to land on the same stem, not on a real word.
    """
    w = word.split("'")[0]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    for suffix, min_len in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if len(w) >= min_len and w.endswith(suffix) and not w.endswith("ss"):
            return w[: -len(suffix)]
    return w


def _tokens(text: str, drop_stop: bool) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = _TOKEN.findall(text.lower())
    if drop_stop:
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return [_stem(w) for w in words if len(w) > 1 or w.isdigit()]


def retrieve(chunks: list[str], question: str, top_k: int = 3) -> list[Retrieved]:
    if not chunks:
        raise LLMError("The document produced no chunks.")
    if not question or not question.strip():
        raise LLMError("Type a question.")
    from sklearn.feature_extraction.text import TfidfVectorizer

    k = int(min(max(int(top_k or 1), 1), len(chunks)))
    matrix = None
    for drop_stop in (True, False):         # all-stopword text has no other vocabulary
        try:
            vec = TfidfVectorizer(tokenizer=lambda t, d=drop_stop: _tokens(t, d), token_pattern=None,
                                  lowercase=False, ngram_range=(1, 2), sublinear_tf=True)
            matrix = vec.fit_transform(chunks)
            q = vec.transform([question])
            break
        except ValueError:
            continue
    if matrix is None:
        raise LLMError("The document has no words to search.")
    scores = (matrix @ q.T).toarray().ravel()
    order = np.argsort(-scores, kind="stable")[:k]
    return [Retrieved(rank=i + 1, score=float(scores[j]), chunk_id=int(j) + 1, text=chunks[j])
            for i, j in enumerate(order)]


def build_messages(question: str, hits: list[Retrieved]) -> list[dict]:
    context = "\n\n".join(f"[{h.chunk_id}] {h.text}" for h in hits)
    system = ("Answer the question using only the numbered excerpts. Cite the excerpt numbers you used, "
              "like [3]. If the excerpts do not contain the answer, say you cannot find it in the document.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question.strip()}"}]


def ask(document_path, question: str, chunk_words: int, overlap_words: int, top_k: int,
        cfg: LLMConfig | None, *, transport: Transport | None = None) -> tuple[str | None, list[Retrieved], int]:
    """Return ``(answer or None, retrieved chunks, number of chunks)``.

    ``answer`` is None when no model server is configured (retrieval only).
    """
    text = read_document(document_path)
    chunks = chunk_text(text, chunk_words, overlap_words)
    hits = retrieve(chunks, question, top_k)
    if cfg is None:
        return None, hits, len(chunks)
    answer = chat(cfg, build_messages(question, hits), transport=transport)
    return answer, hits, len(chunks)
