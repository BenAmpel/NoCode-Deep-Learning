"""A minimal client for OpenAI-compatible chat endpoints.

One request format covers a local Ollama or LM Studio server, OpenAI, and
Google's OpenAI-compatible Gemini endpoint, so the app needs no SDK.

The API key lives only in the ``LLMConfig`` object for the length of a
request. It is never written to disk or logs, and every error message is
scrubbed of it.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # Ollama
DEFAULT_MODEL = "llama3.2:3b"


class LLMError(RuntimeError):
    """A readable reason a language-model request failed."""


@dataclass
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = field(default="", repr=False)
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 512
    retries: int = 2

    def __post_init__(self):
        self.base_url = (self.base_url or "").strip()
        self.model = (self.model or "").strip()
        self.api_key = (self.api_key or "").strip()

    @property
    def endpoint(self) -> str:
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        return url

    def validate(self) -> None:
        if not self.base_url:
            raise LLMError("Enter the model server URL (for example http://localhost:11434/v1 for Ollama).")
        if not self.base_url.lower().startswith(("http://", "https://")):
            raise LLMError("The server URL must start with http:// or https://.")
        if not self.model:
            raise LLMError("Enter a model name (for example llama3.2:3b).")

    def scrub(self, text: str) -> str:
        """Remove the API key from any text that might be shown or logged."""
        text = str(text)
        if self.api_key and len(self.api_key) >= 4:
            text = text.replace(self.api_key, "***")
        return text


# transport(url, body_bytes, headers, timeout) -> response bytes
Transport = Callable[[str, bytes, dict, float], bytes]


def _urllib_transport(url: str, body: bytes, headers: dict, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310 (validate() allows http/https only)
        return resp.read()


def chat(cfg: LLMConfig, messages: list[dict], *, transport: Transport | None = None,
         sleep: Callable[[float], None] = time.sleep) -> str:
    """Send ``messages`` and return the reply text."""
    cfg.validate()
    transport = transport or _urllib_transport
    body = json.dumps({
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    attempts = max(1, int(cfg.retries) + 1)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            raw = transport(cfg.endpoint, body, headers, cfg.timeout)
            return _parse_reply(raw)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            last = LLMError(cfg.scrub(f"The server answered HTTP {exc.code}: {detail or exc.reason}"))
            if exc.code in (401, 403):
                raise LLMError("The server refused the API key (HTTP %d). Check the key." % exc.code) from None
            if exc.code == 404:
                raise LLMError(cfg.scrub(
                    f"Not found (HTTP 404) at {cfg.endpoint}. Check the URL and that model "
                    f"'{cfg.model}' exists on the server.")) from None
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise last from None
        except urllib.error.URLError as exc:
            last = LLMError(cfg.scrub(f"Cannot reach {cfg.base_url}: {exc.reason}. Is the model server running?"))
        except (TimeoutError, OSError) as exc:
            last = LLMError(cfg.scrub(f"The request to {cfg.base_url} failed: {exc}"))
        except LLMError:
            raise
        if attempt + 1 < attempts:
            sleep(min(2 ** attempt, 8))
    raise last or LLMError("The request failed.")


def _parse_reply(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, AttributeError) as exc:
        raise LLMError("The server's reply was not JSON. Is the URL an OpenAI-compatible /v1 endpoint?") from exc
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message") if isinstance(err, dict) else err
        raise LLMError(f"The server returned an error: {str(msg)[:300]}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("The server's reply had no message. Is the URL an OpenAI-compatible endpoint?") from exc
    if isinstance(content, list):          # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "").strip()


def test_connection(cfg: LLMConfig, *, transport: Transport | None = None) -> str:
    reply = chat(cfg, [{"role": "user", "content": "Reply with the single word: ready"}],
                 transport=transport, sleep=lambda _s: None)
    return reply[:80]
