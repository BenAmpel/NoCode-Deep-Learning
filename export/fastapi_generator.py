"""
Generates a FastAPI inference server inside a model bundle folder.

The generated main.py is a thin HTTP layer over bundle_runtime.py (copied into
every bundle by the exporter), so served predictions use exactly the same
preprocessing as the app's "Try Your Model" tab.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from export.exporter import RUNTIME_SOURCE, _pinned, runtime_requirements

SERVE_PACKAGES = ["fastapi", "uvicorn[standard]", "python-multipart"]
FILE_MODALITIES = {"image", "audio", "video"}


def serve_requirements(prep: dict, bundle: Path) -> list[str]:
    reqs = runtime_requirements(prep, bundle)
    for pkg in SERVE_PACKAGES:
        base = pkg.split("[")[0]
        pinned = _pinned(base)
        reqs.append(pinned.replace(base, pkg, 1))
    return reqs


def generate_fastapi_app(bundle_path: str, modality: str | None = None) -> str:
    """Write main.py, requirements_serve.txt (and bundle_runtime.py if missing).

    The modality is read from the bundle's preprocessing.json; the ``modality``
    argument is accepted for backwards compatibility but the bundle wins, so a
    stale UI selection can't produce a server for the wrong input type.
    Returns the path to main.py.
    """
    bundle = Path(bundle_path)
    prep = json.loads((bundle / "preprocessing.json").read_text(encoding="utf-8"))
    bundle_modality = prep.get("modality") or modality or ""
    if bundle_modality == "graph":
        raise ValueError("Graph bundles can't be served over HTTP yet: inference needs the full graph.")
    if bundle_modality not in FILE_MODALITIES | {"tabular", "timeseries", "text"}:
        raise ValueError(f"Serving is not supported for '{bundle_modality}' bundles.")

    if not (bundle / "bundle_runtime.py").is_file():  # bundles exported before the runtime existed
        shutil.copy2(RUNTIME_SOURCE, bundle / "bundle_runtime.py")

    main_path = bundle / "main.py"
    main_path.write_text(_app_code(bundle_modality), encoding="utf-8")
    (bundle / "requirements_serve.txt").write_text(
        "\n".join(serve_requirements(prep, bundle)) + "\n", encoding="utf-8")
    return str(main_path)


_HEADER = '''\
"""
FastAPI server for this NoCode-DL {modality} model.

Preprocessing comes from bundle_runtime.py in this folder: the same code the
NoCode-DL app uses, so predictions match "Try Your Model".

Run (reachable only from this computer):
    pip install -r requirements_serve.txt
    python main.py

Interactive docs: http://127.0.0.1:8000/docs

Environment variables:
    NOCODE_DL_API_KEY        if set, /predict requires this value in an X-API-Key header
    NOCODE_DL_MAX_UPLOAD_MB  largest accepted upload in MB (default 25)
    HOST, PORT               bind address (default 127.0.0.1) and port (default 8000).
                             Only bind 0.0.0.0 on a trusted network, ideally with an API key.
"""
import hmac
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bundle_runtime import Bundle, BundleInputError  # noqa: E402

log = logging.getLogger("nocode_dl.server")
BUNDLE = Bundle(HERE)
API_KEY = os.environ.get("NOCODE_DL_API_KEY", "")
MAX_UPLOAD_BYTES = int(float(os.environ.get("NOCODE_DL_MAX_UPLOAD_MB", "25")) * 1024 * 1024)

app = FastAPI(title="NoCode-DL {modality} model", version="1.0")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and not (x_api_key and hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def run_prediction(raw: Any, top_k: int) -> dict:
    try:
        return BUNDLE.predict(raw, top_k=top_k)
    except BundleInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        # Log the details here; don't send internals back to the caller.
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. See the server log.")


@app.get("/health")
def health() -> dict:
    info = {{"status": "ok", "modality": BUNDLE.modality, "task": BUNDLE.task}}
    if BUNDLE.task == "classification":
        info["classes"] = BUNDLE.classes
    return info

'''

_FILE_ENDPOINT = '''
@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict:
    """Upload one {modality} file."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than NOCODE_DL_MAX_UPLOAD_MB.")
    if BUNDLE.modality == "image":
        return run_prediction(data, top_k)
    # Audio/video decoders need a real file; keep the extension for format detection.
    suffix = Path(file.filename or "").suffix[:10] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        return run_prediction(tmp.name, top_k)
    finally:
        os.unlink(tmp.name)
'''

_JSON_ENDPOINTS = {
    "tabular": '''
class TabularInput(BaseModel):
    features: dict[str, Any]   # raw column values, e.g. {{"age": 31, "city": "oslo"}}
    top_k: int = 3


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(payload: TabularInput) -> dict:
    return run_prediction(payload.features, payload.top_k)
''',
    "timeseries": '''
class TimeseriesInput(BaseModel):
    rows: list[list[float]] | list[dict[str, float]]   # recent rows, oldest first
    top_k: int = 3


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(payload: TimeseriesInput) -> dict:
    return run_prediction(payload.rows, payload.top_k)
''',
    "text": '''
class TextInput(BaseModel):
    text: str
    top_k: int = 3


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(payload: TextInput) -> dict:
    return run_prediction(payload.text, payload.top_k)
''',
}

_FOOTER = '''

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
'''


def _app_code(modality: str) -> str:
    endpoint = _FILE_ENDPOINT if modality in FILE_MODALITIES else _JSON_ENDPOINTS[modality]
    return _HEADER.format(modality=modality) + endpoint.format(modality=modality) + _FOOTER
