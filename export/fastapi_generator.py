"""
Generates a self-contained FastAPI inference app inside the model bundle folder.
The generated app loads the ONNX model and serves predictions over HTTP.
"""
from __future__ import annotations
from pathlib import Path
import json


def generate_fastapi_app(bundle_path: str, modality: str) -> str:
    """
    Writes main.py and requirements_serve.txt to bundle_path.
    Returns the path to main.py.
    """
    bundle = Path(bundle_path)

    with open(bundle / "preprocessing.json") as f:
        prep = json.load(f)
    with open(bundle / "labels.json") as f:
        labels = json.load(f)

    app_code = _generate_app_code(modality, prep, labels)
    main_path = bundle / "main.py"
    main_path.write_text(app_code)

    req_path = bundle / "requirements_serve.txt"
    req_path.write_text(
        "fastapi>=0.104.0\n"
        "uvicorn[standard]>=0.24.0\n"
        "onnxruntime>=1.16.0\n"
        "numpy>=1.24.0\n"
        "python-multipart>=0.0.6\n"
        + _extra_requirements(modality)
    )

    return str(main_path)


def _extra_requirements(modality: str) -> str:
    extras = {
        "image": "Pillow>=10.0.0\n",
        "audio": "torchaudio>=2.1.0\n",
        "video": "opencv-python-headless>=4.8.0\n",
        "text":  "",
        "tabular": "pandas>=2.0.0\n",
        "timeseries": "pandas>=2.0.0\n",
    }
    return extras.get(modality, "")


def _generate_app_code(modality: str, prep: dict, labels: dict) -> str:
    label_map = repr(labels)

    base = f'''\
"""
Auto-generated FastAPI inference server.
Start with: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app    = FastAPI(title="NoCode-DL Inference", version="1.0")
LABELS = {label_map}
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

with open("preprocessing.json") as f:
    PREP = json.load(f)


def run_inference(x: np.ndarray) -> dict:
    if x.ndim == len(session.get_inputs()[0].shape) - 1:
        x = x[np.newaxis]
    logits  = session.run(None, {{INPUT_NAME: x.astype(np.float32)}})[0]
    probs   = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    return {{
        "predicted_class": LABELS[str(pred_idx)],
        "confidence":      round(float(probs[0, pred_idx]), 4),
        "all_probabilities": {{LABELS[str(i)]: round(float(p), 4)
                               for i, p in enumerate(probs[0])}},
    }}

@app.get("/health")
def health():
    return {{"status": "ok", "model_input": INPUT_NAME}}

'''

    if modality in ("image", "audio"):
        return base + '''\
from fastapi import UploadFile, File
from PIL import Image
import io

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img = img.resize(tuple(PREP["resize"]))
        x   = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(PREP["mean"]); std = np.array(PREP["std"])
        x   = (x - mean) / std
        x   = x.transpose(2, 0, 1)   # HWC → CHW
        return run_inference(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
'''

    if modality == "tabular":
        return base + '''\
from pydantic import BaseModel

class TabularInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_tabular(payload: TabularInput):
    try:
        x = np.array(payload.features, dtype=np.float32)
        return run_inference(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
'''

    if modality == "text":
        return base + '''\
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_text(payload: TextInput):
    try:
        vocab     = PREP.get("vocab", {})
        max_len   = PREP.get("max_length", 128)
        tokens    = payload.text.lower().split()[:max_len]
        ids       = [vocab.get(t, 1) for t in tokens]
        ids      += [0] * (max_len - len(ids))
        x         = np.array([ids], dtype=np.int64)
        logits    = session.run(None, {INPUT_NAME: x})[0]
        probs     = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        pred_idx  = int(np.argmax(probs, axis=1)[0])
        return {
            "predicted_class":   LABELS[str(pred_idx)],
            "confidence":        round(float(probs[0, pred_idx]), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
'''

    # timeseries / video fallback
    return base + '''\
from pydantic import BaseModel

class ArrayInput(BaseModel):
    data: list[list[float]]

@app.post("/predict")
def predict(payload: ArrayInput):
    try:
        x = np.array(payload.data, dtype=np.float32)
        return run_inference(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
'''
