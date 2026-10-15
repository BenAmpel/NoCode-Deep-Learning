"""
Docker deployment file generator.

Writes a self-contained Docker setup into an exported model bundle so it can be
containerised and served with one command. It (re)generates the FastAPI server
first, so Docker export works on its own.

Generated files
---------------
main.py, requirements_serve.txt  — the FastAPI server (see export/fastapi_generator.py)
Dockerfile                       — python:3.12-slim image running main.py
docker-compose.yml               — one service, published on 127.0.0.1 only
.dockerignore                    — excludes build noise and checkpoints
README_deploy.md                 — build / run / test instructions

Usage
-----
>>> from export.docker_generator import generate_docker_bundle
>>> paths = generate_docker_bundle("/path/to/bundle", modality="image", port=8000)
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from export.fastapi_generator import FILE_MODALITIES, generate_fastapi_app


def dockerfile_content(modality: str, port: int = 8000) -> str:
    """Dockerfile for serving the bundle's FastAPI app on ``port``."""
    return dedent(f"""\
        # NoCode-DL inference image ({modality})
        # Python 3.12 matches the environment the model was trained and exported in.
        FROM python:3.12-slim

        ENV PYTHONDONTWRITEBYTECODE=1 \\
            PYTHONUNBUFFERED=1

        WORKDIR /app

        # libgomp1: OpenMP runtime used by onnxruntime; libglib2.0-0: OpenCV.
        RUN apt-get update && \\
            apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && \\
            rm -rf /var/lib/apt/lists/*

        COPY requirements_serve.txt .
        RUN pip install --no-cache-dir -r requirements_serve.txt

        COPY . .

        # Run as an unprivileged user.
        RUN useradd --create-home --uid 10001 app && chown -R app /app
        USER app

        # Inside the container the server must listen on all interfaces; what is
        # reachable from outside is decided by the port mapping (see compose file).
        ENV HOST=0.0.0.0 PORT={port}
        EXPOSE {port}

        HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \\
            CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health')" || exit 1

        CMD ["python", "main.py"]
    """)


def _docker_compose_content(modality: str, port: int) -> str:
    return dedent(f"""\
        # NoCode-DL deployment ({modality}). Usage: docker compose up --build
        services:
          nocode-dl:
            build: .
            image: nocode-dl-{modality}:latest
            container_name: nocode-dl-{modality}
            ports:
              # Reachable only from this computer. To expose it on your network,
              # change to "{port}:{port}" and set NOCODE_DL_API_KEY.
              - "127.0.0.1:{port}:{port}"
            environment:
              - NOCODE_DL_API_KEY=${{NOCODE_DL_API_KEY:-}}
            restart: unless-stopped
            deploy:
              resources:
                limits:
                  memory: 2G
    """)


def _dockerignore_content() -> str:
    return dedent("""\
        __pycache__/
        *.pyc
        .git/
        .env
        .env.*
        .DS_Store
        .ipynb_checkpoints/
        # Training checkpoints aren't needed to serve (model.onnx / model.joblib are)
        checkpoints/
        *.ckpt
    """)


def _example_request(modality: str, port: int, prep: dict) -> str:
    url = f"http://127.0.0.1:{port}/predict"
    if modality in FILE_MODALITIES:
        sample = {"image": "photo.jpg", "audio": "clip.wav", "video": "clip.mp4"}[modality]
        return f"curl -X POST {url} -F 'file=@{sample}'"
    if modality == "tabular":
        example = {c: 0.0 for c in prep.get("numeric_cols", [])[:3]}
        for col, cats in list(prep.get("ohe_categories", {}).items())[:2]:
            example[col] = str(cats[0]) if cats else ""
        body = json.dumps({"features": example})
    elif modality == "timeseries":
        cols = prep.get("raw_feature_columns") or prep.get("feature_columns") or ["x"]
        body = json.dumps({"rows": [[0.0] * len(cols)] * 2})
    else:
        body = json.dumps({"text": "an example sentence"})
    return f"curl -X POST {url} -H 'Content-Type: application/json' -d '{body}'"


def _readme_deploy_content(modality: str, port: int, prep: dict | None = None) -> str:
    prep = prep or {}
    return dedent(f"""\
        # Deploying this {modality} model with Docker

        ## 1. Build and start

        ```bash
        docker compose up --build
        ```

        or without Compose:

        ```bash
        docker build -t nocode-dl-{modality} .
        docker run --rm -p 127.0.0.1:{port}:{port} nocode-dl-{modality}
        ```

        The server is only reachable from this computer. To reach it from other
        machines, publish the port on all interfaces and set an API key:

        ```bash
        docker run --rm -p {port}:{port} -e NOCODE_DL_API_KEY=choose-a-secret nocode-dl-{modality}
        ```

        Clients then send the key in an `X-API-Key` header.

        ## 2. Check it is up

        ```bash
        curl http://127.0.0.1:{port}/health
        ```

        ## 3. Predict

        ```bash
        {_example_request(modality, port, prep)}
        ```

        Classification models answer with
        `{{"task": "classification", "predictions": [{{"label": ..., "confidence": ..., "rank": 1}}, ...]}}`;
        regression models with `{{"task": "regression", "value": ...}}`.
        Interactive API docs: http://127.0.0.1:{port}/docs

        Uploads are limited to 25 MB by default (`NOCODE_DL_MAX_UPLOAD_MB`).
    """)


def generate_docker_bundle(bundle_path: str, modality: str | None = None, port: int = 8000) -> list[str]:
    """Write the FastAPI server and Docker files into ``bundle_path``.

    The modality comes from the bundle's preprocessing.json (the argument is
    kept for backwards compatibility). Returns the paths written, Docker files
    first: [Dockerfile, docker-compose.yml, .dockerignore, README_deploy.md,
    main.py].
    """
    bundle = Path(bundle_path)
    if not bundle.exists():
        raise FileNotFoundError(
            f"Bundle directory not found: {bundle_path}\n"
            "Export the model bundle first before generating Docker files."
        )
    prep = json.loads((bundle / "preprocessing.json").read_text(encoding="utf-8"))
    modality = prep.get("modality") or modality or ""
    main_path = generate_fastapi_app(str(bundle), modality)

    created: list[str] = []
    for filename, content in (
        ("Dockerfile", dockerfile_content(modality, port)),
        ("docker-compose.yml", _docker_compose_content(modality, port)),
        (".dockerignore", _dockerignore_content()),
        ("README_deploy.md", _readme_deploy_content(modality, port, prep)),
    ):
        path = bundle / filename
        path.write_text(content, encoding="utf-8")
        created.append(str(path))
    created.append(main_path)
    return created
