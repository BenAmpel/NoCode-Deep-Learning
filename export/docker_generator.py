"""
Docker deployment file generator.

Writes a self-contained Docker bundle alongside an existing model export bundle
so the bundle can be containerised and served with a single command.

Generated files
---------------
Dockerfile            — multi-stage-free production image based on python:3.11-slim
docker-compose.yml    — one-service compose file with port mapping and volume
.dockerignore         — excludes build noise; explicitly INCLUDES *.onnx models
README_deploy.md      — step-by-step human instructions for build / run / test

Usage
-----
>>> from export.docker_generator import generate_docker_bundle
>>> paths = generate_docker_bundle("/path/to/bundle", modality="image", port=8000)
"""
from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent


# ─────────────────────────────────────────────────────────────────────────────
# Modality-specific pip requirements
# ─────────────────────────────────────────────────────────────────────────────

_MODALITY_EXTRAS: dict[str, list[str]] = {
    "image": ["Pillow>=10.0.0", "opencv-python-headless>=4.8.0"],
    "video": ["opencv-python-headless>=4.8.0"],
    "audio": ["torchaudio>=2.1.0", "librosa>=0.10.0"],
    "text":  [],
    "tabular": ["pandas>=2.0.0", "scikit-learn>=1.3.0"],
    "timeseries": ["pandas>=2.0.0"],
}


def _modality_pip_lines(modality: str) -> str:
    """Return a newline-joined list of extra pip packages for the modality."""
    extras = _MODALITY_EXTRAS.get(modality, [])
    return "\n".join(extras)


# ─────────────────────────────────────────────────────────────────────────────
# Individual file generators
# ─────────────────────────────────────────────────────────────────────────────

def dockerfile_content(modality: str, port: int = 8000) -> str:
    """
    Return a Dockerfile string for the given modality and port.

    The image is based on ``python:3.11-slim`` and:
    * copies all bundle files into ``/app``,
    * installs dependencies from ``requirements_serve.txt``,
    * exposes ``port`` and launches the FastAPI app via uvicorn.

    Can be called standalone to preview the Dockerfile without writing to disk.

    Parameters
    ----------
    modality : str
        One of ``"image"``, ``"video"``, ``"audio"``, ``"text"``,
        ``"tabular"``, ``"timeseries"``.
    port : int
        The port uvicorn will listen on inside the container.

    Returns
    -------
    str
        Complete Dockerfile content.
    """
    extra_comment = ""
    if modality in ("image", "video"):
        extra_comment = (
            "# opencv-python-headless is included for image/video pre-processing.\n"
            "# If you need a display (e.g. cv2.imshow), switch to opencv-python.\n"
        )
    elif modality == "audio":
        extra_comment = (
            "# torchaudio is included for audio loading and mel-spectrogram extraction.\n"
        )

    return dedent(f"""\
        # ──────────────────────────────────────────────────────────────────────
        # NoCode-DL inference image — modality: {modality}
        # Base: python:3.11-slim  |  Port: {port}
        # ──────────────────────────────────────────────────────────────────────
        FROM python:3.11-slim

        # Keeps Python from generating .pyc files and enables stdout/stderr logging
        ENV PYTHONDONTWRITEBYTECODE=1 \\
            PYTHONUNBUFFERED=1

        WORKDIR /app

        # Install system-level dependencies (minimal)
        RUN apt-get update && \\
            apt-get install -y --no-install-recommends \\
                libglib2.0-0 \\
                libgomp1 && \\
            rm -rf /var/lib/apt/lists/*

        # Copy requirements first to exploit Docker layer caching
        COPY requirements_serve.txt .

        {extra_comment.rstrip()}
        RUN pip install --no-cache-dir --upgrade pip && \\
            pip install --no-cache-dir -r requirements_serve.txt

        # Copy all bundle files (model.onnx, main.py, labels.json, preprocessing.json, …)
        COPY . .

        # Expose the application port
        EXPOSE {port}

        # Health check — calls the /health endpoint every 30 s
        HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
            CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

        # Launch the FastAPI app with uvicorn
        CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
    """)


def _docker_compose_content(modality: str, port: int) -> str:
    """Return a docker-compose.yml string."""
    return dedent(f"""\
        # docker-compose.yml — NoCode-DL deployment ({modality})
        # Usage: docker-compose up --build
        version: "3.9"

        services:
          nocode-dl:
            build: .
            image: nocode-dl-{modality}:latest
            container_name: nocode-dl-{modality}
            ports:
              - "{port}:{port}"
            volumes:
              # Mount the local bundle directory so model files are accessible
              - .:/app:ro
            restart: unless-stopped
            environment:
              - PYTHONUNBUFFERED=1
            # Resource limits — adjust to your hardware
            deploy:
              resources:
                limits:
                  memory: 2G
    """)


def _dockerignore_content() -> str:
    """
    Return a .dockerignore string.

    Excludes build noise and large PyTorch checkpoints (.pt/.pth) that are
    NOT needed at inference time — the bundle uses model.onnx instead.
    The ``!*.onnx`` rule ensures the exported ONNX model IS included.
    """
    return dedent("""\
        # ── Python build artefacts ────────────────────────────────────────────
        __pycache__/
        *.pyc
        *.pyo
        *.pyd
        *.egg-info/
        dist/
        build/
        .eggs/

        # ── Version control ───────────────────────────────────────────────────
        .git/
        .gitignore
        .gitattributes

        # ── PyTorch checkpoints (large; not needed — use model.onnx instead) ─
        *.pt
        *.pth
        *.ckpt

        # ── Explicitly INCLUDE ONNX models (the inference artifact) ──────────
        !*.onnx

        # ── Environment / secrets ─────────────────────────────────────────────
        .env
        .env.*
        *.env
        secrets/

        # ── IDE / OS noise ────────────────────────────────────────────────────
        .vscode/
        .idea/
        .DS_Store
        Thumbs.db

        # ── Jupyter / notebooks ───────────────────────────────────────────────
        *.ipynb
        .ipynb_checkpoints/

        # ── Test / CI artefacts ───────────────────────────────────────────────
        tests/
        .pytest_cache/
        htmlcov/
        .coverage
    """)


def _readme_deploy_content(modality: str, port: int) -> str:
    """Return a README_deploy.md string with step-by-step deployment instructions."""
    # Pre-compute the modality-specific curl snippet lines so no backslashes
    # appear inside f-string expression slots (not allowed in Python < 3.12).
    if modality in ("image", "audio"):
        request_section_header = "### Image / Audio file upload"
        ext = "jpg" if modality == "image" else "wav"
        curl_line1 = f"curl -X POST http://localhost:{port}/predict \\"
        curl_line2 = f"  -F 'file=@/path/to/your/sample.{ext}' \\"
        curl_line3 = "  --output -"
    elif modality == "video":
        request_section_header = "### Video file upload"
        curl_line1 = f"curl -X POST http://localhost:{port}/predict \\"
        curl_line2 = "  -F 'file=@/path/to/your/video.mp4' \\"
        curl_line3 = "  --output -"
    elif modality == "tabular":
        request_section_header = "### JSON payload"
        curl_line1 = f"curl -X POST http://localhost:{port}/predict \\"
        curl_line2 = "  -H 'Content-Type: application/json' \\"
        curl_line3 = "  -d '{\"features\": [1.0, 2.0, 3.0]}'"
    else:
        request_section_header = "### JSON payload"
        curl_line1 = f"curl -X POST http://localhost:{port}/predict \\"
        curl_line2 = "  -H 'Content-Type: application/json' \\"
        curl_line3 = "  -d '{\"data\": [[1.0, 0.5], [0.3, 0.8]]}'"

    return dedent(f"""\
        # NoCode-DL Deployment Guide — {modality.capitalize()} Model

        This bundle contains everything needed to serve your trained model via a
        containerised FastAPI REST API.

        ## Prerequisites

        - [Docker](https://docs.docker.com/get-docker/) >= 24.0
        - [Docker Compose](https://docs.docker.com/compose/install/) >= 2.0
          (ships with Docker Desktop on macOS / Windows)

        ---

        ## Option A — Docker Compose (recommended)

        ```bash
        # 1. Build the image and start the container in one step
        docker-compose up --build

        # The API is now available at http://localhost:{port}
        # Press Ctrl+C to stop.  Use -d to run in the background:
        docker-compose up --build -d
        ```

        ---

        ## Option B — Plain Docker

        ### Step 1 — Build the image

        ```bash
        docker build -t nocode-dl-{modality} .
        ```

        ### Step 2 — Run the container

        ```bash
        docker run --rm -p {port}:{port} nocode-dl-{modality}
        ```

        To run in the background:

        ```bash
        docker run -d --name nocode-dl-{modality} -p {port}:{port} nocode-dl-{modality}
        ```

        ---

        ## Step 3 — Verify the service is healthy

        ```bash
        curl http://localhost:{port}/health
        # Expected: {{"status": "ok", "model_input": "..."}}
        ```

        ---

        ## Step 4 — Send a prediction request

        {request_section_header}

        ```bash
        {curl_line1}
        {curl_line2}
        {curl_line3}
        ```

        Expected response shape:

        ```json
        {{
          "predicted_class": "cat",
          "confidence": 0.9312,
          "all_probabilities": {{
            "cat": 0.9312,
            "dog": 0.0688
          }}
        }}
        ```

        ---

        ## Stopping the service

        ```bash
        # Compose:
        docker-compose down

        # Plain Docker:
        docker stop nocode-dl-{modality}
        ```

        ---

        ## Troubleshooting

        | Symptom | Fix |
        |---------|-----|
        | Port already in use | Change the port in `docker-compose.yml` and rebuild |
        | Out-of-memory error | Increase the `memory` limit in `docker-compose.yml` |
        | Model not found | Ensure `model.onnx` is present in this directory |
        | Slow first request | Normal — ONNX Runtime warms up on the first call |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def generate_docker_bundle(
    bundle_path: str,
    modality: str,
    port: int = 8000,
) -> list[str]:
    """
    Write all Docker deployment files into ``bundle_path`` and return their paths.

    Parameters
    ----------
    bundle_path : str
        Directory that already contains the exported model bundle
        (``model.onnx``, ``main.py``, ``labels.json``, ``preprocessing.json``,
        ``requirements_serve.txt``).  The Docker files are written into the
        **same** directory so the bundle is entirely self-contained.
    modality : str
        The data modality the model operates on.  Accepted values:
        ``"image"``, ``"video"``, ``"audio"``, ``"text"``,
        ``"tabular"``, ``"timeseries"``.
    port : int
        The port the container will expose and uvicorn will bind to.
        Defaults to ``8000``.

    Returns
    -------
    list[str]
        Absolute paths to every file that was created, in the order they were
        written:
        ``[Dockerfile, docker-compose.yml, .dockerignore, README_deploy.md]``

    Raises
    ------
    FileNotFoundError
        If ``bundle_path`` does not exist.
    """
    bundle = Path(bundle_path)
    if not bundle.exists():
        raise FileNotFoundError(
            f"Bundle directory not found: {bundle_path}\n"
            "Export the model bundle first before generating Docker files."
        )

    created: list[str] = []

    def _write(filename: str, content: str) -> str:
        path = bundle / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    # 1. Dockerfile
    created.append(_write("Dockerfile", dockerfile_content(modality, port)))

    # 2. docker-compose.yml
    created.append(_write("docker-compose.yml", _docker_compose_content(modality, port)))

    # 3. .dockerignore
    created.append(_write(".dockerignore", _dockerignore_content()))

    # 4. README_deploy.md
    created.append(_write("README_deploy.md", _readme_deploy_content(modality, port)))

    return created
