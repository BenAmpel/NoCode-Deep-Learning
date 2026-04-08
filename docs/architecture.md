# Architecture & Packaging

## Application architecture

NoCode Deep Learning Studio is a **local Gradio web application** packaged as a native desktop installer. The stack:

| Layer | Technology |
|---|---|
| UI | [Gradio](https://gradio.app/) (Svelte frontend, Python backend) |
| ML | [PyTorch](https://pytorch.org/) + [scikit-learn](https://scikit-learn.org/) |
| Inference export | [ONNX Runtime](https://onnxruntime.ai/) |
| Packaging | PyInstaller-style bootstrap → native `.pkg` / `.exe` |

The user's browser connects to a local server at `http://127.0.0.1:7860`. No data ever leaves the machine unless the user explicitly exports a bundle.

```
Browser (localhost:7860)
       ↕
  Gradio server (Python)
       ↕
  PyTorch / sklearn
       ↕
  Local filesystem (datasets, model bundles, outputs/)
```

## Two-stage bootstrap

The macOS and Windows installers are intentionally tiny (< 2.5 MB). They contain only a **bootstrap launcher** — not Python or any ML libraries.

**Stage 1 (installer):** Copies the launcher scripts to a local app directory.

**Stage 2 (first launch):** The launcher detects whether a `.venv` exists. If not, it:
1. Downloads the appropriate Python 3.12 installer for the user's OS and architecture
2. Installs Python silently
3. Creates a virtual environment
4. Installs `requirements.txt` via pip

Subsequent launches skip all of this and start the Gradio server directly.

## Cross-platform packaging

### macOS

Built with a custom Python script that produces a `.pkg` installer:
- Signed with a Developer ID Application certificate
- Notarised with Apple's `notarytool`
- Stapled so Gatekeeper passes without an internet connection at install time
- Architecture detection: `uname -m` returns `arm64` (Apple Silicon) or `x86_64` (Intel)

### Windows (cross-compiled on macOS)

The Windows `.exe` is built on macOS using **Wine + Inno Setup 6**:

```bash
brew install wine-stable
# Install Inno Setup 6 into Wine prefix
python3 packaging/build_windows_installer.py --version 1.0.0
```

Architecture detection on Windows uses WMI (`Win32_Processor.Architecture`): `0` = x86, `12` = ARM64.

## Hardware acceleration

| Platform | Accelerator | Notes |
|---|---|---|
| Apple Silicon | MPS | Mixed precision via `torch.autocast("mps")` — no GradScaler needed |
| NVIDIA GPU | CUDA | Mixed precision via `torch.autocast("cuda")` + GradScaler |
| CPU fallback | — | Used automatically when no accelerator is detected |

## Model registry

All supported architectures are registered in `models/registry.py`. Each entry specifies:
- Modality compatibility
- Task types (classification, regression, detection, etc.)
- Builder function
- Default hyperparameter suggestions

Adding a new architecture requires a single registry entry plus (optionally) a custom forward pass in `models/`.

## Export formats

| Format | Description |
|---|---|
| ONNX | Cross-platform inference graph with `onnxruntime` |
| FastAPI | Auto-generated Python server with `/predict` endpoint |
| Docker | `Dockerfile` + `requirements.txt` wrapping the FastAPI server |
| Streamlit | Auto-generated interactive demo app |
| Model card | Markdown summary of training config, metrics, and intended use |
