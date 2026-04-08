<div align="center">

<img src="https://raw.githubusercontent.com/BenAmpel/NoCode-Deep-Learning/main/assets/logo.svg" alt="NoCode Deep Learning Studio" width="180"/>

# NoCode Deep Learning Studio

**Train, evaluate, and export deep learning models — no Python required.**

A local-first, privacy-preserving desktop application for building ML pipelines across six data modalities, designed for researchers, educators, and domain experts.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: macOS | Windows](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey?logo=apple)](https://github.com/BenAmpel/NoCode-Deep-Learning/releases)
[![macOS Installer](https://img.shields.io/badge/macOS-797%20KB%20.pkg-blue?logo=apple)](https://github.com/BenAmpel/NoCode-Deep-Learning/releases/latest)
[![Windows Installer](https://img.shields.io/badge/Windows-2.3%20MB%20.exe-blue?logo=windows)](https://github.com/BenAmpel/NoCode-Deep-Learning/releases/latest)
[![JOSS](https://joss.theoj.org/papers/YOUR_JOSS_DOI/status.svg)](https://joss.theoj.org/papers/YOUR_JOSS_DOI)

[**Download**](#installation) · [**Documentation**](#usage) · [**Cite**](#citation) · [**Contribute**](#contributing)

---

<img src="https://raw.githubusercontent.com/BenAmpel/NoCode-Deep-Learning/main/assets/screenshot-hero.png" alt="NoCode Deep Learning Studio hero" width="85%"/>

</div>

---

## Why NoCode Deep Learning Studio?

Training a deep learning model in PyTorch takes 50–100 lines of Python — before you even think about data loading, GPU configuration, or evaluation. For domain experts in biology, journalism, medicine, or business, that barrier is insurmountable.

**NoCode Deep Learning Studio** provides a guided, tab-based interface that mirrors the conceptual stages of a machine learning workflow: **Data → Model → Train → Evaluate → Export**. It runs entirely on your own machine, requires no Python installation, no cloud account, and no command-line experience.

> *"Your data never leaves this machine."*

---

## Features

| | |
|---|---|
| **6 Data Modalities** | Image, Tabular, Text, Audio, Time Series, Video |
| **40+ Architectures** | ResNet, ViT, MobileNet, BERT, Whisper, XGBoost, and more |
| **Built-in Tutorials** | One-click datasets for every modality — no manual download needed |
| **Zero Prerequisites** | Ships a 2.3 MB installer; bootstraps Python + dependencies on first launch |
| **Fully Local** | No cloud account, no data upload, no billing |
| **Explainability** | GradCAM, SHAP, token importance — shown by default, not buried in menus |
| **ONNX Export** | Export to ONNX + auto-generate FastAPI server, Docker bundle, Streamlit app, or model card |
| **Batch Prediction** | Run inference on an entire folder; download results as CSV |
| **Apple Silicon Native** | MPS acceleration with mixed precision; no Rosetta emulation |
| **Windows CI Build** | Windows installer validated in native GitHub Actions runners |

---

## Installation

### Desktop Installers (recommended for non-developers)

Download the latest installer from the [Releases page](https://github.com/BenAmpel/NoCode-Deep-Learning/releases/latest):

| Platform | File | Size | Notes |
|---|---|---|---|
| macOS (Intel + Apple Silicon) | `NoCode-DL-x.x.x.pkg` | 797 KB | Signed & notarised; works with Gatekeeper |
| Windows (x64 + ARM64) | `NoCode-DL-Setup-x.x.x.exe` | 2.3 MB | Detects architecture automatically |

On first launch the app downloads Python 3.12 and installs all dependencies (~5 min on typical WiFi). Subsequent launches take 3–5 seconds.

### Developer Install

```bash
git clone https://github.com/BenAmpel/NoCode-Deep-Learning.git
cd NoCode-Deep-Learning
python3 install.py       # creates .venv and installs requirements
python3 run_local.py     # launches at http://127.0.0.1:7860
```

> **Requirements:** Python 3.12, macOS 12+ or Windows 10+. GPU optional (MPS on Apple Silicon, CUDA on NVIDIA).

---

## Supported Modalities

| Modality | Task Types | Example Architectures | Built-in Tutorial |
|---|---|---|---|
| **Image** | Classification, Object Detection | ResNet, ViT, MobileNetV3, YOLOv8 | MNIST digits (10 classes) |
| **Tabular** | Classification, Regression | XGBoost, LightGBM, MLP, RandomForest | Iris species (3 classes) |
| **Text** | Classification, Sentiment | BERT, DistilBERT, TF-IDF + LR | 20 Newsgroups (3 classes) |
| **Audio** | Classification | CNN-Spectrogram, Whisper features | Free Spoken Digit Dataset |
| **Time Series** | Classification, Forecasting | 1D-CNN, LSTM, Transformer | Synthetic sinusoidal signals |
| **Video** | Classification | 3D-CNN, SlowFast | Synthetic shape clips |

---

## Workflow

The interface is organised into sequential tabs that make the learning sequence explicit:

```
 Data  →  Model  →  Train  →  Evaluate  →  Dashboard  →  Try Your Model  →  Export
```

1. **Data** — Upload or select a dataset. Validates structure, infers schema, reports class balance, flags quality issues.
2. **Model** — Choose from 40+ architectures. A recommendation engine suggests models based on dataset size and modality.
3. **Train** — Configure hyperparameters (learning rate, batch size, epochs, augmentation). Watch live loss and accuracy curves.
4. **Evaluate** — Review confusion matrices, ROC curves, GradCAM saliency maps, SHAP plots, and misclassification tables.
5. **Dashboard** — Executive summary: KPI cards, primary diagnostic chart, explainability plot, and auto-generated action items.
6. **Try Your Model** — Single-file inference with ranked confidence bars, or batch-predict a folder and download results as CSV.
7. **Export** — Save as ONNX, generate a FastAPI server, Docker bundle, Streamlit dashboard, or model card.

---

## Screenshots

<div align="center">
<table>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/BenAmpel/NoCode-Deep-Learning/main/assets/screenshot-data.png" width="340"/><br/><sub>Data tab — auto-detected schema & class balance</sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/BenAmpel/NoCode-Deep-Learning/main/assets/screenshot-train.png" width="340"/><br/><sub>Train tab — run configuration & live telemetry</sub></td>
</tr>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/BenAmpel/NoCode-Deep-Learning/main/assets/screenshot-evaluate.png" width="340"/><br/><sub>Evaluate tab — metrics, GradCAM & confusion matrix</sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/BenAmpel/NoCode-Deep-Learning/main/assets/screenshot-export.png" width="340"/><br/><sub>Export tab — FastAPI, Docker & model card generation</sub></td>
</tr>
</table>
</div>

---

## Building Installers

### macOS (signed + notarised)

```bash
python3 packaging/build_macos_installer.py --version 1.0.0
python3 packaging/sign_and_notarize_macos.py \
  --version 1.0.0 \
  --app-sign-identity "Developer ID Application: Your Name (TEAMID)" \
  --installer-sign-identity "Developer ID Installer: Your Name (TEAMID)" \
  --keychain-profile AC_PASSWORD
```

### Windows

Run the Windows packager on a Windows machine or in a Windows GitHub Actions runner:

```powershell
python packaging/build_windows_installer.py --version 1.0.0
```

The build requires Inno Setup 6 (`iscc.exe`) to be installed and available on `PATH`.

---

## Academic Publications

This project has been written up for multiple venues. Manuscript drafts are included in this repository:

| Venue | Focus | Folder |
|---|---|---|
| [JOSS](https://joss.theoj.org/) | Open-source software description | [`JOSS_Manuscript/`](JOSS_Manuscript/) |
| [JMLR MLOSS](https://jmlr.org/mloss/) | ML system design and architecture | [`JMLR_Manuscript/`](JMLR_Manuscript/) |
| [ACM SIGCSE ERT](https://sigcse.org/) | Computing education pedagogy | [`ACM_SIGCSE_Template/`](ACM_SIGCSE_Template/) |
| [IEEE Software](https://www.computer.org/csdl/magazine/so) | Packaging engineering lessons | [`IEEE_Software_Template/`](IEEE_Software_Template/) |

---

## Citation

If you use NoCode Deep Learning Studio in your research or teaching, please cite:

### General / JOSS

```bibtex
@article{ampel2025nocode,
  title   = {{NoCode Deep Learning Studio}: A Local-First, No-Code Desktop Application
             for Multi-Modal Machine Learning},
  author  = {Ampel, Benjamin},
  journal = {Journal of Open Source Software},
  year    = {2025},
  note    = {Under review}
}
```

### JMLR Machine Learning Open Source Software

```bibtex
@article{ampel2025nocode_jmlr,
  title   = {{NoCode Deep Learning Studio}: A Multi-Modal, Local-First Machine Learning
             Workbench for Non-Developer Users},
  author  = {Ampel, Benjamin},
  journal = {Journal of Machine Learning Research},
  year    = {2025},
  note    = {Under review (MLOSS track)}
}
```

### ACM SIGCSE (Education)

```bibtex
@inproceedings{ampel2025nocode_sigcse,
  title     = {Removing the {Python} Barrier: A No-Code Desktop Application for
               Teaching Multi-Modal Deep Learning},
  author    = {Ampel, Benjamin},
  booktitle = {Proceedings of the ACM SIGCSE Technical Symposium on Computer Science Education},
  year      = {2025},
  note      = {Under review}
}
```

### IEEE Software (Packaging Engineering)

```bibtex
@article{ampel2025nocode_ieee,
  title   = {Packaging a {Python} {ML} Application as a Native Desktop Installer:
             Lessons Learned},
  author  = {Ampel, Benjamin},
  journal = {IEEE Software},
  year    = {2025},
  note    = {Under review}
}
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss the change you have in mind.

```bash
git clone https://github.com/BenAmpel/NoCode-Deep-Learning.git
cd NoCode-Deep-Learning
python3 install.py
python3 run_local.py
```

Areas where help is especially welcome:
- **New modalities** — video segmentation, graph data, multi-label classification
- **Additional architectures** — add entries to `models/registry.py`
- **Localisation** — interface currently English-only
- **Controlled evaluation** — classroom studies measuring learning outcomes
- **Accessibility** — keyboard navigation, screen-reader support

Please read [`SETUP.md`](SETUP.md) for environment details and [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) for the release process.

---

## License

[MIT](LICENSE) © Benjamin Ampel, University of Arizona

---

<div align="center">

Built with [PyTorch](https://pytorch.org/) · [Gradio](https://gradio.app/) · [scikit-learn](https://scikit-learn.org/) · [ONNX Runtime](https://onnxruntime.ai/)

*Runs on your machine. Your data stays with you.*

</div>
