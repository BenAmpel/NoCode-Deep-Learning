# Changelog

All notable changes to NoCode Deep Learning Studio are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](https://semver.org/).

---

## [1.0.11] — 2025-04-08

### Added
- Built-in tutorial datasets for all six modalities (MNIST, Iris, 20 Newsgroups, Free Spoken Digit, synthetic time-series and video)
- Batch prediction: run inference on an entire folder and download results as CSV
- Privacy badge in hero UI ("Your data never leaves this machine")
- Tutorial modality selector dropdown with live preview

### Fixed
- MPS mixed-precision training on Apple Silicon (AMP now works without GradScaler on MPS devices)
- MNIST tutorial path error in packaged installer (data now downloaded at runtime via torchvision)

### Changed
- macOS installer reduced from 44 MB → 797 KB (bootstrap model)
- Windows installer reduced from 11 MB → 2.3 MB

---

## [1.0.6] — 2025-04-05

### Added
- Signed and notarised macOS `.pkg` installer (Gatekeeper-compatible)
- Cross-compiled Windows installer built on macOS via Wine + Inno Setup
- Architecture detection: Apple Silicon vs Intel (macOS), x64 vs ARM64 (Windows)
- Executive Dashboard tab with KPI cards, primary diagnostic chart, and auto-generated action items
- Run comparison and K-fold cross-validation tabs

### Fixed
- Windows Defender false positives from batch-file launcher (replaced with PowerShell)
- `os.execv` replacement failure on Windows (replaced with `threading.Timer` + `sys.exit`)

---

## [1.0.0] — 2025-03-30

### Added
- Initial release
- Six data modalities: Image, Tabular, Text, Audio, Time Series, Video
- 40+ model architectures including ResNet, ViT, MobileNetV3, BERT, DistilBERT, Whisper, XGBoost, LightGBM
- Guided no-code workflow: Data → Model → Train → Evaluate → Dashboard → Try Your Model → Export
- ONNX export with auto-generated FastAPI server, Docker bundle, Streamlit app, and model card
- GradCAM, SHAP, and token-importance explainability
- Object detection tab (YOLOv8)
- Local-first architecture — no data leaves the machine
