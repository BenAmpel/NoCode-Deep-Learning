# Changelog

All notable changes to NoCode Deep Learning Studio are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](https://semver.org/).

---

## [2.0.0] - 2026-10-15

A stability release after a full audit: training results you can trust, data that can't be corrupted, current dependencies with no known vulnerabilities, and a notarized macOS download. It's a major version because of the breaking changes below.

### Breaking changes
- **macOS now ships as a notarized DMG instead of a .pkg.** Drag NoCode-DL to Applications. If Python 3.12 isn't installed, the app offers to install it from python.org on first launch (it asks for an administrator password). Upgrading from 1.0.x: replace the old app; your `outputs/` folder is not touched.
- **XGBoost is no longer offered or installed on macOS.** Its wheels need Homebrew's `libomp`, and with PyTorch loaded a second OpenMP runtime crashes the app. Use HistGradientBoosting instead; guided mode now recommends it. Saved configurations that name XGBoost explain the switch instead of failing.
- **Model files are loaded without running code.** Checkpoints load with `weights_only=True`, and scikit-learn bundles (pickle) load only from the app's own `outputs/` folder. Bundles downloaded or shared from elsewhere are refused; re-export them in this version, or move them into `outputs/` if you trust them.
- **Regression models scale their targets during training**, and splits are now stratified for classification. Metrics and predictions differ from 1.0.x for the same data and settings (they're now correct); retrain models you compare against.
- **Gradio 6 and transformers 5** replace Gradio 4 and transformers 4; Python 3.11 to 3.13 is required.

### Added
- HistGradientBoosting (scikit-learn's gradient-boosted trees) for tabular and time-series data on every platform, including ONNX export.
- WAV, MP3, FLAC and OGG audio decode without ffmpeg (only `.m4a` still needs it; the dataset preview warns before training).
- Exported FastAPI, Docker and CLI bundles share one inference runtime with the app, so served predictions match the app's.

### Fixed
- Stop now stops: between training batches, between cross-validation folds, and during long scikit-learn and XGBoost fits.
- Data leakage between training and validation, double subsetting, class weights with a missing class, and the warmup-cosine schedule at one epoch.
- Annotation tools no longer corrupt datasets; CSV and TSV delimiters and ID columns are detected; the quality report no longer says "Ready" when there are errors; uploaded zips are size-limited.
- One heavy job runs at a time; project settings save without losing keys; plots no longer leak memory.
- Dark mode follows the theme you pick, and the light-mode headline is readable (contrast was about 1.2:1).
- Apple Silicon: falls back to CPU when MPS is reported but unusable.
- macOS and Windows installer and launcher bugs.

### Security
- All dependencies updated; `pip-audit` reports no known vulnerabilities and gates CI, along with bandit.
- Untrusted model files can no longer run code when loaded (see Breaking changes).

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
