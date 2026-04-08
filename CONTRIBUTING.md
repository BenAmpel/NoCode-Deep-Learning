# Contributing to NoCode Deep Learning Studio

Thanks for your interest in contributing! This document covers how to get set up, what kinds of contributions are most welcome, and the review process.

---

## Getting started

```bash
git clone https://github.com/BenAmpel/NoCode-Deep-Learning.git
cd NoCode-Deep-Learning
python3 install.py        # creates .venv and installs dependencies
python3 run_local.py      # launches at http://127.0.0.1:7860
```

> **Requirements:** Python 3.12, macOS 12+ or Windows 10+.

---

## What we're looking for

| Area | Examples |
|---|---|
| **New modalities** | Video segmentation, graph data, multi-label classification |
| **New architectures** | Add entries to `models/registry.py` |
| **Bug fixes** | Crash reports, UI glitches, training failures |
| **Evaluation metrics** | Additional plots, explainability methods |
| **Export formats** | New deployment targets beyond ONNX/FastAPI/Docker |
| **Documentation** | Tutorials, worked examples, translations |
| **Tests** | Unit tests for data pipeline, model registry, export |

---

## Code style

- Python 3.12+, formatted with `ruff`
- Keep functions short and single-purpose
- New UI components go in `ui/`, new model families in `models/`
- All modality-specific logic is gated behind the `modality` string — follow the existing pattern

---

## Adding a new model architecture

1. Add an entry to `models/registry.py` with the architecture name, task compatibility, and builder function
2. If it needs a custom forward pass, add it to `models/`
3. Update the modalities table in `README.md`
4. Test with at least one built-in tutorial dataset

---

## Submitting a pull request

1. Fork the repo and create a feature branch
2. Make your changes with clear, focused commits
3. Open a PR against `main` — fill in the PR template
4. A maintainer will review within a few days

---

## Reporting bugs

Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml) issue template. Include your platform, app version, and steps to reproduce.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
