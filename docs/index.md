# NoCode Deep Learning Studio — Documentation

Welcome to the documentation for **NoCode Deep Learning Studio**, a local-first desktop application for training, evaluating, and exporting deep learning models without writing code.

## Contents

- [Installation](installation.md) — desktop installers, developer setup, system requirements
- [Quickstart & Tutorials](tutorials.md) — your first model in 10 minutes, built-in datasets
- [Architecture & Packaging](architecture.md) — how the app is built and distributed

## Overview

The application exposes seven sequential tabs that mirror the stages of a real ML project:

| Tab | Purpose |
|---|---|
| **Data** | Upload or point at a dataset; the app infers schema, detects modality, and reports quality issues |
| **Model** | Choose from 40+ architectures; a recommendation engine suggests models based on your data |
| **Train** | Configure hyperparameters and run training; watch live loss and accuracy curves |
| **Evaluate** | Review confusion matrices, ROC curves, GradCAM saliency maps, and SHAP plots |
| **Dashboard** | Executive summary — KPI cards, primary diagnostic, explainability, action items |
| **Try Your Model** | Single-file or batch inference; download results as CSV |
| **Export** | Save as ONNX; generate FastAPI server, Docker bundle, Streamlit app, or model card |

## Supported Modalities

Image · Tabular · Text · Audio · Time Series · Video · Object Detection

## Quick links

- [GitHub repository](https://github.com/BenAmpel/NoCode-Deep-Learning)
- [Releases & installers](https://github.com/BenAmpel/NoCode-Deep-Learning/releases)
- [Bug reports](https://github.com/BenAmpel/NoCode-Deep-Learning/issues/new/choose)
- [Contributing guide](../CONTRIBUTING.md)
