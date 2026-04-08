# Quickstart & Tutorials

## Your first model in 10 minutes

### Step 1 — Open the app

Launch NoCode Deep Learning Studio. You will see the hero landing page with system information (runtime, hardware accelerator, and workflow overview).

### Step 2 — Load a tutorial dataset

Scroll down to the **Load Tutorial** section and select a modality from the dropdown:

| Modality | Dataset | Classes |
|---|---|---|
| Image | MNIST handwritten digits | 10 |
| Tabular | Iris flower species | 3 |
| Text | 20 Newsgroups (subset) | 3 |
| Audio | Free Spoken Digit Dataset | 10 |
| Time Series | Synthetic sinusoidal signals | 3 |
| Video | Synthetic shape clips | 3 |

Click **Load Tutorial**. The app downloads the dataset (first time only) and pre-fills the Data tab.

### Step 3 — Review the Data tab

Switch to the **Data** tab. You will see:
- Auto-detected schema (columns, label field)
- Class balance preview
- Dataset quality report

### Step 4 — Choose a model

Switch to the **Model** tab. Accept the recommended architecture or choose a different one from the dropdown. The recommendation engine suggests models based on your dataset size and modality.

### Step 5 — Train

Switch to the **Train** tab. Set a bundle name, accept the default hyperparameters, and click **Start Training**. Watch the live loss and accuracy curves update in real time.

### Step 6 — Evaluate

Once training completes, switch to **Evaluate**. Review:
- Overall accuracy and loss
- Confusion matrix
- GradCAM saliency maps (image modality)
- SHAP feature importance (tabular modality)

### Step 7 — Export

Switch to **Export & History**. Click **Generate FastAPI Server** or **Export ONNX** to produce a deployment artifact.

---

## Using your own data

### Image data

Organise images into a folder-per-class structure:

```
dataset/
  cat/
    img001.jpg
    img002.jpg
  dog/
    img001.jpg
```

Upload the top-level folder in the **Data** tab.

### Tabular data

Provide a CSV with one row per sample. Include a `label` column (or specify the label column name in the Data tab settings).

### Text data

Provide a CSV with a `text` column and a `label` column.

### Audio data

Organise `.wav` files into a folder-per-class structure, same as image data.

---

## Batch prediction

In the **Try Your Model** tab, select the **Batch** sub-tab. Point it at a folder of files and click **Run**. Download the results as a CSV with filenames and predicted labels.
