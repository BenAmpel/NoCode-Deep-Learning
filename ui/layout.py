"""The Studio's component tree.

``build_layout()`` creates every component inside a new ``gr.Blocks`` and
returns it with the components by name; the callbacks wire events to them.
"""
from __future__ import annotations

import gradio as gr

from config import DEVICE
from models.registry import XGBOOST_UNAVAILABLE_NOTE, xgboost_available
from ui.helpers import (
    AUG_LEVELS,
    CLASSIFICATION_METRIC_OPTIONS,
    HISTORY_COLUMNS,
    MODALITIES,
    OPTIMIZERS,
    PROJECT_MODES,
    SCHEDULERS,
    TASKS,
    YOLO_MODELS,
    _DEFAULT_CLASSIFICATION_NOTE,
    _DEFAULT_DIAGNOSTICS_NOTE,
    _DEFAULT_EXPLANATION_NOTE,
    _DEFAULT_SAMPLE_REVIEW_NOTE,
    _TUTORIAL_MODALITIES,
    _cleaning_preview_empty_html,
    _format_eval_kpis,
    _format_live_training_metrics_card,
    _project_mode_default,
    _section_intro,
    _tutorial_markdown,
    _why_this_matters_markdown,
    tip,
)


class Components(dict):
    """Components by variable name, readable as ``c.name`` or ``c["name"]``."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def build_layout() -> tuple[gr.Blocks, Components]:
    with gr.Blocks(title='NoCode-DL') as demo:
        with gr.Column(elem_classes=["studio-shell"]):
            gr.HTML('<a href="#studio-main-content" class="studio-skip-link">Skip to main content</a>')
            gr.HTML(
                f"""
<section class="studio-hero" aria-label="Welcome">
  <div class="studio-hero__grid">
    <div class="studio-hero__content">
      <div class="studio-hero__eyebrow">NoCode Deep Learning Studio</div>
      <h1 class="studio-hero__title">Build confident deep learning projects.</h1>
      <p class="studio-hero__subtitle">
        A local-first studio: drop in a dataset, let the workspace infer the setup,
        experiment with modern model families, and learn the workflow visually instead of wrestling with boilerplate.
      </p>
      <div class="studio-hero__pills">
        <span class="studio-hero__pill">Guided no-code workflow</span>
        <span class="studio-hero__pill">Pretrained and from-scratch models</span>
        <span class="studio-hero__pill">Runs locally on your machine</span>
      </div>
      <div class="studio-hero__meta">
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Runtime</span>
          <span class="studio-hero__meta-value">Local-first on your machine</span>
        </div>
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Hardware</span>
          <span class="studio-hero__meta-value">Accelerator detected: {DEVICE.upper()}</span>
        </div>
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Workflow</span>
          <span class="studio-hero__meta-value">Data → Model → Train → Evaluate → Export</span>
        </div>
        <div class="studio-hero__meta-card">
          <span class="studio-hero__meta-label">Privacy</span>
          <span class="studio-hero__meta-value">🔒 Your data never leaves this machine</span>
        </div>
      </div>
    </div>
  </div>
</section>
"""
            )

            project_mode_state = gr.State(value=_project_mode_default())
            recommendation_state = gr.State(value={})
            quality_report_state = gr.State(value={})

            with gr.Row():
                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]) as why_group:
                    project_mode = gr.Radio(
                        PROJECT_MODES,
                        value=_project_mode_default(),
                        label="Workspace mode",
                        info=tip("project_mode"),
                    )
                with gr.Group(elem_classes=["studio-card"]) as explanation_group_top:
                    why_box = gr.Markdown(
                        _why_this_matters_markdown(_project_mode_default(), "image", "", "classification", {})
                    )

            with gr.Row(elem_classes=["studio-cockpit"]):
                with gr.Group(elem_classes=["studio-card"]):
                    workspace_snapshot = gr.Markdown(
                        "### Workspace cockpit\n\n- **Dataset**: Waiting\n- **Model**: Waiting\n- **Training**: Blocked"
                    )
                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                    next_step_box = gr.Markdown(
                        "### Next step\n\nUpload a dataset or paste a local path to get started."
                    )

            with gr.Row():
                with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                    with gr.Row():
                        tutorial_modality_dropdown = gr.Dropdown(
                            choices=_TUTORIAL_MODALITIES,
                            value="image",
                            label="Tutorial modality",
                            scale=1,
                            min_width=160,
                        )
                        load_tutorial_btn = gr.Button("✨ Load Tutorial", variant="secondary", scale=3)
                    tutorial_box = gr.Markdown(_tutorial_markdown("image"))

            with gr.Tabs(elem_classes=["studio-panel"], elem_id="studio-main-content"):

                # ── Tab 1: Data ───────────────────────────────────────────────────────
                with gr.Tab("🗂 Data", elem_classes=["studio-tab-panel"]):
                    gr.HTML(
                        """
<section class="studio-banner" aria-label="Step 1: Data setup">
  <div class="studio-banner__grid">
    <div>
      <div class="studio-banner__eyebrow">Step 1</div>
      <h2>Turn raw files into a confident starting point.</h2>
      <p>Upload a structured file or folder-based dataset and let the workspace detect the shape of the problem. The goal is to reduce manual setup so you can focus on reasoning about the data.</p>
    </div>
    <div class="studio-showcase">
      <div class="studio-showcase__card"><strong>Auto-detected schema</strong><span>Columns and likely label fields appear as selectors instead of requiring memorized names.</span></div>
      <div class="studio-showcase__card"><strong>Class balance preview</strong><span>Students can quickly see skew, missing fields, or modality mismatches before training.</span></div>
    </div>
  </div>
</section>
"""
                    )
                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"]) as model_selection_group:
                                gr.HTML(_section_intro("Dataset setup", "Point the workspace at your source data, then let the app infer the columns and prepare the right controls for the modality."))
                                modality = gr.Dropdown(MODALITIES, value="image", label="Modality",
                                                       info=tip("modality"))

                                with gr.Accordion("📂 Upload Dataset", open=True):
                                    gr.Markdown(
                                        "Upload a **ZIP** (image/audio/video class folders or a graph folder with `nodes.csv` + `edges.csv`) "
                                        "or **CSV/TSV/JSON** (tabular/text/timeseries). "
                                        "The path below will be filled automatically."
                                    )
                                    data_upload = gr.File(
                                        label="Drop file here",
                                        file_types=[".zip", ".csv", ".tsv", ".json"],
                                        file_count="single",
                                    )

                                data_path = gr.Textbox(
                                    label="Data path",
                                    placeholder="/path/to/dataset  or  /path/to/data.csv",
                                    info=tip("data_path"))

                            with gr.Group(elem_classes=["studio-card", "studio-card--compact"]) as column_mapping_group:
                                gr.HTML(_section_intro("Column mapping", "For structured data, selectors populate from detected fields so you can choose from the actual dataset schema."))
                                label_col   = gr.Dropdown(
                                    choices=[],
                                    value="label",
                                    label="Label column",
                                    info=tip("label_col"),
                                    allow_custom_value=True,
                                    visible=True,
                                )
                                text_col    = gr.Dropdown(
                                    choices=[],
                                    value="text",
                                    label="Input text column (X)",
                                    info=tip("text_col"),
                                    allow_custom_value=True,
                                    visible=False,
                                )
                                feature_cols = gr.Dropdown(
                                    choices=[],
                                    value=[],
                                    multiselect=True,
                                    label="Feature columns",
                                    info=tip("feature_cols"),
                                    visible=False,
                                )
                                time_col    = gr.Dropdown(
                                    choices=[],
                                    value="",
                                    label="Timestamp column (optional)",
                                    info=tip("time_col"),
                                    allow_custom_value=True,
                                    visible=False,
                                )
                                window_size = gr.Slider(10, 500, value=50,    step=10,
                                                        label="Window size (timeseries)", visible=False,
                                                        info=tip("window_size"))
                                n_frames    = gr.Slider(4,  32,  value=8,     step=4,
                                                        label="Frames to sample (video)", visible=False,
                                                        info=tip("n_frames"))
                                image_size  = gr.Slider(64, 384, value=160, step=16,
                                                        label="Image size", visible=False,
                                                        info=tip("image_size"))
                                image_size_hint = gr.Markdown(
                                    value="",
                                    visible=False,
                                )
                                sample_rate = gr.Slider(8000, 44100, value=22050, step=1000,
                                                        label="Sample rate Hz (audio)", visible=False,
                                                        info=tip("sample_rate"))
                                audio_image_size = gr.Slider(96, 384, value=224, step=32,
                                                             label="Spectrogram image size", visible=False,
                                                             info=tip("audio_image_size"))
                                audio_n_mels = gr.Slider(32, 256, value=64, step=16,
                                                         label="Mel bins", visible=False,
                                                         info=tip("audio_n_mels"))
                                augmentation = gr.Dropdown(AUG_LEVELS, value="light",
                                                           label="Augmentation level",
                                                           info=tip("augmentation"))

                            with gr.Group(elem_classes=["studio-card", "studio-card--compact"]) as split_cleaning_group:
                                with gr.Accordion("⚙️ Split, Subset & Cleaning", open=True):
                                    val_split = gr.Slider(0.1, 0.4, value=0.2, step=0.05,
                                                          label="Validation split",
                                                          info=tip("val_split"))
                                    use_random_subset = gr.Checkbox(
                                        value=False,
                                        label="Use random subset for quick experiments",
                                        info=tip("use_random_subset"),
                                        visible=False,
                                    )
                                    subset_percent = gr.Slider(
                                        0.1, 100.0, value=1.0, step=0.1,
                                        label="Subset percentage",
                                        info=tip("subset_percent"),
                                        visible=False,
                                    )
                                    subset_seed = gr.Number(
                                        value=42,
                                        precision=0,
                                        label="Subset seed",
                                        info=tip("subset_seed"),
                                        visible=False,
                                    )
                                    use_data_cleaning = gr.Checkbox(value=False,
                                                                     label="Auto-clean tabular data",
                                                                     info=tip("use_data_cleaning"),
                                                                     visible=False)
                                    tabular_missing_strategy = gr.Dropdown(
                                        ["auto", "median", "mean", "drop"],
                                        value="auto",
                                        label="Tabular missing-value strategy",
                                        info=tip("tabular_missing_strategy"),
                                        visible=False,
                                    )
                                    tabular_clip_outliers = gr.Checkbox(
                                        value=False,
                                        label="Clip extreme tabular values",
                                        info=tip("tabular_clip_outliers"),
                                        visible=False,
                                    )
                                    text_lowercase = gr.Checkbox(
                                        value=True,
                                        label="Lowercase text",
                                        info=tip("text_lowercase"),
                                        visible=False,
                                    )
                                    text_strip_urls = gr.Checkbox(
                                        value=True,
                                        label="Remove URLs from text",
                                        info=tip("text_strip_urls"),
                                        visible=False,
                                    )
                                    text_strip_punctuation = gr.Checkbox(
                                        value=False,
                                        label="Remove punctuation from text",
                                        info=tip("text_strip_punctuation"),
                                        visible=False,
                                    )
                                    text_remove_stopwords = gr.Checkbox(
                                        value=False,
                                        label="Remove common stop words",
                                        info=tip("text_remove_stopwords"),
                                        visible=False,
                                    )
                                    text_deduplicate = gr.Checkbox(
                                        value=False,
                                        label="Remove duplicate text rows",
                                        info=tip("text_deduplicate"),
                                        visible=False,
                                    )
                                    text_apply_stemming = gr.Checkbox(
                                        value=False,
                                        label="Apply stemming",
                                        info=tip("text_apply_stemming"),
                                        visible=False,
                                    )
                                    text_apply_lemmatization = gr.Checkbox(
                                        value=False,
                                        label="Apply lightweight lemmatization",
                                        info=tip("text_apply_lemmatization"),
                                        visible=False,
                                    )
                                    text_use_ngrams = gr.Checkbox(
                                        value=False,
                                        label="Use n-gram baseline features for classical text models",
                                        info=tip("text_use_ngrams"),
                                        visible=False,
                                    )
                                    timeseries_sort_by_time = gr.Checkbox(
                                        value=True,
                                        label="Sort rows by timestamp",
                                        info=tip("timeseries_sort_by_time"),
                                        visible=False,
                                    )
                                    timeseries_fill_strategy = gr.Dropdown(
                                        ["none", "forward_fill", "interpolate"],
                                        value="none",
                                        label="Time-series missing-value strategy",
                                        info=tip("timeseries_fill_strategy"),
                                        visible=False,
                                    )
                                    timeseries_forecast_mode = gr.Checkbox(
                                        value=False,
                                        label="Forecast future values",
                                        info=tip("timeseries_forecast_mode"),
                                        visible=False,
                                    )
                                    timeseries_forecast_horizon = gr.Slider(
                                        1, 24, value=1, step=1,
                                        label="Forecast horizon",
                                        info=tip("timeseries_forecast_horizon"),
                                        visible=False,
                                    )
                                    timeseries_lag_steps = gr.Slider(
                                        0, 12, value=0, step=1,
                                        label="Lag features",
                                        info=tip("timeseries_lag_steps"),
                                        visible=False,
                                    )
                                    timeseries_rolling_window = gr.Slider(
                                        0, 24, value=0, step=1,
                                        label="Rolling statistics window",
                                        info=tip("timeseries_rolling_window"),
                                        visible=False,
                                    )
                                    image_verify_files = gr.Checkbox(
                                        value=False,
                                        label="Skip unreadable image files",
                                        info=tip("image_verify_files"),
                                        visible=False,
                                    )
                                    image_aug_flip = gr.Checkbox(
                                        value=True,
                                        label="Image: horizontal flip",
                                        info=tip("image_aug_flip"),
                                        visible=False,
                                    )
                                    image_aug_vertical = gr.Checkbox(
                                        value=False,
                                        label="Image: vertical flip",
                                        info=tip("image_aug_vertical"),
                                        visible=False,
                                    )
                                    image_aug_rotation = gr.Checkbox(
                                        value=False,
                                        label="Image: rotation",
                                        info=tip("image_aug_rotation"),
                                        visible=False,
                                    )
                                    image_aug_color = gr.Checkbox(
                                        value=False,
                                        label="Image: color jitter",
                                        info=tip("image_aug_color"),
                                        visible=False,
                                    )
                                    image_aug_gray = gr.Checkbox(
                                        value=False,
                                        label="Image: grayscale",
                                        info=tip("image_aug_gray"),
                                        visible=False,
                                    )
                                    image_aug_perspective = gr.Checkbox(
                                        value=False,
                                        label="Image: perspective warp",
                                        info=tip("image_aug_perspective"),
                                        visible=False,
                                    )
                                    image_normalization = gr.Dropdown(
                                        ["imagenet", "simple_0_1", "none"],
                                        value="imagenet",
                                        label="Image normalization preset",
                                        info=tip("image_normalization"),
                                        visible=False,
                                    )
                                    image_force_grayscale = gr.Checkbox(
                                        value=False,
                                        label="Convert all images to grayscale before training",
                                        info=tip("image_force_grayscale"),
                                        visible=False,
                                    )
                                    audio_verify_files = gr.Checkbox(
                                        value=False,
                                        label="Skip unreadable audio files",
                                        info=tip("audio_verify_files"),
                                        visible=False,
                                    )
                                    audio_normalize_waveform = gr.Checkbox(
                                        value=True,
                                        label="Normalize audio loudness",
                                        info=tip("audio_normalize_waveform"),
                                        visible=False,
                                    )
                                    audio_aug_noise = gr.Checkbox(
                                        value=True,
                                        label="Audio: add noise",
                                        info=tip("audio_aug_noise"),
                                        visible=False,
                                    )
                                    audio_aug_shift = gr.Checkbox(
                                        value=False,
                                        label="Audio: time shift",
                                        info=tip("audio_aug_shift"),
                                        visible=False,
                                    )
                                    audio_aug_gain = gr.Checkbox(
                                        value=False,
                                        label="Audio: gain jitter",
                                        info=tip("audio_aug_gain"),
                                        visible=False,
                                    )
                                    audio_aug_time_mask = gr.Checkbox(
                                        value=False,
                                        label="Audio: time masking",
                                        info=tip("audio_aug_time_mask"),
                                        visible=False,
                                    )
                                    audio_aug_freq_mask = gr.Checkbox(
                                        value=False,
                                        label="Audio: frequency masking",
                                        info=tip("audio_aug_freq_mask"),
                                        visible=False,
                                    )
                                    video_verify_files = gr.Checkbox(
                                        value=False,
                                        label="Skip unreadable video files",
                                        info=tip("video_verify_files"),
                                        visible=False,
                                    )
                                    tabular_scaling = gr.Dropdown(
                                        ["standard", "minmax", "robust", "none"],
                                        value="standard",
                                        label="Tabular scaling strategy",
                                        info=tip("tabular_scaling"),
                                        visible=False,
                                    )

                            preview_btn = gr.Button("🔍 Preview Dataset", variant="secondary")

                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"]) as evaluation_report_group:
                                gr.HTML(_section_intro("Dataset intelligence", "Preview the upload, review validation feedback, and sanity-check class balance before you commit to training."))
                                stats_summary_box = gr.Textbox(label="Dataset summary", lines=10,
                                                                interactive=False)
                                dist_plot         = gr.Plot(label="Class distribution")
                                validation_box    = gr.Markdown(
                                    value="Review validation feedback and inferred columns here after previewing the dataset."
                                )
                                quality_report_box = gr.Markdown(
                                    value="The preflight Data Quality Report will appear here after preview."
                                )
                                quality_examples_box = gr.HTML(
                                    value=_cleaning_preview_empty_html(
                                        "Preview a dataset to inspect concrete before-and-after examples from the current cleaning choices."
                                    )
                                )

                # ── Tab 2: Model ──────────────────────────────────────────────────────
                with gr.Tab("🧠 Model", elem_classes=["studio-tab-panel"]):
                    gr.HTML(
                        """
<section class="studio-banner" aria-label="Step 2: Model selection">
  <div class="studio-banner__grid">
    <div>
      <div class="studio-banner__eyebrow">Step 2</div>
      <h2>Choose a model path that informs, not overwhelms.</h2>
      <p>Stay in a guided lane with sensible defaults, while still having room to explore pretrained models, deeper architectures, and training strategy choices when you want them.</p>
    </div>
    <div class="studio-badge-list">
      <span class="studio-badge">Fine-tune pretrained backbones</span>
      <span class="studio-badge">Train from scratch</span>
      <span class="studio-badge">Recommended defaults</span>
      <span class="studio-badge">Advanced knobs when needed</span>
    </div>
  </div>
</section>
"""
                    )
                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"]) as export_utilities_group:
                                gr.HTML(_section_intro("Architecture", "Choose how much of the model should be pretrained, select the task, and let the app guide the setup toward safe defaults."))
                                training_mode = gr.Radio(["fine-tune", "from_scratch"], value="fine-tune",
                                                         label="Training mode",
                                                         info=tip("training_mode"))
                                model_name    = gr.Dropdown([], label="Model architecture",
                                                            info=tip("model_name"))
                                task          = gr.Dropdown(TASKS, value="classification", label="Task",
                                                            info=tip("task"))
                                n_clusters    = gr.Slider(2, 32, value=8, step=1,
                                                          label="Number of clusters (clustering only)",
                                                          info=tip("n_clusters"),
                                                          visible=False)
                                use_class_weights = gr.Checkbox(
                                    value=False, label="Auto-balance class weights",
                                    info=tip("use_class_weights"))
                                recommend_btn       = gr.Button("✨ Get Guided Recommendations",
                                                                variant="secondary")
                                apply_recommendations_btn = gr.Button(
                                    "Apply recommendations",
                                    variant="primary"
                                )
                                recommendations_box = gr.Markdown("*Load a dataset first, then click above.*")
                                recommendation_rationale_box = gr.Markdown("*The rationale for the recommendation will appear here.*")

                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"]) as hyperparams_group:
                                gr.HTML(_section_intro("Hyperparameters", "Keep the primary settings visible and tuck specialist controls away until they’re actually relevant."))
                                lr         = gr.Slider(1e-5, 1e-1, value=1e-3, step=1e-5,
                                                       label="Learning rate",
                                                       info=tip("learning_rate"))
                                batch_size = gr.Slider(4, 64, value=16, step=4, label="Batch size",
                                                       info=tip("batch_size"))
                                epochs     = gr.Slider(1, 100, value=10, step=1, label="Epochs",
                                                       info=tip("epochs"))
                                dropout    = gr.Slider(0.0, 0.8, value=0.3, step=0.05, label="Dropout",
                                                       info=tip("dropout"))
                                optimizer  = gr.Dropdown(OPTIMIZERS, value="adam", label="Optimizer",
                                                         info=tip("optimizer"))
                                scheduler_name = gr.Dropdown(SCHEDULERS, value="cosine",
                                                             label="LR scheduler",
                                                             info=tip("scheduler"))
                                use_amp    = gr.Checkbox(value=False, label="Mixed precision (CUDA only)",
                                                         info=tip("use_amp"))

                                with gr.Accordion("Sequence model options", open=False) as sequence_options_group:
                                    hidden_size = gr.Slider(32, 512, value=128, step=32,
                                                            label="Hidden size",
                                                            info=tip("hidden_size"))
                                    num_layers  = gr.Slider(1, 4, value=1, step=1,
                                                            label="Num layers",
                                                            info=tip("num_layers"))

                                with gr.Accordion("Tree & gradient boosting options", open=False) as sklearn_options_group:
                                    gr.Markdown(XGBOOST_UNAVAILABLE_NOTE, visible=not xgboost_available())
                                    n_estimators = gr.Slider(10,  500, value=100, step=10,
                                                              label="n_estimators",
                                                              info=tip("n_estimators"))
                                    max_depth    = gr.Slider(0,   50,  value=0,   step=1,
                                                              label="Max depth (0 = unlimited)",
                                                              info=tip("max_depth"))
                                    C_param      = gr.Slider(0.01, 10, value=1.0, step=0.01,
                                                              label="C (LogisticRegression)",
                                                              info=tip("C_param"))
                                    max_iter     = gr.Slider(100, 2000, value=1000, step=100,
                                                              label="Max iter (LogReg)",
                                                              info=tip("max_iter"))
                                    lr_xgb       = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                                              label="Learning rate (gradient boosting)",
                                                              info=tip("lr_xgb"))

                # ── Tab 3: Train ──────────────────────────────────────────────────────
                with gr.Tab("🚀 Train", elem_classes=["studio-tab-panel"]):
                    gr.HTML(
                        """
<div class="studio-kpi-grid">
  <div class="studio-kpi"><strong>Launch point</strong><span>Run the core training path first</span></div>
  <div class="studio-kpi"><strong>Outputs</strong><span>Curves, summaries, exports, and bundle path</span></div>
  <div class="studio-kpi"><strong>Advanced mode</strong><span>Cross-validation and search after baseline success</span></div>
</div>
"""
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["studio-card"]):
                                gr.HTML(_section_intro("Run configuration", "Name the bundle, control checkpoint behavior, and start or stop the active training session from one place."))
                                bundle_name = gr.Textbox(value="my_model", label="Bundle name",
                                                          info=tip("bundle_name"))
                                checkpoint_every = gr.Slider(0, 20, value=0, step=1,
                                                              label="Save checkpoint every N epochs",
                                                              info=tip("checkpoint_every"))
                                train_btn   = gr.Button("▶ Train & Export", variant="primary")
                                stop_btn    = gr.Button("⏹ Stop", variant="stop")
                                gr.HTML("<p class='studio-inline-note'>Tip: start with a short run to verify your dataset and model pairing before longer experiments.</p>")

                            with gr.Group(elem_classes=["studio-card", "studio-card--compact", "studio-card--soft"]) as advanced_training_group:
                                gr.HTML(_section_intro("Advanced training", "Use these after the baseline workflow is stable. They reuse your Data and Model settings."))
                                with gr.Accordion("🔬 Cross-validation and search", open=False):
                                    with gr.Tabs():
                                        with gr.Tab("🔁 K-Fold Cross-Validation"):
                                            gr.Markdown(
                                                "Evaluate model stability across multiple train/val splits. "
                                                "Supports up to 10 folds and reports a richer metric suite for classification."
                                            )
                                            cv_k = gr.Slider(2, 10, value=10, step=1,
                                                              label="Number of folds (k)",
                                                              info=tip("cv_k"))
                                            cv_run_btn = gr.Button("▶ Run Cross-Validation",
                                                                   variant="secondary")
                                            cv_significance_models = gr.Dropdown(
                                                choices=[],
                                                value=[],
                                                multiselect=True,
                                                label="Compare against additional model families",
                                                info=tip("cv_significance_models"),
                                            )
                                            cv_significance_metric = gr.Dropdown(
                                                choices=["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_ovr", "mcc"],
                                                value="accuracy",
                                                label="Significance metric",
                                                info=tip("cv_significance_metric"),
                                            )
                                            cv_significance_btn = gr.Button(
                                                "🧪 Test Significance Between Families",
                                                variant="secondary",
                                            )
                                            cv_significance_box = gr.Markdown(
                                                "*Select one or more additional families and use 10-fold CV to compare them against the active model.*"
                                            )
                                            cv_significance_plot = gr.Plot(
                                                label="Fold-by-fold family comparison"
                                            )
                                            cv_results_box = gr.Textbox(
                                                label="Cross-validation results", lines=10,
                                                interactive=False,
                                                value="*Configure Data & Model tabs, then click Run.*")

                                        with gr.Tab("🎲 Random Hyperparameter Search"):
                                            gr.Markdown(
                                                "Try multiple learning rate and dropout combinations "
                                                "and report which performs best on the validation set."
                                            )
                                            hparam_lr_vals = gr.Textbox(
                                                value="1e-4, 1e-3, 1e-2",
                                                label="Learning rate candidates (comma-separated)",
                                                info=tip("hparam_lr_vals"))
                                            hparam_dropout_vals = gr.Textbox(
                                                value="0.1, 0.3, 0.5",
                                                label="Dropout candidates (comma-separated)",
                                                info=tip("hparam_dropout_vals"))
                                            hparam_n_trials     = gr.Slider(2, 20, value=6, step=1,
                                                                             label="Max trials (random subset of grid)",
                                                                             info=tip("hparam_n_trials"))
                                            hparam_epochs       = gr.Slider(1, 20, value=3, step=1,
                                                                             label="Epochs per trial",
                                                                             info=tip("hparam_epochs"))
                                            hparam_run_btn      = gr.Button("▶ Run Hyperparameter Search",
                                                                             variant="secondary")
                                            hparam_results_box  = gr.Textbox(
                                                label="Search results", lines=12,
                                                interactive=False,
                                                value="*Configure Data & Model tabs, then click Run.*")

                                        with gr.Tab("🧭 Model Sweep"):
                                            gr.Markdown(
                                                "Train multiple selected model families with the current Data and Hyperparameter settings, then rank them in one sortable table."
                                            )
                                            sweep_models = gr.Dropdown(
                                                choices=[],
                                                value=[],
                                                multiselect=True,
                                                label="Models to train and compare",
                                                info=tip("sweep_models"),
                                            )
                                            sweep_metrics = gr.Dropdown(
                                                choices=CLASSIFICATION_METRIC_OPTIONS,
                                                value=["accuracy", "f1_macro", "auc_ovr"],
                                                multiselect=True,
                                                label="Metrics to show in the table",
                                                info=tip("sweep_metrics"),
                                            )
                                            with gr.Row():
                                                sweep_sort_metric = gr.Dropdown(
                                                    choices=CLASSIFICATION_METRIC_OPTIONS,
                                                    value="accuracy",
                                                    label="Sort by metric",
                                                    info=tip("sweep_sort_metric"),
                                                )
                                                sweep_sort_order = gr.Dropdown(
                                                    choices=["descending", "ascending"],
                                                    value="descending",
                                                    label="Sort order",
                                                    info=tip("sweep_sort_order"),
                                                )
                                            sweep_run_btn = gr.Button("▶ Train Selected Models", variant="secondary")
                                            sweep_status_box = gr.Markdown(
                                                "*Choose models and metrics, then run the sweep to build a comparison table.*"
                                            )
                                            sweep_table = gr.DataFrame(
                                                label="Model comparison table",
                                                interactive=False,
                                                wrap=True,
                                                line_breaks=True,
                                                max_height=320,
                                            )

                        with gr.Column(scale=2):
                            with gr.Group(elem_classes=["studio-card"]):
                                gr.HTML(_section_intro("Live training telemetry", "Watch progress, inspect the training trace, and monitor the learning curves while the job runs."))
                                eta_box       = gr.Textbox(label="ETA", interactive=False, max_lines=1)
                                live_metrics_card = gr.HTML(
                                    value=_format_live_training_metrics_card()
                                )
                                status_box    = gr.Textbox(label="Training log", lines=16, interactive=False)
                                loss_plot     = gr.Plot(label="Loss & accuracy curves")

                            with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                gr.HTML(_section_intro("Run summary", "When training completes, this area consolidates the narrative summary and architecture overview into one executive view."))
                                training_summary_out = gr.Markdown(
                                    value="> ℹ️ Training summary will appear here once training is complete.")
                                arch_viz_out  = gr.Plot(label="Model architecture (parameter counts)")

                # ── Tab 4: Evaluate ───────────────────────────────────────────────────
                with gr.Tab("📊 Evaluate", elem_classes=["studio-tab-panel"]):
                    gr.HTML(
                        """
<div class="studio-kpi-grid">
  <div class="studio-kpi"><strong>Core report</strong><span>Overall metrics and narrative summary</span></div>
  <div class="studio-kpi"><strong>Diagnostics</strong><span>Task-specific plots for classification, regression, clustering, and anomaly runs</span></div>
  <div class="studio-kpi"><strong>Failure analysis</strong><span>Misclassifications, explanations, and saliency tools</span></div>
</div>
"""
                    )
                    with gr.Group(elem_classes=["studio-card"]) as history_group:
                        gr.HTML(_section_intro("Evaluation report", "High-level metrics land here first so you have a readable summary before digging into technical plots."))
                        eval_kpi_bar = gr.HTML(
                            value=_format_eval_kpis("", [], "classification")
                        )
                        eval_summary_box = gr.Markdown(
                            value="> ℹ️ Evaluation metrics will appear here after training completes."
                        )

                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"], visible=True) as classification_views_group:
                                gr.HTML(_section_intro("Classification views", "Use these to inspect class-level performance and threshold behavior."))
                                cm_plot  = gr.Plot(label="Confusion Matrix")
                                roc_plot = gr.Plot(label="ROC / AUC Curves")
                                classification_note = gr.Markdown(_DEFAULT_CLASSIFICATION_NOTE)
                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"], visible=True) as explanation_group:
                                gr.HTML(_section_intro("Explanations", "Surface why the model behaved the way it did across visual, audio, and tabular workflows."))
                                gradcam_out = gr.Plot(label="Grad-CAM heatmap (image / audio / video)")
                                shap_out    = gr.Plot(label="SHAP feature importance (tabular)")
                                text_shap_out = gr.HTML(
                                    value="<p style='color: var(--color-text-muted, #6a857c); font-style: italic;'>"
                                          "Token importance highlights will appear here after training a text model.</p>")
                                explanation_note = gr.Markdown(_DEFAULT_EXPLANATION_NOTE)

                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"], visible=False) as diagnostics_group:
                                gr.HTML(_section_intro("Prediction diagnostics", "Regression and anomaly plots help you read error shape, spread, and outlier behavior quickly."))
                                regression_plot_out = gr.Plot(label="Regression diagnostics (Predicted vs Actual · Residuals)")
                                anomaly_plot_out = gr.Plot(label="Anomaly detection — reconstruction errors")
                                diagnostics_note = gr.Markdown(_DEFAULT_DIAGNOSTICS_NOTE)
                        with gr.Column():
                            with gr.Group(elem_classes=["studio-card"], visible=True) as sample_review_group:
                                gr.HTML(_section_intro("Sample review", "Look at the model’s mistakes and cluster structure to understand where learning is breaking down."))
                                misclass_img = gr.Plot(label="Misclassified samples — visual")
                                misclass_df  = gr.DataFrame(
                                    label="Misclassified samples — table",
                                    wrap=True,
                                    line_breaks=True,
                                    max_height=320,
                                    column_widths=["80px", "120px", "140px", "140px", "140px", "160px", "140px"],
                                )
                                sample_review_note = gr.Markdown(_DEFAULT_SAMPLE_REVIEW_NOTE)
                            with gr.Group(elem_classes=["studio-card"], visible=False) as clustering_group:
                                gr.HTML(_section_intro("Clustering review", "Inspect the learned structure and separation between clusters for clustering workflows."))
                                tsne_plot_out   = gr.Plot(label="t-SNE cluster visualisation")
                                clustering_note = gr.Markdown(
                                    "*Clustering structure will appear here for clustering runs.*"
                                )

                # ── Tab 5: Dashboard ──────────────────────────────────────────────────
                with gr.Tab("📈 Dashboard", elem_classes=["studio-tab-panel"]):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(_section_intro(
                            "Executive Dashboard",
                            "Consolidated view of your training run. Key metrics, the most relevant diagnostic, explainability, and actionable recommendations — all in one place."))
                        dashboard_kpi = gr.HTML(
                            value="<p style='color: var(--color-text-muted, #6a857c); font-style: italic;'>"
                                  "Train a model to populate the dashboard.</p>")

                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Group(elem_classes=["studio-card"]):
                                gr.HTML(_section_intro("Primary Diagnostic", "The most relevant chart for your task type."))
                                dashboard_primary_chart = gr.Plot(label="Diagnostic chart")
                            with gr.Group(elem_classes=["studio-card"]):
                                gr.HTML(_section_intro("Explainability", "Why the model made its decisions."))
                                dashboard_explain_plot = gr.Plot(label="Explanation chart")
                                dashboard_explain_html = gr.HTML(value="")
                        with gr.Column(scale=2):
                            with gr.Group(elem_classes=["studio-card"]):
                                gr.HTML(_section_intro("Training Curves", "Loss and metric progression across epochs."))
                                dashboard_loss_chart = gr.Plot(label="Loss & accuracy")
                            with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                dashboard_actions = gr.Markdown(
                                    "### Recommendations\n\n*Train a model to see recommendations.*")
                            with gr.Group(elem_classes=["studio-card"]):
                                streamlit_btn = gr.Button("🚀 Export Streamlit Dashboard", variant="secondary")
                                streamlit_status = gr.Markdown("")

                # ── Tab 6: Try Your Model ─────────────────────────────────────────────
                with gr.Tab("🧪 Try Your Model", elem_classes=["studio-tab-panel"]):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(_section_intro("Interactive inference", "Load the latest bundle, try new examples, and inspect confidence scores without leaving the workspace."))
                        infer_bundle_path = gr.Dropdown(
                            choices=[],
                            label="Bundle path (select a saved bundle or paste a path)",
                            info=tip("infer_bundle_path"),
                            allow_custom_value=True,
                        )
                        infer_refresh_btn = gr.Button("🔄 Refresh bundles", variant="secondary", size="sm")

                    with gr.Tabs():
                        with gr.Tab("🖼 Image"):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card"]):
                                        gr.HTML(_section_intro("Image prediction", "Upload a single image and review the ranked class predictions."))
                                        infer_img_input = gr.Image(type="filepath", label="Upload image",
                                                                    sources=["upload", "clipboard"])
                                        infer_img_btn   = gr.Button("🔮 Predict", variant="primary")
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                        infer_img_md    = gr.Markdown("*Upload an image and click Predict.*")
                                        infer_img_chart = gr.Plot(label="Confidence scores")

                        with gr.Tab("✍️ Text"):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card"]):
                                        gr.HTML(_section_intro("Text prediction", "Paste a sample sentence or paragraph and run the trained classifier against it."))
                                        infer_text_input = gr.Textbox(label="Enter text", lines=4,
                                                                       placeholder="Type or paste text here…",
                                                                       info=tip("infer_text_input"))
                                        infer_text_btn   = gr.Button("🔮 Predict", variant="primary")
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                        infer_text_md    = gr.Markdown("*Enter text and click Predict.*")
                                        infer_text_chart = gr.Plot(label="Confidence scores")
                                        infer_text_explain = gr.HTML(value="")

                        with gr.Tab("📊 Tabular"):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card"]):
                                        gr.HTML(_section_intro("Tabular prediction", "Provide feature values as JSON so the model can score a single record."))
                                        gr.Markdown(
                                            "Enter feature values as JSON, e.g. "
                                            "`{\"age\": 25, \"income\": 50000, \"score\": 0.8}`"
                                        )
                                        infer_tab_fill_btn = gr.Button("✨ Use detected sample", variant="secondary")
                                        infer_tab_input = gr.Textbox(label="Feature values (JSON)", lines=5,
                                                                      placeholder='{"feature1": value, …}',
                                                                      info=tip("infer_tab_input"))
                                        infer_tab_btn   = gr.Button("🔮 Predict", variant="primary")
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                        infer_tab_md    = gr.Markdown("*Enter features and click Predict.*")
                                        infer_tab_chart = gr.Plot(label="Confidence scores")

                        with gr.Tab("📈 Time-series"):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card"]):
                                        gr.HTML(_section_intro("Time-series prediction", "Provide a short JSON window of sequential feature rows and score it with the current bundle."))
                                        gr.Markdown(
                                            "Enter a JSON array like "
                                            "`[{\"sensor_a\": 0.12, \"sensor_b\": 1.4}, {\"sensor_a\": 0.18, \"sensor_b\": 1.3}]`"
                                        )
                                        infer_ts_fill_btn = gr.Button("✨ Use detected window sample", variant="secondary")
                                        infer_ts_input = gr.Textbox(label="Time-series window (JSON)", lines=7,
                                                                    placeholder='[{"feature1": value, "feature2": value}, …]',
                                                                    info=tip("infer_ts_input"))
                                        infer_ts_btn = gr.Button("🔮 Predict", variant="primary")
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                        infer_ts_md = gr.Markdown("*Enter a window and click Predict.*")
                                        infer_ts_chart = gr.Plot(label="Prediction summary")

                        with gr.Tab("🔊 Audio"):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card"]):
                                        gr.HTML(_section_intro("Audio prediction", "Upload a single clip to test the current bundle on new audio input."))
                                        infer_aud_input = gr.Audio(type="filepath", label="Upload audio")
                                        infer_aud_btn   = gr.Button("🔮 Predict", variant="primary")
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                        infer_aud_md    = gr.Markdown("*Upload audio and click Predict.*")
                                        infer_aud_chart = gr.Plot(label="Confidence scores")

                        with gr.Tab("📁 Batch"):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card"]):
                                        gr.HTML(_section_intro(
                                            "Batch prediction",
                                            "Point at a folder of files — the app detects the modality from the "
                                            "loaded bundle and scores every matching file, then lets you download the results as CSV.",
                                        ))
                                        batch_folder = gr.Textbox(
                                            label="Input folder path",
                                            placeholder="/path/to/folder of images (or audio, text, video…)",
                                            info="Paste an absolute path to a folder containing files to classify.",
                                        )
                                        batch_run_btn = gr.Button("🔮 Run Batch Prediction", variant="primary")
                                with gr.Column():
                                    with gr.Group(elem_classes=["studio-card", "studio-card--soft"]):
                                        batch_status = gr.Markdown("*Set a bundle and folder path, then click Run.*")
                                        batch_df = gr.Dataframe(
                                            headers=["file", "prediction", "confidence"],
                                            datatype=["str", "str", "str"],
                                            row_count=1,
                                            column_count=3,
                                            interactive=False,
                                            wrap=True,
                                            label="Predictions",
                                        )
                                        batch_save_btn = gr.Button("⬇️ Save CSV to outputs/", variant="secondary")
                                        batch_save_status = gr.Markdown("")

                # ── Tab 6: Export & History ───────────────────────────────────────────
                with gr.Tab("📦 Export & History", elem_classes=["studio-tab-panel"]):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(_section_intro("Bundle outputs", "Generate deployment artifacts, inspect the saved bundle path, and keep track of previous runs from one place."))
                        bundle_out = gr.Textbox(label="Saved bundle path", interactive=False)
                        with gr.Row():
                            fastapi_btn    = gr.Button("⚡ Generate FastAPI Server", variant="secondary")
                            model_card_btn = gr.Button("📄 Generate Model Card",    variant="secondary")
                            docker_btn     = gr.Button("🐳 Generate Docker Bundle", variant="secondary")
                        with gr.Row():
                            fastapi_status_box    = gr.Markdown("*FastAPI export status will appear here.*")
                            model_card_status_box = gr.Markdown("*Model card generation status will appear here.*")
                            docker_status_box     = gr.Markdown("*Docker bundle status will appear here.*")

                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(_section_intro("Run history", "Review earlier training runs and compare selected experiments side by side."))
                        refresh_btn  = gr.Button("🔄 Refresh History")
                        history_df   = gr.Dataframe(
                            headers=HISTORY_COLUMNS,
                            datatype=["number", "str", "str", "str", "str", "str", "number", "number", "str"],
                            row_count=1,
                            column_count=len(HISTORY_COLUMNS),
                            interactive=False,
                            wrap=True,
                            line_breaks=True,
                            max_height=320,
                            column_widths=["70px", "150px", "120px", "110px", "130px", "120px", "90px", "90px", "360px"],
                            label="All training runs",
                        )
                        compare_ids  = gr.Textbox(value="",
                                                   label="Compare run IDs (comma-separated, e.g. 0,1,2)",
                                                   info=tip("compare_ids"))
                        compare_btn  = gr.Button("📈 Compare Selected Runs")
                        compare_status_box = gr.Markdown("*Select one or more run IDs to generate a comparison plot.*")
                        compare_plot = gr.Plot(label="Run comparison")

                # ── Tab 7: Object Detection ───────────────────────────────────────────
                with gr.Tab("🎯 Object Detection"):
                    gr.Markdown(
                        """
    ### YOLO Object Detection
    Upload an **image** or **video** to detect objects using a pretrained YOLOv8 model,
    or **train a custom YOLO classifier** on your own image dataset.
    No extra setup — weights are downloaded automatically on first use.
                    """
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            det_model = gr.Dropdown(
                                choices=list(YOLO_MODELS.keys()),
                                value=list(YOLO_MODELS.keys())[0],
                                label="Model",
                                info=tip("det_model"),
                            )
                            det_conf  = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                                  label="Confidence threshold",
                                                  info=tip("det_conf"))
                            det_iou   = gr.Slider(0.05, 0.95, value=0.45, step=0.05,
                                                  label="IoU threshold (NMS)",
                                                  info=tip("det_iou"))

                    with gr.Tabs():

                        # ── Inference: Image ──────────────────────────────────────────
                        with gr.Tab("🖼 Image"):
                            with gr.Row():
                                with gr.Column():
                                    det_img_input  = gr.Image(type="filepath",
                                                               label="Upload image",
                                                               sources=["upload", "clipboard"])
                                    det_img_btn    = gr.Button("🔍 Detect Objects", variant="primary")
                                with gr.Column():
                                    det_img_output = gr.Image(label="Detected objects",
                                                               interactive=False)
                            det_img_stats_md = gr.Markdown("*Upload an image and click Detect.*")
                            det_img_chart    = gr.Plot(label="Detection breakdown")

                        # ── Inference: Video ──────────────────────────────────────────
                        with gr.Tab("🎬 Video"):
                            with gr.Row():
                                with gr.Column():
                                    det_vid_input  = gr.Video(label="Upload video",
                                                               sources=["upload"],
                                                               format="mp4")
                                    det_vid_btn    = gr.Button("▶ Detect Objects in Video",
                                                                variant="primary")
                                with gr.Column():
                                    det_vid_output = gr.Video(label="Annotated video",
                                                               interactive=False)
                            det_vid_status   = gr.Textbox(label="Progress", lines=3,
                                                           interactive=False)
                            det_vid_stats_md = gr.Markdown("*Upload a video and click Detect.*")
                            det_vid_chart    = gr.Plot(label="Detection breakdown")

                        # ── Custom Training ───────────────────────────────────────────
                        with gr.Tab("🏋 Train Custom Classifier"):
                            gr.Markdown(
                                """
    Train YOLOv8 as an **image classifier** on your own classes.
    Organise your images as `dataset_folder/class_name/image.jpg` (same format as the Data tab),
    then point to the root folder below.
                            """
                            )
                            with gr.Row():
                                with gr.Column():
                                    yolo_data_path = gr.Textbox(
                                        label="Dataset folder",
                                        placeholder="/path/to/image_class_folders",
                                        info=tip("yolo_data_dir"))
                                    yolo_model_size = gr.Dropdown(
                                        choices=["YOLOv8 Nano", "YOLOv8 Small", "YOLOv8 Medium"],
                                        value="YOLOv8 Nano",
                                        label="Model size",
                                        info=tip("yolo_model_size"))
                                    yolo_epochs   = gr.Slider(1, 100, value=20, step=1,
                                                               label="Epochs",
                                                               info=tip("yolo_epochs"))
                                    yolo_batch    = gr.Slider(4, 64, value=16, step=4,
                                                               label="Batch size",
                                                               info=tip("yolo_batch"))
                                    yolo_train_btn = gr.Button("▶ Train", variant="primary")
                                with gr.Column():
                                    yolo_status_box = gr.Textbox(label="Training log", lines=12,
                                                                  interactive=False)
                            yolo_result_md = gr.Markdown("*Start training to see results.*")

                # ── Tab 8: Annotation Studio ────────────────────────────────────────
                with gr.Tab("🏷️ Annotation Studio", elem_classes=["studio-tab-panel"]):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(_section_intro(
                            "Annotation and relabeling",
                            "Review concrete samples, correct labels in-place, and turn object detections into saved annotation files without leaving the app."
                        ))

                    with gr.Tabs():
                        with gr.Tab("🖼 Image relabel"):
                            with gr.Row():
                                with gr.Column():
                                    annotation_image_dir = gr.Textbox(
                                        label="Image dataset folder",
                                        placeholder="/path/to/image_class_folders",
                                        info=tip("annotation_image_dir"),
                                    )
                                    annotation_image_refresh = gr.Button("🔄 Load image samples", variant="secondary")
                                    annotation_image_sample = gr.Dropdown(
                                        choices=[],
                                        label="Choose image sample",
                                        info=tip("annotation_image_sample"),
                                        allow_custom_value=True,
                                    )
                                    annotation_image_new_label = gr.Dropdown(
                                        choices=[],
                                        label="New label",
                                        info=tip("annotation_image_new_label"),
                                        allow_custom_value=True,
                                    )
                                    annotation_image_save = gr.Button("💾 Save relabel", variant="primary")
                                with gr.Column():
                                    annotation_image_preview = gr.Image(
                                        type="filepath",
                                        label="Image preview",
                                        interactive=False,
                                    )
                                    annotation_image_meta = gr.Markdown("*Load image samples to begin.*")
                                    annotation_image_status = gr.Markdown("")

                        with gr.Tab("✍️ Text relabel"):
                            with gr.Row():
                                with gr.Column():
                                    annotation_text_path = gr.Textbox(
                                        label="Text dataset path",
                                        placeholder="/path/to/text.csv",
                                        info=tip("annotation_text_path"),
                                    )
                                    annotation_text_col = gr.Textbox(
                                        label="Text column",
                                        value="text",
                                    )
                                    annotation_text_label_col = gr.Textbox(
                                        label="Label column",
                                        value="label",
                                    )
                                    annotation_text_refresh = gr.Button("🔄 Load text rows", variant="secondary")
                                    annotation_text_sample = gr.Dropdown(
                                        choices=[],
                                        label="Choose text row",
                                        info=tip("annotation_text_sample"),
                                    )
                                    annotation_text_new_label = gr.Dropdown(
                                        choices=[],
                                        label="New label",
                                        info=tip("annotation_text_new_label"),
                                        allow_custom_value=True,
                                    )
                                    annotation_text_save = gr.Button("💾 Save relabel", variant="primary")
                                with gr.Column():
                                    annotation_text_preview = gr.Markdown("*Load text rows to begin.*")
                                    annotation_text_status = gr.Markdown("")

                        with gr.Tab("🔊 Audio relabel"):
                            with gr.Row():
                                with gr.Column():
                                    annotation_audio_dir = gr.Textbox(
                                        label="Audio dataset folder",
                                        placeholder="/path/to/audio_class_folders",
                                        info=tip("annotation_audio_dir"),
                                    )
                                    annotation_audio_refresh = gr.Button("🔄 Load audio samples", variant="secondary")
                                    annotation_audio_sample = gr.Dropdown(
                                        choices=[],
                                        label="Choose audio sample",
                                        info=tip("annotation_audio_sample"),
                                        allow_custom_value=True,
                                    )
                                    annotation_audio_new_label = gr.Dropdown(
                                        choices=[],
                                        label="New label",
                                        info=tip("annotation_audio_new_label"),
                                        allow_custom_value=True,
                                    )
                                    annotation_audio_save = gr.Button("💾 Save relabel", variant="primary")
                                with gr.Column():
                                    annotation_audio_preview = gr.Audio(
                                        type="filepath",
                                        label="Audio preview",
                                        interactive=False,
                                    )
                                    annotation_audio_meta = gr.Markdown("*Load audio samples to begin.*")
                                    annotation_audio_status = gr.Markdown("")

                        with gr.Tab("📦 Object box review"):
                            with gr.Row():
                                with gr.Column():
                                    annotation_det_model = gr.Dropdown(
                                        choices=list(YOLO_MODELS.keys()),
                                        value=list(YOLO_MODELS.keys())[0],
                                        label="Detection model",
                                        info=tip("det_model"),
                                    )
                                    annotation_det_conf = gr.Slider(
                                        0.05, 0.95, value=0.25, step=0.05,
                                        label="Confidence threshold",
                                        info=tip("det_conf"),
                                    )
                                    annotation_det_iou = gr.Slider(
                                        0.05, 0.95, value=0.45, step=0.05,
                                        label="IoU threshold",
                                        info=tip("det_iou"),
                                    )
                                    annotation_det_image = gr.Image(
                                        type="filepath",
                                        label="Detection review image",
                                        sources=["upload", "clipboard"],
                                        interactive=True,
                                    )
                                    annotation_det_review_btn = gr.Button("🔍 Review boxes", variant="primary")
                                    annotation_det_save_btn = gr.Button("💾 Save YOLO review", variant="secondary")
                                with gr.Column():
                                    annotation_det_output = gr.Image(
                                        label="Detected boxes",
                                        interactive=False,
                                    )
                                    annotation_det_status = gr.Markdown("*Upload an image and review the proposed boxes.*")
                            annotation_det_table = gr.Dataframe(
                                headers=["keep", "label", "confidence", "x1", "y1", "x2", "y2"],
                                datatype=["bool", "str", "number", "number", "number", "number", "number"],
                                row_count=1,
                                column_count=7,
                                interactive=True,
                                label="Editable detection review",
                                wrap=True,
                                line_breaks=True,
                            )

    # Every component above is a local variable; hand them over by name.
    local_names = locals()
    return demo, Components(
        (name, value) for name, value in local_names.items()
        if isinstance(value, gr.blocks.Block) and value is not demo
    )
