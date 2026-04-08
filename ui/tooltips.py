"""
Human-readable explanations for every hyperparameter and UI control.
Used to populate help text (?) next to each Gradio component.
"""

TOOLTIPS: dict[str, str] = {
    # Data
    "modality":
        "The type of data you're working with. Each modality uses a specialised pipeline.",
    "project_mode":
        "Beginner hides most advanced controls, Guided keeps the main workflow visible with curated options, and Advanced exposes the full workspace.",
    "data_upload":
        "Upload a ZIP or structured file to populate the workspace quickly. Use this when you do not want to paste a local path manually.",
    "data_path":
        "Path to your dataset folder (for images/audio/video: subfolders = classes) "
        "or CSV file (for text/tabular/timeseries).",
    "label_col":
        "The CSV column name that contains your class labels.",
    "text_col":
        "The CSV column name that contains the raw text to classify.",
    "feature_cols":
        "Choose which input columns should be used as X features. Any columns you leave out will be ignored during training.",
    "time_col":
        "Optional: CSV column with timestamps. Rows will be sorted by this column before windowing.",
    "window_size":
        "How many consecutive timesteps form one input sample. "
        "Larger = more context, but fewer samples and slower training.",
    "n_frames":
        "Number of evenly-spaced frames to sample from each video. "
        "More frames = richer temporal signal but higher memory use.",
    "image_size":
        "Target size used before the image reaches the model. Smaller sizes train faster and use less memory; larger sizes preserve detail but cost more.",
    "sample_rate":
        "Target audio sample rate in Hz. Files are resampled if they differ. "
        "22050 Hz (CD quality) is a good default for most tasks.",
    "audio_image_size":
        "Final width and height of the spectrogram image created from audio. Smaller sizes are faster; larger sizes preserve more time-frequency detail.",
    "audio_n_mels":
        "Number of mel-frequency bands in the spectrogram. Higher values capture finer detail but increase compute.",
    "val_split":
        "Fraction of data reserved for validation. A larger split gives more trustworthy evaluation, but leaves less data for training.",
    "use_random_subset":
        "Use only a random slice of the structured dataset for quick experiments. This is helpful when the full CSV is too large for fast iteration.",
    "subset_percent":
        "Percentage of the cleaned structured dataset to sample. For example, 1 means the app will use about 1% of the rows.",
    "subset_seed":
        "Random seed used when picking the subset. Keep the same seed when you want the same sample again.",
    "use_data_cleaning":
        "Automatically clean structured tabular data before training. This is useful when you want a safer baseline without writing preprocessing code yourself.",
    "tabular_missing_strategy":
        "How missing numeric values are handled. Median is usually a strong default because it is robust to outliers.",
    "tabular_clip_outliers":
        "Caps unusually large or small numeric values so a few extreme rows do not dominate training.",
    "text_lowercase":
        "Converts text to lowercase so the model treats 'Cat' and 'cat' as the same token when appropriate.",
    "text_strip_urls":
        "Removes web links that often add noise instead of useful meaning in text classification tasks.",
    "text_strip_punctuation":
        "Removes punctuation marks for simpler bag-of-words style baselines. Keep punctuation when it may carry meaning.",
    "text_remove_stopwords":
        "Removes very common filler words such as 'the' and 'and'. This can simplify some baselines, but may remove useful nuance in other tasks.",
    "text_deduplicate":
        "Drops repeated text rows before training so copied examples do not inflate the model's apparent confidence.",
    "text_apply_stemming":
        "Reduces words to shorter stems such as 'running' → 'run'. This can help simpler baselines treat related words similarly.",
    "text_apply_lemmatization":
        "Applies lightweight word simplification so plural and inflected forms map closer together.",
    "text_use_ngrams":
        "Marks the text preprocessing as n-gram aware for classical-style text baselines. Use this when short phrases matter more than single tokens alone.",
    "timeseries_sort_by_time":
        "Sort the rows before windowing so each training sample follows the real time order of the series.",
    "timeseries_fill_strategy":
        "Choose how gaps in numeric time-series values should be handled before windows are created.",
    "image_verify_files":
        "Checks that image files can actually be opened before training. This avoids crashes from a few broken files at the cost of a slower startup scan.",
    "image_aug_flip":
        "Randomly flips images left-to-right during training. Helpful when direction does not change the label.",
    "image_aug_vertical":
        "Randomly flips images top-to-bottom. Use only when upside-down samples should still belong to the same class.",
    "image_aug_rotation":
        "Adds random rotations so the model becomes less sensitive to orientation.",
    "image_aug_color":
        "Adds random brightness and color changes so the model relies less on lighting conditions.",
    "image_aug_gray":
        "Occasionally converts images to grayscale. Useful when shape matters more than color.",
    "image_aug_perspective":
        "Applies mild geometric warping so the model is more robust to viewpoint changes.",
    "image_normalization":
        "Controls how pixel values are normalised before training. ImageNet is best for pretrained vision backbones, while simpler presets are useful for scratch baselines.",
    "image_force_grayscale":
        "Converts every image to grayscale before training while still feeding a 3-channel tensor into the model. Use this when shape matters more than colour.",
    "audio_verify_files":
        "Checks audio files before training so unreadable clips can be skipped instead of causing failures later.",
    "audio_normalize_waveform":
        "Scales audio to a more consistent loudness before spectrogram conversion, which can make training more stable.",
    "audio_aug_noise":
        "Adds light background noise during training so the model becomes more robust to imperfect recordings.",
    "audio_aug_shift":
        "Shifts audio slightly in time so the model does not depend on sounds starting at exactly the same moment.",
    "audio_aug_gain":
        "Randomly changes loudness during training to reduce sensitivity to recording volume.",
    "audio_aug_time_mask":
        "Masks short spans of time in the spectrogram so the model learns to rely on broader patterns instead of one precise moment.",
    "audio_aug_freq_mask":
        "Masks frequency bands in the spectrogram so the model becomes less sensitive to narrow-band noise or missing frequencies.",
    "video_verify_files":
        "Checks that video files can be opened before training, which helps catch broken files earlier.",
    "tabular_scaling":
        "Choose how numeric columns are scaled. Standard scaling is a strong default, robust scaling helps when outliers are common, and none leaves values untouched.",

    # Model
    "training_mode":
        "Fine-tune: start from a model pre-trained on millions of samples — faster and needs less data.\n"
        "From scratch: train all weights from random initialisation — needs more data and epochs.",
    "model_name":
        "The neural network architecture to use. Options depend on your modality and training mode.",
    "task":
        "Classification: predict which class a sample belongs to.\n"
        "Clustering: group similar samples without labels (uses learned embeddings + KMeans).\n"
        "Regression: predict a continuous numeric value.",
    "n_clusters":
        "How many groups the clustering workflow should try to discover. Use this when you already have a rough expectation of the number of natural groups.",
    "use_class_weights":
        "Makes rare classes matter more during training so the model does not focus only on the majority class.",

    # Hyperparameters
    "learning_rate":
        "How fast the model updates its weights. Too high → unstable training. Too low → slow convergence. "
        "1e-3 is a safe default; use 2e-5 for fine-tuning transformers.",
    "batch_size":
        "Number of samples processed together in one forward/backward pass. "
        "Larger = faster but more memory. Halve this if you see out-of-memory errors.",
    "epochs":
        "Maximum number of complete passes through the training data. "
        "Early stopping will often terminate training before this limit.",
    "dropout":
        "Fraction of neurons randomly disabled during training to prevent overfitting. "
        "0.3 is a good default; increase if the model overfits (val_loss rising while train_loss falls).",
    "optimizer":
        "Algorithm that updates model weights.\n"
        "Adam: adaptive, good default.\n"
        "AdamW: Adam with weight decay — better for transformers.\n"
        "SGD: classic momentum SGD — can generalise better with careful tuning.",
    "scheduler":
        "Learning rate schedule applied during training.\n"
        "Cosine: smoothly decays LR to 0 — recommended for most tasks.\n"
        "Step: halves LR every few epochs — good for longer runs.\n"
        "Warmup+Cosine: ramps up first, then cosine decay — best for transformers.\n"
        "None: constant LR throughout.",
    "use_amp":
        "Mixed precision (fp16) training. Only activates on NVIDIA GPUs. "
        "Speeds up training by ~2× with no accuracy loss.",

    # RNN
    "hidden_size":
        "Number of units in each RNN hidden layer. Larger = more capacity but slower. "
        "128 is a good starting point for most text/timeseries tasks.",
    "num_layers":
        "Number of stacked RNN layers. Deeper = more expressive but more prone to overfitting on small datasets. "
        "Start with 1.",

    # sklearn
    "n_estimators":
        "Number of trees in the forest (RandomForest) or boosting rounds (XGBoost). "
        "More trees → better accuracy but slower training.",
    "max_depth":
        "Maximum depth of each tree. 0 = unlimited. "
        "Shallower trees reduce overfitting on small datasets.",
    "C_param":
        "Regularisation strength for LogisticRegression. Smaller C = stronger regularisation (less overfitting).",
    "max_iter":
        "Maximum iterations for LogisticRegression solver convergence.",
    "lr_xgb":
        "Step size used by XGBoost. Lower values are often more stable, while higher values learn faster but can overshoot.",

    # Augmentation
    "augmentation":
        "Data augmentation applies random transformations during training to artificially increase diversity.\n"
        "None: no augmentation (fast, use when dataset is large).\n"
        "Light: gentle flips/noise — safe default.\n"
        "Medium: colour jitter, rotation, noise — good for most cases.\n"
        "Heavy: aggressive transforms — use only if the model overfits badly.",

    # Export
    "bundle_name":
        "Name for the saved model bundle folder. A timestamp is appended automatically.",
    "checkpoint_every":
        "How often to save intermediate checkpoints during training. Use this when you want recovery points for long runs, but remember it uses more disk space.",

    # Cross-validation and experiments
    "cv_k":
        "Number of folds for cross-validation. More folds give a more stable estimate, but take longer because the model is retrained many times.",
    "cv_significance_models":
        "Additional model families to compare against the current one using the same folds. This helps answer whether a better score is likely meaningful or just noise.",
    "cv_significance_metric":
        "Metric used when comparing model families statistically. Choose the score that matters most for your problem.",
    "hparam_lr_vals":
        "Candidate learning rates to try during the random search. Include a small spread so the app can test both conservative and aggressive settings.",
    "hparam_dropout_vals":
        "Candidate dropout values to try during random search. This explores different regularisation strengths.",
    "hparam_n_trials":
        "Maximum number of hyperparameter combinations to test. More trials search more broadly but take longer.",
    "hparam_epochs":
        "Training epochs used for each hyperparameter trial. Keep this short so the search stays fast.",
    "sweep_models":
        "Choose several model families to train with the same data and settings so you can compare them side by side.",
    "sweep_metrics":
        "Select which scores should appear in the comparison table. Pick the metrics that matter for your task, not just accuracy by default.",
    "sweep_sort_metric":
        "Metric used to rank the compared models in the results table.",
    "sweep_sort_order":
        "Choose descending when higher is better, or ascending for metrics where lower is better such as MAE or RMSE.",

    # Inference and history
    "infer_bundle_path":
        "Path to a saved model bundle. This tells the app which trained model and preprocessing setup to use for prediction.",
    "infer_img_input":
        "Upload one image to test how the saved model behaves on a new example.",
    "infer_text_input":
        "Enter raw text exactly the way a real user might provide it so you can test the saved model realistically.",
    "infer_tab_input":
        "Provide one tabular example as JSON with feature names and values. This lets you test a structured prediction without writing code.",
    "infer_ts_input":
        "Provide one time-series window as JSON so the saved model can score a sequence-shaped example.",
    "infer_aud_input":
        "Upload one audio clip to see how the saved audio model responds to a fresh sample.",
    "compare_ids":
        "Enter run IDs from the history table to compare training curves and outcomes between experiments.",

    # Detection
    "det_model":
        "Pretrained YOLO model used for object detection. Larger models are usually more accurate but slower.",
    "det_conf":
        "Minimum confidence required before a detected box is kept. Raise this to reduce weak detections; lower it to catch more candidates.",
    "det_iou":
        "IoU threshold for non-maximum suppression. Higher values keep more overlapping boxes; lower values merge aggressive duplicates.",
    "det_img_input":
        "Upload one image for object detection with the selected pretrained YOLO model.",
    "det_vid_input":
        "Upload one video for frame-by-frame object detection. Longer videos will take more time to process.",

    # YOLO training
    "yolo_data_dir":
        "Path to a folder-organised image dataset for YOLO image classification training.",
    "yolo_model_size":
        "YOLO classifier size to train. Smaller variants are faster; larger ones may be more accurate.",
    "yolo_epochs":
        "How many passes YOLO training should make over the dataset.",
    "yolo_batch":
        "Batch size used during YOLO training. Increase it when you have memory headroom; reduce it if the run runs out of memory.",
}

GUIDANCE: dict[str, str] = {
    "small_dataset":
        "Your dataset is small (< 200 samples). Recommendations:\n"
        "• Use Fine-tune mode (much faster convergence)\n"
        "• Keep epochs ≤ 10 with early stopping\n"
        "• Use Medium or Heavy augmentation\n"
        "• Use a smaller model (MobileNetV3-Small, DistilBERT)",
    "large_dataset":
        "Your dataset is large (> 5000 samples). Recommendations:\n"
        "• From-scratch training is viable\n"
        "• Increase epochs to 20–50\n"
        "• Enable mixed precision if you have a GPU\n"
        "• Use Cosine or Warmup+Cosine scheduler",
    "imbalanced":
        "Class imbalance detected. Recommendations:\n"
        "• Enable weighted loss (coming soon)\n"
        "• Use augmentation to oversample minority classes\n"
        "• Monitor per-class F1 in the Evaluation tab, not just accuracy",
}
