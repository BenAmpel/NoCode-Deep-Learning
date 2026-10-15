"""Stage 1: load the dataset into loaders and arrays."""
from __future__ import annotations

import tempfile
from typing import Iterator

from config import TEXT_DEFAULTS, VIDEO_DEFAULTS
from pipeline.context import RunContext
from pipeline.progress import Update


def load_data(ctx: RunContext) -> Iterator[Update]:
    cfg, log = ctx.cfg, ctx.log
    data_path = ctx.data_path
    extra_data = ctx.extra_data
    train_loader = val_loader = None
    classes: list = []
    prep: dict = {}

    log.set_phase("loading data", eta_text="ETA: loading data")
    ctx.progress(0.05, desc="Loading data…")
    log.log(f"Loading {cfg.modality} data…")
    if cfg.modality in {"graph", "tabular", "text", "timeseries"} and bool(cfg.use_random_subset) and float(cfg.subset_percent) < 100:
        log.log(
            f"Using a random subset of {float(cfg.subset_percent):.2f}% of the dataset "
            f"(seed {int(cfg.subset_seed)})."
        )
    yield Update()

    bs = int(cfg.batch_size)

    _val_split = float(cfg.val_split)

    if cfg.modality == "graph":
        from modalities.graph import load_graph_data
        classes, prep, graph_data = load_graph_data(
            data_path,
            label_col=cfg.label_col,
            feature_cols=cfg.feature_cols,
            val_split=_val_split,
            subset_percent=float(cfg.subset_percent) if bool(cfg.use_random_subset) else 100.0,
            subset_seed=int(cfg.subset_seed),
        )
        extra_data.update(graph_data)

    elif cfg.modality == "image":
        from modalities.image import load_image_data
        image_aug_options = {
            "horizontal_flip": bool(cfg.image_aug_flip),
            "vertical_flip": bool(cfg.image_aug_vertical),
            "rotation": bool(cfg.image_aug_rotation),
            "color_jitter": bool(cfg.image_aug_color),
            "grayscale": bool(cfg.image_aug_gray),
            "perspective": bool(cfg.image_aug_perspective),
        }
        train_loader, val_loader, classes, prep, val_samples = load_image_data(
            data_path, mode=cfg.training_mode, batch_size=bs,
            val_split=_val_split, augmentation=cfg.augmentation,
            verify_files=bool(cfg.image_verify_files),
            image_size=int(cfg.image_size),
            augmentation_options=image_aug_options,
            normalization_preset=str(cfg.image_normalization),
            force_grayscale=bool(cfg.image_force_grayscale),
        )
        extra_data["val_samples"] = val_samples

    elif cfg.modality == "text":
        from modalities.text import load_text_data
        text_cleaning = {
            "lowercase": bool(cfg.text_lowercase),
            "strip_urls": bool(cfg.text_strip_urls),
            "strip_punctuation": bool(cfg.text_strip_punctuation),
            "remove_stopwords": bool(cfg.text_remove_stopwords),
            "deduplicate": bool(cfg.text_deduplicate),
            "apply_stemming": bool(cfg.text_apply_stemming),
            "apply_lemmatization": bool(cfg.text_apply_lemmatization),
            "use_ngrams": bool(cfg.text_use_ngrams),
        }
        # The loader samples the subset itself unless cleaning already did;
        # sampling twice turned a 50% subset into 25%.
        loader_subset_percent = float(cfg.subset_percent) if bool(cfg.use_random_subset) else 100.0
        if any(text_cleaning.values()):
            from data_pipeline.cleaning import clean_text_dataframe
            from data_pipeline.io_utils import read_structured_file

            raw_df = read_structured_file(data_path)
            if bool(cfg.use_random_subset) and float(cfg.subset_percent) < 100:
                from data_pipeline.io_utils import apply_random_subset
                raw_df, sampled_rows = apply_random_subset(
                    raw_df,
                    enabled=True,
                    subset_percent=float(cfg.subset_percent),
                    subset_seed=int(cfg.subset_seed),
                )
                loader_subset_percent = 100.0
                log.log(f"  ✓ Sampled {sampled_rows} text rows before cleaning")
                yield Update()
            cleaned_df, clean_report = clean_text_dataframe(
                raw_df,
                text_col=cfg.text_col,
                lowercase=bool(cfg.text_lowercase),
                strip_urls=bool(cfg.text_strip_urls),
                strip_punctuation=bool(cfg.text_strip_punctuation),
                remove_stopwords=bool(cfg.text_remove_stopwords),
                deduplicate=bool(cfg.text_deduplicate),
                apply_stemming=bool(cfg.text_apply_stemming),
                apply_lemmatization=bool(cfg.text_apply_lemmatization),
            )
            log.log("  ✓ Text cleaning:\n" + clean_report["summary"])
            yield Update()
            _cleaning_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            cleaned_df.to_csv(_cleaning_tmp.name, index=False)
            _cleaning_tmp.close()
            ctx.temp_files.append(_cleaning_tmp.name)
            data_path = _cleaning_tmp.name

        train_loader, val_loader, classes, prep, vocab_size, val_texts = load_text_data(
            data_path, text_col=cfg.text_col, label_col=cfg.label_col,
            model_name=cfg.model_name, batch_size=bs,
            max_length=TEXT_DEFAULTS["max_length"],
            val_split=_val_split,
            cleaning_options=text_cleaning,
            subset_percent=loader_subset_percent,
            subset_seed=int(cfg.subset_seed))
        extra_data.update({"vocab_size": vocab_size, "val_texts": val_texts})

    elif cfg.modality == "tabular":
        from modalities.tabular import load_tabular_data
        # Optional data cleaning
        _cleaning_tmp = None
        # See the text branch: don't let the loader sample a second time.
        loader_subset_percent = float(cfg.subset_percent) if bool(cfg.use_random_subset) else 100.0
        if cfg.use_data_cleaning:
            from data_pipeline.cleaning import clean_dataframe, cleaning_report_markdown
            from data_pipeline.io_utils import read_structured_file
            raw_df = read_structured_file(data_path)
            if bool(cfg.use_random_subset) and float(cfg.subset_percent) < 100:
                from data_pipeline.io_utils import apply_random_subset
                raw_df, sampled_rows = apply_random_subset(
                    raw_df,
                    enabled=True,
                    subset_percent=float(cfg.subset_percent),
                    subset_seed=int(cfg.subset_seed),
                )
                loader_subset_percent = 100.0
                log.log(f"  ✓ Sampled {sampled_rows} tabular rows before cleaning")
                yield Update()
            cleaned_df, clean_report = clean_dataframe(
                raw_df,
                cfg.label_col,
                strategy=str(cfg.tabular_missing_strategy),
                clip_outliers=bool(cfg.tabular_clip_outliers),
            )
            log.log("  ✓ Data cleaning:\n" + cleaning_report_markdown(clean_report))
            yield Update()
            _cleaning_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            cleaned_df.to_csv(_cleaning_tmp.name, index=False)
            _cleaning_tmp.close()
            ctx.temp_files.append(_cleaning_tmp.name)
            data_path = _cleaning_tmp.name

        train_loader, val_loader, classes, prep, input_size, \
            X_train, y_train, X_val, y_val = load_tabular_data(
                data_path, label_col=cfg.label_col, feature_cols=cfg.feature_cols, batch_size=bs,
                val_split=_val_split, augmentation=cfg.augmentation,
                scaling_strategy=str(cfg.tabular_scaling),
                subset_percent=loader_subset_percent,
                subset_seed=int(cfg.subset_seed),
                task=cfg.task)
        extra_data.update({"input_size": input_size,
                            "X_train": X_train, "y_train": y_train,
                            "X_val":   X_val,   "y_val":   y_val})

        # Column type inference summary
        try:
            from data_pipeline.type_inference import suggest_features_and_label
            from data_pipeline.io_utils import read_structured_file
            df_peek = read_structured_file(data_path, nrows=500)
            suggestions  = suggest_features_and_label(df_peek)
            if suggestions.get("warnings"):
                for w in suggestions["warnings"]:
                    log.log(f"  ⚠️ {w}")
                yield Update()
        except Exception:
            pass

    elif cfg.modality == "timeseries":
        from modalities.timeseries import load_timeseries_data
        if bool(cfg.timeseries_sort_by_time) or str(cfg.timeseries_fill_strategy) != "none":
            from data_pipeline.cleaning import clean_timeseries_dataframe
            from data_pipeline.io_utils import read_structured_file

            raw_df = read_structured_file(data_path)
            cleaned_df, clean_report = clean_timeseries_dataframe(
                raw_df,
                label_col=cfg.label_col,
                time_col=cfg.time_col if cfg.time_col else None,
                sort_by_time=bool(cfg.timeseries_sort_by_time),
                fill_strategy=str(cfg.timeseries_fill_strategy),
            )
            log.log("  ✓ Time-series cleaning:\n" + clean_report["summary"])
            yield Update()
            _cleaning_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            cleaned_df.to_csv(_cleaning_tmp.name, index=False)
            _cleaning_tmp.close()
            ctx.temp_files.append(_cleaning_tmp.name)
            data_path = _cleaning_tmp.name

        train_loader, val_loader, classes, prep, input_size, \
            X_train, y_train, X_val, y_val = load_timeseries_data(
            data_path, label_col=cfg.label_col,
            feature_cols=cfg.feature_cols,
            time_col=cfg.time_col if cfg.time_col else None,
            window_size=int(cfg.window_size), batch_size=bs,
            val_split=_val_split, augmentation=cfg.augmentation, task=cfg.task,
            subset_percent=float(cfg.subset_percent) if bool(cfg.use_random_subset) else 100.0,
            subset_seed=int(cfg.subset_seed),
            forecast_mode=bool(cfg.timeseries_forecast_mode),
            forecast_horizon=int(cfg.timeseries_forecast_horizon),
            lag_steps=int(cfg.timeseries_lag_steps),
            rolling_window=int(cfg.timeseries_rolling_window))
        extra_data.update({
            "input_size": input_size,
            "sklearn_input_size": prep.get("sklearn_input_size", X_train.shape[1] if len(X_train.shape) > 1 else input_size),
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        })

    elif cfg.modality == "audio":
        from modalities.audio import load_audio_data
        audio_aug_options = {
            "noise": bool(cfg.audio_aug_noise),
            "time_shift": bool(cfg.audio_aug_shift),
            "gain_jitter": bool(cfg.audio_aug_gain),
            "time_mask": bool(cfg.audio_aug_time_mask),
            "freq_mask": bool(cfg.audio_aug_freq_mask),
        }
        train_loader, val_loader, classes, prep = load_audio_data(
            data_path, batch_size=bs, sample_rate=int(cfg.sample_rate),
            val_split=_val_split, augmentation=cfg.augmentation,
            verify_files=bool(cfg.audio_verify_files),
            normalize_waveform=bool(cfg.audio_normalize_waveform),
            image_size=int(cfg.audio_image_size),
            n_mels=int(cfg.audio_n_mels),
            augmentation_options=audio_aug_options,
        )

    elif cfg.modality == "video":
        from modalities.video import load_video_data
        train_loader, val_loader, classes, prep = load_video_data(
            data_path, n_frames=int(cfg.n_frames),
            batch_size=VIDEO_DEFAULTS["batch_size"],
            val_split=_val_split,
            verify_files=bool(cfg.video_verify_files))

    num_classes = len(classes)
    log.log(f"  ✓ {num_classes} classes: {', '.join(classes)}")
    yield Update()

    ctx.data_path = data_path
    ctx.train_loader, ctx.val_loader = train_loader, val_loader
    ctx.classes, ctx.prep, ctx.num_classes = classes, prep, num_classes
