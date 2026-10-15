"""Settings for one training run.

``RunConfig`` replaces the ~75 positional parameters ``run_pipeline`` used to
take. Field names match the Gradio components that supply them, so the UI
builds a config from a ``{field name: component}`` dict and the Train and
sweep buttons can no longer drift out of step with the pipeline.

Values arrive exactly as the UI sends them (Gradio numbers may be floats,
checkboxes bools, dropdowns strings); the pipeline converts them where used.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RunConfig:
    # Data
    modality: str
    data_path: str
    text_col: str
    label_col: str
    feature_cols: Any
    time_col: str
    window_size: Any
    n_frames: Any
    image_size: Any
    sample_rate: Any
    audio_image_size: Any
    audio_n_mels: Any
    augmentation: str
    val_split: Any
    use_random_subset: Any
    subset_percent: Any
    subset_seed: Any
    use_data_cleaning: Any
    tabular_missing_strategy: str
    tabular_clip_outliers: Any
    tabular_scaling: str
    text_lowercase: Any
    text_strip_urls: Any
    text_strip_punctuation: Any
    text_remove_stopwords: Any
    text_deduplicate: Any
    text_apply_stemming: Any
    text_apply_lemmatization: Any
    text_use_ngrams: Any
    timeseries_sort_by_time: Any
    timeseries_fill_strategy: str
    timeseries_forecast_mode: Any
    timeseries_forecast_horizon: Any
    timeseries_lag_steps: Any
    timeseries_rolling_window: Any
    image_verify_files: Any
    image_aug_flip: Any
    image_aug_vertical: Any
    image_aug_rotation: Any
    image_aug_color: Any
    image_aug_gray: Any
    image_aug_perspective: Any
    image_normalization: str
    image_force_grayscale: Any
    audio_verify_files: Any
    audio_normalize_waveform: Any
    audio_aug_noise: Any
    audio_aug_shift: Any
    audio_aug_gain: Any
    audio_aug_time_mask: Any
    audio_aug_freq_mask: Any
    video_verify_files: Any
    # Model
    training_mode: str
    model_name: str
    task: str
    # Hyperparameters
    epochs: Any
    lr: Any
    batch_size: Any
    dropout: Any
    optimizer: str
    scheduler_name: str
    use_amp: Any
    hidden_size: Any
    num_layers: Any
    n_estimators: Any
    max_depth: Any
    C_param: Any
    max_iter: Any
    lr_xgb: Any
    n_clusters: Any
    use_class_weights: Any
    checkpoint_every: Any
    # Export
    bundle_name: str

    @classmethod
    def from_component_values(
        cls,
        components: Mapping[str, Any],
        values: Mapping[Any, Any],
        **overrides: Any,
    ) -> "RunConfig":
        """Build a config from a Gradio event's dict of input values.

        ``components`` maps field names to components; ``values`` is what
        Gradio passes a handler whose ``inputs`` is a set of components
        (component → value). ``overrides`` fill fields that have no component
        in this event, e.g. ``model_name`` for a model sweep.
        """
        return cls(**{name: values[comp] for name, comp in components.items()}, **overrides)

    def replace(self, **changes: Any) -> "RunConfig":
        return dataclasses.replace(self, **changes)


RUN_CONFIG_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(RunConfig))
