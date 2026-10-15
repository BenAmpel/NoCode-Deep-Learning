"""Stage 2: build the model the run trains."""
from __future__ import annotations

from typing import Iterator

from models.registry import is_sklearn, is_vit, is_whisper
from pipeline.context import RunContext
from pipeline.progress import Update


def build_model(ctx: RunContext) -> Iterator[Update]:
    cfg, log = ctx.cfg, ctx.log
    extra_data = ctx.extra_data
    num_classes = ctx.num_classes
    runtime_device = ctx.runtime_device

    log.set_phase("building model", eta_text="ETA: building model")
    ctx.progress(0.12, desc="Building model…")
    log.log(f"Building {cfg.model_name} ({cfg.training_mode})…")
    yield Update()

    graph_node2vec = cfg.modality == "graph" and cfg.model_name == "Node2Vec"
    sklearn_model = is_sklearn(cfg.model_name)

    if cfg.model_name == "Autoencoder":
        from models.autoencoder import get_autoencoder
        ae_modality = cfg.modality if cfg.modality in ("image", "audio", "tabular") else (
            "tabular" if cfg.modality == "timeseries" else None
        )
        if ae_modality is None:
            raise ValueError(
                f"Autoencoder is not supported for the '{cfg.modality}' modality. "
                "Supported: image, audio, tabular, timeseries."
            )
        model = get_autoencoder(
            modality=ae_modality,
            input_size=extra_data.get("input_size"),
            latent_dim=128,
        )

    elif graph_node2vec:
        model = None

    elif sklearn_model:
        from models.tabular_models import get_tabular_model
        model = get_tabular_model(
            cfg.model_name, num_classes=num_classes,
            input_size=extra_data.get("sklearn_input_size", extra_data.get("input_size", 10)),
            task=cfg.task,
            n_estimators=int(cfg.n_estimators),
            max_depth=int(cfg.max_depth),
            C=float(cfg.C_param), max_iter=int(cfg.max_iter),
            learning_rate=float(cfg.lr_xgb),
        )

    elif is_vit(cfg.model_name):
        from models.vit_models import get_vit_model
        model = get_vit_model(cfg.model_name, num_classes=num_classes,
                               pretrained=(cfg.training_mode == "fine-tune"))

    elif is_whisper(cfg.model_name):
        from models.whisper_model import get_whisper_model
        size = "tiny" if "Tiny" in cfg.model_name else "base"
        model = get_whisper_model(num_classes=num_classes, model_size=size,
                                   freeze_encoder=(cfg.training_mode == "fine-tune"))

    elif cfg.modality in ("image", "audio"):
        from models.image_models import get_image_model
        model = get_image_model(cfg.model_name, num_classes=num_classes,
                                mode=cfg.training_mode, dropout=float(cfg.dropout))

    elif cfg.modality == "text":
        from models.text_models import get_text_model
        model = get_text_model(
            cfg.model_name, num_classes=num_classes, mode=cfg.training_mode,
            vocab_size=extra_data.get("vocab_size", 30522),
            hidden_size=int(cfg.hidden_size), num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout))

    elif cfg.modality == "tabular":
        from models.tabular_models import get_tabular_model
        model = get_tabular_model(
            cfg.model_name, num_classes=num_classes,
            input_size=extra_data["input_size"], dropout=float(cfg.dropout))

    elif cfg.modality == "timeseries":
        from models.timeseries_models import get_timeseries_model
        model = get_timeseries_model(
            cfg.model_name, num_classes=num_classes,
            input_size=extra_data["input_size"],
            hidden_size=int(cfg.hidden_size), num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout))

    elif cfg.modality == "graph":
        from models.graph_models import get_graph_model
        model = get_graph_model(
            cfg.model_name,
            num_classes=num_classes,
            input_size=extra_data["input_size"],
            hidden_size=int(cfg.hidden_size),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
        )

    elif cfg.modality == "video":
        from models.video_models import get_video_model
        model = get_video_model(cfg.model_name, num_classes=num_classes, mode=cfg.training_mode)

    # Neural regression: train on standardized targets so the network
    # doesn't spend its epochs learning the target's scale. The wrapper
    # un-scales its output, so metrics, export and inference are unchanged.
    if (cfg.task == "regression" and model is not None and not sklearn_model
            and cfg.model_name != "Autoencoder" and extra_data.get("y_train") is not None):
        from models.target_scaling import wrap_for_regression
        model, (t_mean, t_std) = wrap_for_regression(model, extra_data["y_train"])
        log.log(f"  ✓ Regression targets standardized (training mean {t_mean:.4g}, std {t_std:.4g})")
        yield Update()

    if runtime_device != ctx.device:
        log.log(
            f"  ✓ Device: {runtime_device} (fallback from {ctx.device} for video compatibility)"
        )
    else:
        log.log(f"  ✓ Device: {runtime_device}")
    yield Update()

    ctx.model = model
    ctx.sklearn_model, ctx.graph_node2vec = sklearn_model, graph_node2vec
