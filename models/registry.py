"""
Central registry mapping modality + training mode → available models,
and modality → compatible tasks.
"""
import sys

REGISTRY: dict[str, dict[str, dict]] = {
    "graph": {
        "fine-tune": {},
        "from_scratch": {
            "Node2Vec":         {"params": ["hidden_size", "max_iter"]},
            "GCN":              {"params": ["learning_rate", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "GraphSAGE":        {"params": ["learning_rate", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "GraphTransformer": {"params": ["learning_rate", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
        },
    },
    "image": {
        "fine-tune": {
            "MobileNetV3-Small": {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "MobileNetV3-Large": {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ResNet18":          {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ResNet50":          {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "DenseNet121":       {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "EfficientNet-B0":   {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "EfficientNet-B2":   {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ConvNeXt-Tiny":     {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ViT-Tiny":          {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ViT-Small":         {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "Autoencoder":       {"params": ["learning_rate", "batch_size", "epochs"]},
        },
        "from_scratch": {
            "TinyCNN":      {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ResNet18":     {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "DenseNet121":  {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "Autoencoder":  {"params": ["learning_rate", "batch_size", "epochs"]},
        },
    },
    "text": {
        "fine-tune": {
            "DistilBERT": {"params": ["learning_rate", "batch_size", "epochs", "optimizer"]},
            "BERT":       {"params": ["learning_rate", "batch_size", "epochs", "optimizer"]},
            "RoBERTa":    {"params": ["learning_rate", "batch_size", "epochs", "optimizer"]},
        },
        "from_scratch": {
            "RNN":         {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "LSTM":        {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "BiLSTM":      {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "CNN-LSTM":    {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "Transformer": {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
        },
    },
    "tabular": {
        "fine-tune": {
            "Autoencoder": {"params": ["learning_rate", "batch_size", "epochs"]},
        },
        "from_scratch": {
            "MLP":                {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "RandomForest":       {"params": ["n_estimators", "max_depth"]},
            "LogisticRegression": {"params": ["C", "max_iter"]},
            "XGBoost":            {"params": ["n_estimators", "max_depth", "learning_rate"]},
            "HistGradientBoosting": {"params": ["n_estimators", "max_depth", "learning_rate"]},
            "Autoencoder":        {"params": ["learning_rate", "batch_size", "epochs"]},
        },
    },
    "timeseries": {
        "fine-tune": {},
        "from_scratch": {
            "LSTM":             {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "GRU":              {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "MLP-Window":       {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "dropout", "optimizer"]},
            "CNN1D":            {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "dropout", "optimizer"]},
            "TCN":              {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "Transformer-Tiny": {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "Transformer-Small": {"params": ["learning_rate", "batch_size", "epochs", "hidden_size", "num_layers", "dropout", "optimizer"]},
            "RandomForest":     {"params": ["n_estimators", "max_depth"]},
            "ExtraTrees":       {"params": ["n_estimators", "max_depth"]},
            "LogisticRegression": {"params": ["C", "max_iter"]},
            "KNeighbors":       {"params": []},
            "LinearRegression": {"params": []},
            "Ridge":            {"params": ["C"]},
            "XGBoost":          {"params": ["n_estimators", "max_depth", "learning_rate"]},
            "HistGradientBoosting": {"params": ["n_estimators", "max_depth", "learning_rate"]},
        },
    },
    "audio": {
        # Audio → mel spectrogram → reuse image models; or Whisper encoder
        "fine-tune": {
            "MobileNetV3-Small": {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "MobileNetV3-Large": {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ResNet18":          {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "EfficientNet-B0":   {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "EfficientNet-B2":   {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            # Whisper-Tiny / Whisper-Base are hidden until the audio loader can
            # produce Whisper's 80-band, 16 kHz, 3000-frame log-mel input. The
            # shared loader yields 3x224x224 mel images, which crash the encoder.
            "Autoencoder":       {"params": ["learning_rate", "batch_size", "epochs"]},
        },
        "from_scratch": {
            "TinyCNN":      {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "ResNet18":     {"params": ["learning_rate", "batch_size", "epochs", "dropout", "optimizer"]},
            "Autoencoder":  {"params": ["learning_rate", "batch_size", "epochs"]},
        },
    },
    "video": {
        "fine-tune": {
            "R3D-18":       {"params": ["learning_rate", "batch_size", "epochs", "n_frames", "optimizer"]},
            "MC3-18":       {"params": ["learning_rate", "batch_size", "epochs", "n_frames", "optimizer"]},
            "R(2+1)D-18":   {"params": ["learning_rate", "batch_size", "epochs", "n_frames", "optimizer"]},
        },
        "from_scratch": {
            "TinyR3D": {"params": ["learning_rate", "batch_size", "epochs", "n_frames", "optimizer"]},
        },
    },
}

SKLEARN_MODELS = {
    "RandomForest",
    "ExtraTrees",
    "LogisticRegression",
    "KNeighbors",
    "LinearRegression",
    "Ridge",
    "XGBoost",
    "HistGradientBoosting",
}
VIT_MODELS     = {"ViT-Tiny", "ViT-Small"}
WHISPER_MODELS = {"Whisper-Tiny", "Whisper-Base"}

TASK_MODALITY_SUPPORT: dict[str, list[str]] = {
    "classification":  ["graph", "image", "text", "tabular", "timeseries", "audio", "video"],
    # Multi-label is hidden until loaders emit multi-hot targets; today every
    # loader yields one integer label, so BCEWithLogitsLoss fails on shape.
    "multi-label":     [],
    "clustering":      ["image", "text", "tabular", "timeseries", "audio"],
    "regression":      ["tabular", "timeseries"],
    "anomaly":         ["image", "tabular", "audio"],
}


XGBOOST_UNAVAILABLE_NOTE = (
    "XGBoost is not available on this computer. Use **HistGradientBoosting**, "
    "scikit-learn's built-in gradient-boosted trees, for the same kind of model."
)

_xgboost_status: tuple[bool, str] | None = None


def xgboost_status() -> tuple[bool, str]:
    """Return (available, reason). Cached after the first call.

    On macOS XGBoost is never offered. Its wheels link an external
    ``@rpath/libomp.dylib`` that end users (no Homebrew) do not have, and even
    when one is found it is a second OpenMP runtime next to the one torch
    ships, which segfaults on the first fit once torch has been imported.
    The app always imports torch first, so the import must not be attempted.
    """
    global _xgboost_status
    if _xgboost_status is None:
        if sys.platform == "darwin":
            _xgboost_status = (False, "XGBoost is disabled on macOS (its OpenMP runtime conflicts with PyTorch's).")
        else:
            try:
                import xgboost  # noqa: F401
                _xgboost_status = (True, "")
            except Exception as exc:  # ImportError, or XGBoostError when the native library fails to load
                _xgboost_status = (False, f"XGBoost could not be loaded: {exc}")
    return _xgboost_status


def xgboost_available() -> bool:
    return xgboost_status()[0]


def get_models(modality: str, mode: str) -> list[str]:
    models = list(REGISTRY.get(modality, {}).get(mode, {}).keys())
    if "XGBoost" in models and not xgboost_available():
        models.remove("XGBoost")
    return models


def get_compatible_tasks(modality: str) -> list[str]:
    return [task for task, mods in TASK_MODALITY_SUPPORT.items() if modality in mods]


def get_modes(modality: str) -> list[str]:
    available = []
    for mode in ("fine-tune", "from_scratch"):
        if REGISTRY.get(modality, {}).get(mode):
            available.append(mode)
    return available


def is_sklearn(model_name: str) -> bool:
    return model_name in SKLEARN_MODELS


def is_vit(model_name: str) -> bool:
    return model_name in VIT_MODELS


def is_whisper(model_name: str) -> bool:
    return model_name in WHISPER_MODELS
