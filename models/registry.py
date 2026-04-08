"""
Central registry mapping modality + training mode → available models,
and modality → compatible tasks.
"""

REGISTRY: dict[str, dict[str, dict]] = {
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
            "RandomForest":     {"params": ["n_estimators", "max_depth"]},
            "LogisticRegression": {"params": ["C", "max_iter"]},
            "XGBoost":          {"params": ["n_estimators", "max_depth", "learning_rate"]},
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
            "Whisper-Tiny":      {"params": ["learning_rate", "batch_size", "epochs", "optimizer"]},
            "Whisper-Base":      {"params": ["learning_rate", "batch_size", "epochs", "optimizer"]},
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

SKLEARN_MODELS = {"RandomForest", "LogisticRegression", "XGBoost"}
VIT_MODELS     = {"ViT-Tiny", "ViT-Small"}
WHISPER_MODELS = {"Whisper-Tiny", "Whisper-Base"}

TASK_MODALITY_SUPPORT: dict[str, list[str]] = {
    "classification":  ["image", "text", "tabular", "timeseries", "audio", "video"],
    "multi-label":     ["image", "text", "tabular", "audio"],
    "clustering":      ["image", "text", "tabular", "timeseries", "audio"],
    "regression":      ["tabular", "timeseries"],
    "anomaly":         ["image", "tabular", "audio"],
}


def get_models(modality: str, mode: str) -> list[str]:
    return list(REGISTRY.get(modality, {}).get(mode, {}).keys())


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
