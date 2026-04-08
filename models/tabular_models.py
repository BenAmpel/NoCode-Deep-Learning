import torch.nn as nn


def get_tabular_model(
    model_name: str,
    num_classes: int,
    input_size: int,
    task: str = "classification",
    dropout: float = 0.3,
    n_estimators: int = 100,
    max_depth: int = None,
    C: float = 1.0,
    max_iter: int = 1000,
    learning_rate: float = 0.1,
):
    if model_name == "MLP":
        return MLP(input_size=input_size, num_classes=num_classes, dropout=dropout)

    if model_name == "RandomForest":
        if task == "regression":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth if (max_depth is not None and max_depth > 0) else None,
                random_state=42,
                n_jobs=-1,
            )
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if (max_depth is not None and max_depth > 0) else None,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "LogisticRegression":
        if task == "regression":
            raise ValueError("LogisticRegression is classification-only. Choose RandomForest, XGBoost, or a neural time-series model for regression.")
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    if model_name == "XGBoost":
        try:
            if task == "regression":
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth) if (max_depth is not None and max_depth > 0) else 6,
                    learning_rate=float(learning_rate) if learning_rate else 0.1,
                    random_state=42,
                    verbosity=0,
                )
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if (max_depth is not None and max_depth > 0) else 6,
            learning_rate=float(learning_rate) if learning_rate else 0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )

    raise ValueError(f"Unknown tabular model: {model_name}")


class MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        hidden = max(64, input_size * 2)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def get_features(self, x):
        # Pass through all but last linear layer
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x
