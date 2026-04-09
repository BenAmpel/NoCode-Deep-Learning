"""
Structured recommendation helpers for Phase 1 guided workflow.
"""
from __future__ import annotations

from typing import Any


def recommend(
    modality: str,
    stats: dict[str, Any],
    hardware: str,
) -> dict[str, Any]:
    n_samples = int(stats.get("n_samples", 0) or 0)
    n_classes = int(stats.get("n_classes", 0) or 0)
    imbalance = float(stats.get("imbalance_ratio", 1.0) or 1.0)

    rec: dict[str, Any] = {
        "training_mode": "fine-tune",
        "model_name": "MobileNetV3-Small",
        "task": "classification",
        "augmentation": "light",
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "dropout": 0.3,
        "scheduler": "cosine",
        "use_class_weights": imbalance >= 2.0,
        "warnings": [],
        "rationale": [],
        "confidence": "moderate",
    }

    rec["rationale"].append(
        f"The dataset preview found {n_samples} samples across {n_classes} classes on {hardware.upper()} hardware."
    )

    if modality == "image":
        if n_samples < 400:
            rec.update(
                training_mode="fine-tune",
                model_name="MobileNetV3-Small",
                augmentation="medium",
                image_size=128,
                batch_size=8 if hardware in {"mps", "cpu"} else 16,
                epochs=6,
                dropout=0.35,
            )
            rec["rationale"].append(
                "A compact pretrained image backbone is the safest baseline for a smaller image dataset."
            )
        elif n_samples < 2500:
            rec.update(
                training_mode="fine-tune",
                model_name="EfficientNet-B0",
                augmentation="medium",
                image_size=160,
                batch_size=12 if hardware in {"mps", "cpu"} else 24,
                epochs=8,
            )
            rec["rationale"].append(
                "This dataset is large enough to benefit from a balanced pretrained backbone without paying the cost of a heavier model."
            )
        else:
            rec.update(
                training_mode="fine-tune",
                model_name="ConvNeXt-Tiny",
                augmentation="light",
                image_size=224,
                batch_size=8 if hardware in {"mps", "cpu"} else 16,
                epochs=10,
            )
            rec["rationale"].append(
                "A larger image dataset can support a stronger visual backbone and a slightly larger training budget."
            )
        if imbalance >= 3.0:
            rec["rationale"].append(
                "Class imbalance is high, so class-weighting is recommended alongside augmentation."
            )
    elif modality == "tabular":
        rec.update(
            training_mode="from_scratch",
            task="classification",
            model_name="RandomForest" if n_samples < 5000 else "XGBoost",
            augmentation="none",
            image_size=160,
            batch_size=32,
            epochs=1,
            dropout=0.1,
            scheduler="none",
        )
        rec["rationale"].append(
            "Classical tabular baselines are usually the fastest and strongest place to start before trying a neural model."
        )
        if n_samples >= 5000:
            rec["rationale"].append(
                "The preview suggests enough rows to justify a boosted-tree baseline."
            )
    elif modality == "text":
        rec.update(
            training_mode="fine-tune",
            model_name="DistilBERT" if n_samples < 2000 else "BERT",
            augmentation="none",
            image_size=160,
            batch_size=8 if hardware in {"mps", "cpu"} else 16,
            epochs=4 if n_samples < 2000 else 5,
            dropout=0.2,
            scheduler="warmup_cosine",
        )
        rec["rationale"].append(
            "A transformer fine-tune baseline is the fastest way to get a strong result on text without hand-built features."
        )
        if n_samples < 300:
            rec["warnings"].append(
                "The dataset is small for text classification, so evaluation may vary noticeably across splits."
            )
            rec["confidence"] = "cautious"
    elif modality == "graph":
        rec.update(
            training_mode="from_scratch",
            model_name="Node2Vec" if n_samples < 5000 else "GCN",
            augmentation="none",
            batch_size=1,
            epochs=12,
            dropout=0.2,
            scheduler="none",
        )
        rec["rationale"].append(
            "Graph workflows start best with a structural embedding or a compact message-passing baseline before trying deeper graph architectures."
        )
    else:
        rec["rationale"].append(
            "Guided recommendations are most detailed for image, tabular, and text in this phase."
        )
        rec["confidence"] = "cautious"

    if imbalance >= 5.0:
        rec["warnings"].append(
            "Severe class imbalance detected. Compare accuracy with F1, recall, and per-class performance."
        )
        rec["confidence"] = "cautious"
    elif imbalance >= 2.0:
        rec["rationale"].append(
            "Mild-to-moderate class imbalance was detected, so weighted training is enabled."
        )

    if n_samples < 100:
        rec["warnings"].append(
            "Very small dataset. Treat the first run as a baseline check, not a final benchmark."
        )
        rec["confidence"] = "cautious"

    return rec


def format_recommendation_summary(rec: dict[str, Any]) -> str:
    return (
        "### Recommended baseline\n\n"
        f"- **Training mode**: `{rec.get('training_mode', 'fine-tune')}`\n"
        f"- **Model family**: `{rec.get('model_name', 'MobileNetV3-Small')}`\n"
        f"- **Task**: `{rec.get('task', 'classification')}`\n"
        f"- **Augmentation**: `{rec.get('augmentation', 'light')}`\n"
        f"- **Image size**: `{rec.get('image_size', 160)}`\n"
        f"- **Batch size**: `{rec.get('batch_size', 16)}`\n"
        f"- **Epochs**: `{rec.get('epochs', 8)}`\n"
        f"- **Dropout**: `{rec.get('dropout', 0.3)}`\n"
        f"- **Scheduler**: `{rec.get('scheduler', 'cosine')}`\n"
        f"- **Use class weights**: `{bool(rec.get('use_class_weights', False))}`\n"
        f"- **Recommendation confidence**: `{rec.get('confidence', 'moderate')}`"
    )


def format_recommendation_rationale(rec: dict[str, Any]) -> str:
    rationale = rec.get("rationale", [])
    warnings = rec.get("warnings", [])
    lines = ["### Why this baseline", ""]
    if rationale:
        lines.extend(f"- {line}" for line in rationale)
    else:
        lines.append("- The current recommendation uses conservative defaults.")
    if warnings:
        lines.extend(["", "### Watch-outs", ""])
        lines.extend(f"- {line}" for line in warnings)
    return "\n".join(lines)
