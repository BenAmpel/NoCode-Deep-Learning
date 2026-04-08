from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

config.DEFAULTS["num_workers"] = 0

import app


def main() -> None:
    root = ROOT
    common = dict(
        text_col="",
        label_col="",
        time_col="",
        window_size=12,
        n_frames=8,
        image_size=64,
        sample_rate=16000,
        audio_image_size=128,
        audio_n_mels=64,
        augmentation="none",
        val_split=0.2,
        use_data_cleaning=False,
        tabular_missing_strategy="mean",
        tabular_clip_outliers=False,
        text_lowercase=True,
        text_strip_urls=True,
        text_strip_punctuation=True,
        text_remove_stopwords=False,
        timeseries_sort_by_time=False,
        timeseries_fill_strategy="none",
        image_verify_files=True,
        image_aug_flip=False,
        image_aug_vertical=False,
        image_aug_rotation=False,
        image_aug_color=False,
        image_aug_gray=False,
        image_aug_perspective=False,
        audio_verify_files=False,
        audio_normalize_waveform=False,
        audio_aug_noise=False,
        audio_aug_shift=False,
        audio_aug_gain=False,
        audio_aug_time_mask=False,
        audio_aug_freq_mask=False,
        video_verify_files=False,
        epochs=2,
        lr=1e-3,
        batch_size=16,
        dropout=0.2,
        optimizer="adam",
        scheduler_name="cosine",
        use_amp=False,
        hidden_size=64,
        num_layers=1,
        n_estimators=20,
        max_depth=4,
        C_param=1.0,
        max_iter=200,
        lr_xgb=0.1,
        n_clusters=4,
        use_class_weights=False,
        checkpoint_every=0,
    )

    runs = [
        (
            "pytorch_image",
            dict(
                modality="image",
                data_path=str(root / "fixtures/image_stress_dataset"),
                training_mode="from_scratch",
                model_name="TinyCNN",
                task="classification",
                bundle_name="telemetry_test_pytorch",
                epochs=2,
            ),
        ),
        (
            "sklearn_tabular",
            dict(
                modality="tabular",
                data_path=str(root / "fixtures/tabular_student_success.csv"),
                training_mode="from_scratch",
                model_name="RandomForest",
                task="classification",
                label_col="label",
                bundle_name="telemetry_test_sklearn",
                epochs=1,
                batch_size=8,
            ),
        ),
        (
            "autoencoder_image",
            dict(
                modality="image",
                data_path=str(root / "fixtures/image_stress_dataset"),
                training_mode="from_scratch",
                model_name="Autoencoder",
                task="anomaly",
                bundle_name="telemetry_test_autoencoder",
                epochs=2,
                batch_size=16,
                image_size=224,
            ),
        ),
    ]

    results = []
    for run_name, overrides in runs:
        kwargs = common | overrides
        original_device = app.DEVICE
        if run_name == "autoencoder_image":
            app.DEVICE = "cpu"
        gen = app.run_pipeline(**kwargs)
        etas: list[str] = []
        statuses: list[str] = []
        cards: list[str] = []
        count = 0
        try:
            for out in gen:
                count += 1
                statuses.append(out[0])
                etas.append(out[2])
                cards.append(out[3])
        finally:
            app.DEVICE = original_device
        blank_eta = [eta for eta in etas if eta in (None, "", "Done")]
        results.append(
            {
                "name": run_name,
                "updates": count,
                "first_eta": etas[0] if etas else None,
                "last_eta": etas[-1] if etas else None,
                "blank_eta_count": len(blank_eta),
                "last_status_tail": statuses[-1].splitlines()[-1] if statuses else None,
                "telemetry_card_has_phase": ("Phase" in (cards[-1] or "")) if cards else False,
                "telemetry_card_has_epoch": ("Epoch" in (cards[-1] or "")) if cards else False,
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
