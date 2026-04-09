from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

config.DEFAULTS["num_workers"] = 0

import app
from detection.yolo_detector import YOLO_MODELS
from models.registry import REGISTRY
from ui import annotation_tools


def _make_text_fixture(tmpdir: Path) -> Path:
    path = tmpdir / "text_fixture.csv"
    rows = [
        {"text": "I loved this course and the projects were clear.", "label": "positive"},
        {"text": "This was confusing and frustrating.", "label": "negative"},
        {"text": "Helpful explanations and good pacing.", "label": "positive"},
        {"text": "The examples were too short to understand.", "label": "negative"},
        {"text": "I loved this course and the projects were clear.", "label": "positive"},
        {"text": "Neutral update with no strong feeling.", "label": "neutral"},
        {"text": "Great overview and strong visuals.", "label": "positive"},
        {"text": "I could not follow the text preprocessing part.", "label": "negative"},
        {"text": "Decent walkthrough overall.", "label": "neutral"},
    ]
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_graph_fixture(tmpdir: Path) -> Path:
    graph_dir = tmpdir / "graph_fixture"
    graph_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    nodes = pd.DataFrame(
        [
            {"node_id": f"n{i}", "label": "A" if i < 5 else "B", "f1": float(i), "f2": float(i % 3)}
            for i in range(10)
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "n0", "target": "n1"},
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
            {"source": "n5", "target": "n6"},
            {"source": "n6", "target": "n7"},
            {"source": "n7", "target": "n8"},
            {"source": "n8", "target": "n9"},
            {"source": "n4", "target": "n5"},
            {"source": "n0", "target": "n9"},
        ]
    )
    nodes.to_csv(graph_dir / "nodes.csv", index=False)
    edges.to_csv(graph_dir / "edges.csv", index=False)
    return graph_dir


def _make_video_fixture(tmpdir: Path) -> Path:
    src = ROOT / "fixtures" / "object_detection_test"
    target = tmpdir / "video_fixture"
    (target / "bus").mkdir(parents=True, exist_ok=True)
    (target / "people").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "bus_loop.mp4", target / "bus" / "bus_0.mp4")
    shutil.copy2(src / "bus_loop.mp4", target / "people" / "people_0.mp4")
    return target


def _status_tail(text: str) -> str:
    return (text or "").strip().splitlines()[-1] if text else ""


def _preview(app_module, **kwargs) -> dict[str, Any]:
    out = app_module.preview_dataset(**kwargs)
    return {
        "summary": str(out[0])[:400],
        "quality": str(out[2])[:500],
        "recommendation": str(out[3])[:500],
        "why": str(out[6])[:400] if len(out) > 6 else "",
    }


def _drain_pipeline(**kwargs) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    updates = 0
    last = None
    for out in app.run_pipeline(**kwargs):
        updates += 1
        last = out
    plt.close("all")
    if last is None:
        raise RuntimeError("run_pipeline produced no updates")
    return {
        "updates": updates,
        "status_tail": _status_tail(last[0]),
        "eta": last[2],
        "telemetry_card": str(last[3])[:300],
        "eval_summary": str(last[4])[:500],
        "bundle_path": last[12],
        "dashboard_kpi": str(last[24])[:400],
    }


def _common_args() -> dict[str, Any]:
    return dict(
        text_col="",
        label_col="",
        feature_cols=[],
        time_col="",
        window_size=12,
        n_frames=8,
        image_size=64,
        sample_rate=16000,
        audio_image_size=128,
        audio_n_mels=64,
        augmentation="none",
        val_split=0.2,
        use_random_subset=False,
        subset_percent=100.0,
        subset_seed=42,
        use_data_cleaning=False,
        tabular_missing_strategy="mean",
        tabular_clip_outliers=False,
        tabular_scaling="standard",
        text_lowercase=True,
        text_strip_urls=True,
        text_strip_punctuation=True,
        text_remove_stopwords=False,
        text_deduplicate=False,
        text_apply_stemming=False,
        text_apply_lemmatization=False,
        text_use_ngrams=False,
        timeseries_sort_by_time=True,
        timeseries_fill_strategy="forward_fill",
        timeseries_forecast_mode=False,
        timeseries_forecast_horizon=1,
        timeseries_lag_steps=3,
        timeseries_rolling_window=3,
        image_verify_files=True,
        image_aug_flip=False,
        image_aug_vertical=False,
        image_aug_rotation=False,
        image_aug_color=False,
        image_aug_gray=False,
        image_aug_perspective=False,
        image_normalization="imagenet",
        image_force_grayscale=False,
        audio_verify_files=True,
        audio_normalize_waveform=False,
        audio_aug_noise=False,
        audio_aug_shift=False,
        audio_aug_gain=False,
        audio_aug_time_mask=False,
        audio_aug_freq_mask=False,
        video_verify_files=True,
        training_mode="from_scratch",
        model_name="TinyCNN",
        task="classification",
        epochs=1,
        lr=1e-3,
        batch_size=8,
        dropout=0.2,
        optimizer="adam",
        scheduler_name="none",
        use_amp=False,
        hidden_size=32,
        num_layers=1,
        n_estimators=10,
        max_depth=4,
        C_param=1.0,
        max_iter=100,
        lr_xgb=0.1,
        n_clusters=3,
        use_class_weights=False,
        checkpoint_every=0,
        bundle_name="smoke_bundle",
    )


def _test_preview_layer(text_fixture: Path, graph_fixture: Path) -> dict[str, Any]:
    common = dict(
        subset_enabled=False,
        subset_pct=100.0,
        subset_seed_val=42,
        text_lowercase_val=True,
        text_strip_urls_val=True,
        text_strip_punctuation_val=True,
        text_remove_stopwords_val=False,
        text_deduplicate_val=False,
        text_apply_stemming_val=False,
        text_apply_lemmatization_val=False,
        image_size_val=64,
        image_aug_flip_val=False,
        image_aug_vertical_val=False,
        image_aug_rotation_val=False,
        image_aug_color_val=False,
        image_aug_gray_val=False,
        image_aug_perspective_val=False,
        image_normalization_val="imagenet",
        image_force_grayscale_val=False,
        project_mode_val="Guided",
        model_val="TinyCNN",
        task_val="classification",
    )
    return {
        "image": _preview(app, dpath=str(ROOT / "fixtures" / "image_stress_dataset"), mod="image", lcol="", txt_col="", **common),
        "text": _preview(app, dpath=str(text_fixture), mod="text", lcol="label", txt_col="text", **common),
        "tabular": _preview(app, dpath=str(ROOT / "fixtures" / "tabular_student_success.csv"), mod="tabular", lcol="label", txt_col="", **common),
        "timeseries": _preview(app, dpath=str(ROOT / "fixtures" / "timeseries_stress.csv"), mod="timeseries", lcol="label", txt_col="", **common),
        "audio": _preview(app, dpath=str(ROOT / "fixtures" / "audio_stress_dataset"), mod="audio", lcol="", txt_col="", **common),
        "video": _preview(app, dpath=str(_make_video_fixture(Path(tempfile.mkdtemp(prefix='video_preview_', dir=ROOT / 'outputs')))), mod="video", lcol="", txt_col="", **common),
        "graph": _preview(app, dpath=str(graph_fixture), mod="graph", lcol="label", txt_col="", **common),
    }


def _test_walkthroughs() -> dict[str, Any]:
    result = {}
    for modality in ["image", "tabular", "timeseries", "text"]:
        out = app.load_builtin_walkthrough(modality)
        result[modality] = {
            "items": len(out),
            "first": str(out[0])[:120] if out else "",
        }
    return result


def _test_detection_and_annotation(tmpdir: Path, text_fixture: Path) -> dict[str, Any]:
    image_path = ROOT / "fixtures" / "object_detection_test" / "bus.jpg"
    video_path = ROOT / "fixtures" / "object_detection_test" / "bus_loop.mp4"
    model_key = next(iter(YOLO_MODELS))
    det_img = app.detect_image(str(image_path), model_key, 0.25, 0.45)
    det_vid_last = None
    for det_vid in app.detect_video(str(video_path), model_key, 0.25, 0.45):
        det_vid_last = det_vid
    if det_vid_last is None:
        raise RuntimeError("detect_video produced no updates")

    image_labels = annotation_tools.list_folder_labels(ROOT / "fixtures" / "image_stress_dataset", annotation_tools.IMAGE_SUFFIXES)
    image_samples = annotation_tools.list_folder_samples(ROOT / "fixtures" / "image_stress_dataset", annotation_tools.IMAGE_SUFFIXES, limit=5)
    sample_rel = image_samples[0][1]
    preview_path, preview_label = annotation_tools.preview_folder_sample(ROOT / "fixtures" / "image_stress_dataset", sample_rel)

    img_copy = tmpdir / "image_annot"
    shutil.copytree(ROOT / "fixtures" / "image_stress_dataset", img_copy, dirs_exist_ok=True)
    moved_rel, moved_label = annotation_tools.relabel_folder_sample(img_copy, sample_rel, "relabeled_image")

    audio_copy = tmpdir / "audio_annot"
    shutil.copytree(ROOT / "fixtures" / "audio_stress_dataset", audio_copy, dirs_exist_ok=True)
    audio_samples = annotation_tools.list_folder_samples(audio_copy, annotation_tools.AUDIO_SUFFIXES, limit=5)
    audio_moved_rel, audio_moved_label = annotation_tools.relabel_folder_sample(audio_copy, audio_samples[0][1], "relabeled_audio")

    row_choices, text_labels = annotation_tools.list_text_annotation_rows(text_fixture, "text", "label", limit=10)
    text_preview, current_text_label = annotation_tools.preview_text_row(text_fixture, row_choices[0][1], "text", "label")
    updated_label = annotation_tools.save_text_relabel(text_fixture, row_choices[0][1], "label", "updated_label")

    review_img, review_rows, review_md, review_state = annotation_tools.review_object_boxes(str(image_path), model_key, 0.25, 0.45)
    saved_review = annotation_tools.save_object_box_review(str(image_path), review_rows, output_dir=tmpdir / "review_boxes")

    return {
        "detect_image": {"summary": str(det_img[2])[:300]},
        "detect_video": {"summary": str(det_vid_last[2])[:300]},
        "annotation": {
            "image_labels": image_labels,
            "image_preview_label": preview_label,
            "image_preview_path_exists": Path(preview_path).exists(),
            "image_relabel": {"rel": moved_rel, "label": moved_label},
            "audio_relabel": {"rel": audio_moved_rel, "label": audio_moved_label},
            "text_labels": text_labels,
            "text_preview_label": current_text_label,
            "text_preview_snippet": text_preview[:80],
            "text_updated_label": updated_label,
            "review_rows": len(review_rows),
            "review_markdown": str(review_md)[:300],
            "review_saved": saved_review,
            "review_state_keys": sorted(review_state.keys()),
            "review_has_image": review_img is not None,
        },
    }


def _test_model_constructors() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for modality, mode_map in REGISTRY.items():
        for mode, models in mode_map.items():
            for model_name in models:
                key = f"{modality}:{mode}:{model_name}"
                try:
                    if modality in {"image", "audio"}:
                        if model_name == "Autoencoder":
                            from models.autoencoder import get_autoencoder
                            ae_modality = "audio" if modality == "audio" else "image"
                            model = get_autoencoder(modality=ae_modality, input_size=16, latent_dim=8)
                        elif model_name in {"ViT-Tiny", "ViT-Small"}:
                            from models.vit_models import get_vit_model
                            model = get_vit_model(model_name, num_classes=3, pretrained=(mode == "fine-tune"))
                        elif model_name in {"Whisper-Tiny", "Whisper-Base"}:
                            from models.whisper_model import get_whisper_model
                            size = "tiny" if "Tiny" in model_name else "base"
                            model = get_whisper_model(num_classes=3, model_size=size, freeze_encoder=(mode == "fine-tune"))
                        else:
                            from models.image_models import get_image_model
                            model = get_image_model(model_name, num_classes=3, mode=mode, dropout=0.1)
                    elif modality == "text":
                        from models.text_models import get_text_model
                        model = get_text_model(model_name, num_classes=3, mode=mode, vocab_size=500, hidden_size=32, num_layers=1, dropout=0.1)
                    elif modality == "tabular":
                        if model_name == "Autoencoder":
                            from models.autoencoder import get_autoencoder
                            model = get_autoencoder(modality="tabular", input_size=8, latent_dim=4)
                        else:
                            from models.tabular_models import get_tabular_model
                            model = get_tabular_model(model_name, num_classes=3, input_size=8, task="classification", n_estimators=5, max_depth=3, C=1.0, max_iter=50, learning_rate=0.1)
                    elif modality == "timeseries":
                        if model_name in {"LinearRegression", "Ridge"}:
                            task = "regression"
                        else:
                            task = "classification"
                        from models.timeseries_models import get_timeseries_model
                        if model_name in {"RandomForest", "ExtraTrees", "LogisticRegression", "KNeighbors", "LinearRegression", "Ridge", "XGBoost"}:
                            from models.tabular_models import get_tabular_model
                            model = get_tabular_model(model_name, num_classes=3, input_size=16, task=task, n_estimators=5, max_depth=3, C=1.0, max_iter=50, learning_rate=0.1)
                        else:
                            model = get_timeseries_model(model_name, num_classes=3, input_size=4, hidden_size=16, num_layers=1, dropout=0.1)
                    elif modality == "video":
                        from models.video_models import get_video_model
                        model = get_video_model(model_name, num_classes=2, mode=mode)
                    elif modality == "graph":
                        if model_name == "Node2Vec":
                            model = "node2vec"
                        else:
                            from models.graph_models import get_graph_model
                            model = get_graph_model(model_name, num_classes=2, input_size=2, hidden_size=16, num_layers=1, dropout=0.1)
                    else:
                        model = None
                    results[key] = {"ok": True, "type": type(model).__name__}
                except Exception as exc:
                    results[key] = {"ok": False, "error": str(exc)}
    return results


def _test_training_runs(tmpdir: Path, text_fixture: Path, graph_fixture: Path, video_fixture: Path) -> dict[str, Any]:
    common = _common_args()
    runs = {
        "image_tinycnn": common | {
            "modality": "image",
            "data_path": str(ROOT / "fixtures" / "image_stress_dataset"),
            "training_mode": "from_scratch",
            "model_name": "TinyCNN",
            "task": "classification",
            "bundle_name": "smoke_image_tinycnn",
        },
        "text_rnn": common | {
            "modality": "text",
            "data_path": str(text_fixture),
            "text_col": "text",
            "label_col": "label",
            "training_mode": "from_scratch",
            "model_name": "RNN",
            "task": "classification",
            "batch_size": 4,
            "bundle_name": "smoke_text_rnn",
        },
        "tabular_rf": common | {
            "modality": "tabular",
            "data_path": str(ROOT / "fixtures" / "tabular_student_success.csv"),
            "label_col": "label",
            "training_mode": "from_scratch",
            "model_name": "RandomForest",
            "task": "classification",
            "feature_cols": ["study_hours", "attendance", "prior_grade", "participation_score"],
            "bundle_name": "smoke_tabular_rf",
        },
        "timeseries_linear_forecast": common | {
            "modality": "timeseries",
            "data_path": str(ROOT / "fixtures" / "timeseries_regression_stress.csv"),
            "label_col": "target",
            "time_col": "timestamp",
            "feature_cols": ["signal", "trend", "seasonality"],
            "training_mode": "from_scratch",
            "model_name": "LinearRegression",
            "task": "regression",
            "timeseries_forecast_mode": True,
            "timeseries_forecast_horizon": 1,
            "timeseries_lag_steps": 3,
            "timeseries_rolling_window": 3,
            "bundle_name": "smoke_timeseries_linear",
        },
        "audio_tinycnn": common | {
            "modality": "audio",
            "data_path": str(ROOT / "fixtures" / "audio_stress_dataset"),
            "training_mode": "from_scratch",
            "model_name": "TinyCNN",
            "task": "classification",
            "audio_verify_files": True,
            "batch_size": 4,
            "bundle_name": "smoke_audio_tinycnn",
        },
        "video_tinyr3d": common | {
            "modality": "video",
            "data_path": str(video_fixture),
            "training_mode": "from_scratch",
            "model_name": "TinyR3D",
            "task": "classification",
            "n_frames": 4,
            "batch_size": 1,
            "bundle_name": "smoke_video_tinyr3d",
        },
        "graph_node2vec": common | {
            "modality": "graph",
            "data_path": str(graph_fixture),
            "label_col": "label",
            "feature_cols": ["f1", "f2"],
            "training_mode": "from_scratch",
            "model_name": "Node2Vec",
            "task": "classification",
            "max_iter": 50,
            "bundle_name": "smoke_graph_node2vec",
        },
        "graph_gcn": common | {
            "modality": "graph",
            "data_path": str(graph_fixture),
            "label_col": "label",
            "feature_cols": ["f1", "f2"],
            "training_mode": "from_scratch",
            "model_name": "GCN",
            "task": "classification",
            "epochs": 3,
            "bundle_name": "smoke_graph_gcn",
        },
    }

    results = {}
    for name, kwargs in runs.items():
        results[name] = _drain_pipeline(**kwargs)
    return results


def _test_model_sweep_and_compare(text_fixture: Path) -> dict[str, Any]:
    common = _common_args()
    sweep_last = None
    for sweep in app.run_model_sweep(
        modality_val="tabular",
        data_path_val=str(ROOT / "fixtures" / "tabular_student_success.csv"),
        text_col_val="",
        label_col_val="label",
        feature_cols_val=["study_hours", "attendance", "prior_grade", "participation_score"],
        time_col_val="",
        window_size_val=12,
        n_frames_val=8,
        image_size_val=64,
        sample_rate_val=16000,
        audio_image_size_val=128,
        audio_n_mels_val=64,
        augmentation_val="none",
        val_split_val=0.2,
        subset_enabled_val=False,
        subset_percent_val=100.0,
        subset_seed_val=42,
        use_data_cleaning_val=False,
        tabular_missing_strategy_val="mean",
        tabular_clip_outliers_val=False,
        tabular_scaling_val="standard",
        text_lowercase_val=True,
        text_strip_urls_val=True,
        text_strip_punctuation_val=True,
        text_remove_stopwords_val=False,
        text_deduplicate_val=False,
        text_apply_stemming_val=False,
        text_apply_lemmatization_val=False,
        text_use_ngrams_val=False,
        timeseries_sort_by_time_val=True,
        timeseries_fill_strategy_val="forward_fill",
        timeseries_forecast_mode_val=False,
        timeseries_forecast_horizon_val=1,
        timeseries_lag_steps_val=3,
        timeseries_rolling_window_val=3,
        image_verify_files_val=True,
        image_aug_flip_val=False,
        image_aug_vertical_val=False,
        image_aug_rotation_val=False,
        image_aug_color_val=False,
        image_aug_gray_val=False,
        image_aug_perspective_val=False,
        image_normalization_val="imagenet",
        image_force_grayscale_val=False,
        audio_verify_files_val=False,
        audio_normalize_waveform_val=False,
        audio_aug_noise_val=False,
        audio_aug_shift_val=False,
        audio_aug_gain_val=False,
        audio_aug_time_mask_val=False,
        audio_aug_freq_mask_val=False,
        video_verify_files_val=False,
        training_mode_val="from_scratch",
        task_val="classification",
        epochs_val=1,
        lr_val=1e-3,
        batch_size_val=8,
        dropout_val=0.2,
        optimizer_val="adam",
        scheduler_val="none",
        use_amp_val=False,
        hidden_size_val=32,
        num_layers_val=1,
        n_estimators_val=10,
        max_depth_val=4,
        c_param_val=1.0,
        max_iter_val=100,
        lr_xgb_val=0.1,
        n_clusters_val=3,
        use_class_weights_val=False,
        checkpoint_every_val=0,
        bundle_name_val="smoke_sweep",
        sweep_models_val=["MLP", "RandomForest"],
        sweep_metrics_val=["accuracy", "f1"],
        sweep_sort_metric_val="accuracy",
        sweep_sort_order_val="descending",
    ):
        sweep_last = sweep
    if sweep_last is None:
        raise RuntimeError("run_model_sweep produced no updates")
    try:
        from training.run_comparison import history_dataframe
        history_df = history_dataframe()
        ids = history_df["id"].tail(2).astype(str).tolist()
        compare_result = app.compare_runs(",".join(ids))
        compare_ok = compare_result is not None and compare_result[1] is not None
    except Exception as exc:
        compare_ok = False
        ids = []
        sweep_last = (f"{sweep_last}\n\nCompare failed: {exc}", None)
    return {
        "sweep_summary": str(sweep_last[0])[:600],
        "sweep_rows": int(len(sweep_last[1])) if getattr(sweep_last[1], "__len__", None) else None,
        "compare_ids": ids,
        "compare_ok": compare_ok,
    }


def _test_exports_and_inference(training_runs: dict[str, Any], text_fixture: Path) -> dict[str, Any]:
    import pandas as pd
    from export.streamlit_generator import generate_streamlit_dashboard
    from ui.inference_helpers import (
        predict_audio,
        predict_image,
        predict_tabular,
        predict_text,
        predict_timeseries,
    )

    bundles = {name: details["bundle_path"] for name, details in training_runs.items()}
    image_bundle = bundles["image_tinycnn"]
    text_bundle = bundles["text_rnn"]
    tabular_bundle = bundles["tabular_rf"]
    timeseries_bundle = bundles["timeseries_linear_forecast"]
    audio_bundle = bundles["audio_tinycnn"]

    image_preds = predict_image(image_bundle, str(ROOT / "fixtures" / "image_stress_dataset" / "red_square" / "red_square_00.png"))
    text_preds = predict_text(text_bundle, "The explanations were clear and the project felt approachable.")
    tabular_preds = predict_tabular(
        tabular_bundle,
        {
            "study_hours": 11,
            "attendance_pct": 95,
            "practice_quizzes": 8,
            "project_score": 91,
        },
    )
    audio_preds = predict_audio(audio_bundle, str(ROOT / "fixtures" / "audio_stress_dataset" / "low_tone" / "low_tone_00.wav"))

    ts_df = pd.read_csv(ROOT / "fixtures" / "timeseries_regression_stress.csv").head(12)
    timeseries_input = ts_df[["sensor_a", "sensor_b", "sensor_c"]].to_dict(orient="records")
    timeseries_preds = predict_timeseries(timeseries_bundle, timeseries_input)

    streamlit_status = app.gen_streamlit(
        image_bundle,
        "image",
        "TinyCNN",
        "from_scratch",
        "classification",
        training_runs["image_tinycnn"]["eval_summary"],
    )
    fastapi_status = app.gen_fastapi(image_bundle, "image")
    docker_status = app.gen_docker(image_bundle, "image")
    batch_df, batch_status = app._batch_predict(image_bundle, str(ROOT / "fixtures" / "image_stress_dataset"))

    return {
        "predictions": {
            "image": image_preds[:3],
            "text": text_preds[:3],
            "tabular": tabular_preds[:3],
            "audio": audio_preds[:3],
            "timeseries": timeseries_preds[:3],
        },
        "exports": {
            "streamlit": str(streamlit_status)[:500],
            "fastapi": str(fastapi_status)[:500],
            "docker": str(docker_status)[:500],
            "batch_status": batch_status,
            "batch_rows": int(len(batch_df)) if batch_df is not None else 0,
        },
    }


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="full_smoke_", dir=ROOT / "outputs"))
    try:
        text_fixture = _make_text_fixture(tmpdir)
        graph_fixture = _make_graph_fixture(tmpdir)
        video_fixture = _make_video_fixture(tmpdir)

        training_runs = _test_training_runs(tmpdir, text_fixture, graph_fixture, video_fixture)
        report = {
            "previews": _test_preview_layer(text_fixture, graph_fixture),
            "walkthroughs": _test_walkthroughs(),
            "constructors": _test_model_constructors(),
            "training_runs": training_runs,
            "sweep_compare": _test_model_sweep_and_compare(text_fixture),
            "detection_annotation": _test_detection_and_annotation(tmpdir, text_fixture),
            "exports_inference": _test_exports_and_inference(training_runs, text_fixture),
        }
        print(json.dumps(report, indent=2, default=str))
    finally:
        pass


if __name__ == "__main__":
    main()
