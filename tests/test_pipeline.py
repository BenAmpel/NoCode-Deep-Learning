"""pipeline.runner.run raises on failure instead of reporting it as status text."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipeline import RUN_CONFIG_FIELDS, RunConfig
from pipeline.context import RunResult
from pipeline.runner import PreflightError, run

ROOT = Path(__file__).resolve().parents[1]
TABULAR = str(ROOT / "fixtures" / "tabular_student_success.csv")


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Keep runs from touching the real project state and run history."""
    import training.run_comparison as rc
    import ui.project_state as ps

    monkeypatch.setattr(ps, "STATE_FILE", tmp_path / "project_state.json")
    monkeypatch.setattr(rc, "HISTORY_FILE", tmp_path / "run_history.json")


def _config(**changes) -> RunConfig:
    base = dict.fromkeys(RUN_CONFIG_FIELDS, False)
    base.update(
        modality="tabular", data_path=TABULAR, text_col="", label_col="label",
        feature_cols=["study_hours", "attendance_pct", "practice_quizzes", "project_score"],
        time_col="", window_size=12, n_frames=8, image_size=64, sample_rate=16000,
        audio_image_size=128, audio_n_mels=64, augmentation="none", val_split=0.2,
        subset_percent=100.0, subset_seed=42, tabular_missing_strategy="mean",
        tabular_scaling="standard", timeseries_fill_strategy="none",
        timeseries_forecast_horizon=1, timeseries_lag_steps=3, timeseries_rolling_window=3,
        image_normalization="imagenet", training_mode="from_scratch", model_name="MLP",
        task="classification", epochs=1, lr=1e-3, batch_size=8, dropout=0.2,
        optimizer="adam", scheduler_name="none", hidden_size=32, num_layers=1,
        n_estimators=10, max_depth=4, C_param=1.0, max_iter=100, lr_xgb=0.1,
        n_clusters=3, checkpoint_every=0, bundle_name="pytest_pipeline",
    )
    base.update(changes)
    return RunConfig(**base)


def test_preflight_problems_raise():
    with pytest.raises(PreflightError) as info:
        list(run(_config(data_path="", model_name="")))
    assert len(info.value.errors) == 2


def test_stage_failure_raises_and_removes_temp_files(monkeypatch):
    created: list[str] = []
    import pipeline.runner as runner

    real_cleanup = runner.cleanup

    def spy_cleanup(ctx):
        created.extend(ctx.temp_files)
        real_cleanup(ctx)

    monkeypatch.setattr(runner, "cleanup", spy_cleanup)
    # Cleaning writes a temporary CSV; the missing label column then fails loading.
    with pytest.raises(Exception) as info:
        list(run(_config(label_col="no_such_column", use_data_cleaning=True)))
    assert not isinstance(info.value, PreflightError)
    assert created and not any(os.path.exists(p) for p in created)


def test_successful_run_ends_with_a_result(tmp_path, monkeypatch):
    import pipeline.runner as runner

    bundles = tmp_path / "bundles"
    monkeypatch.setattr(runner, "OUTPUTS_ROOT", bundles)
    items = list(run(_config(model_name="RandomForest")))
    assert isinstance(items[-1], RunResult)
    assert not any(isinstance(i, RunResult) for i in items[:-1])
    assert Path(items[-1].bundle_path).parent == bundles
