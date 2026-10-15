import threading

import matplotlib
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training import cancel  # noqa: E402
from ui import project_state as ps  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_stop_flag():
    cancel.reset()
    yield
    cancel.reset()


def _loader():
    X, y = torch.randn(64, 3), torch.randint(0, 2, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

def test_stop_interrupts_train_pytorch_between_batches():
    from training.trainer import train_pytorch

    seen = []

    class Model(nn.Linear):
        def forward(self, x):
            seen.append(1)
            if len(seen) == 2:
                cancel.request_stop()  # pressed during the second batch
            return super().forward(x)

    with pytest.raises(cancel.TrainingStopped):
        for _ in train_pytorch(Model(3, 2), _loader(), _loader(), epochs=5, lr=0.01, device="cpu"):
            pass
    assert len(seen) == 2  # no further batches ran


def test_stop_interrupts_cross_validation():
    from training.cross_val import cross_validate

    cancel.request_stop()
    X, y = torch.randn(30, 3), torch.randint(0, 2, (30,))
    with pytest.raises(cancel.TrainingStopped):
        for _ in cross_validate(TensorDataset(X, y), model_factory=lambda: nn.Linear(3, 2),
                                k=3, epochs=2, lr=0.01, optimizer_name="adam", task="classification"):
            pass


# ---------------------------------------------------------------------------
# Project state
# ---------------------------------------------------------------------------

def test_saves_merge_instead_of_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "STATE_FILE", tmp_path / "project_state.json")
    ps.save_project_state({"recommendation": {"model": "MLP"}})
    ps.save_project_state({"dataset_path": "/data.csv"})  # e.g. preview_dataset
    assert ps.load_project_state() == {"recommendation": {"model": "MLP"}, "dataset_path": "/data.csv"}
    ps.save_project_state({"only": 1}, replace=True)
    assert ps.load_project_state() == {"only": 1}


def test_concurrent_saves_lose_no_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "STATE_FILE", tmp_path / "project_state.json")
    threads = [threading.Thread(target=ps.save_project_state, args=({f"k{i}": i},)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ps.load_project_state() == {f"k{i}": i for i in range(20)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_loss_plot_uses_no_shared_pyplot_figure():
    import app

    before = set(plt.get_fignums())
    history = [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "val_acc": 50.0},
               {"epoch": 2, "train_loss": 0.8, "val_loss": 0.9, "val_acc": 60.0}]
    a, b = app._loss_plot(history), app._loss_plot(history)
    assert a is not b  # two runs never draw into the same figure
    assert set(plt.get_fignums()) == before


def test_prediction_chart_leaves_no_open_figures():
    from ui.inference_helpers import predictions_to_chart

    before = set(plt.get_fignums())
    fig = predictions_to_chart([{"label": "a", "confidence": 0.7, "rank": 1},
                                {"label": "b", "confidence": 0.3, "rank": 2}])
    assert fig is not None
    assert set(plt.get_fignums()) == before
