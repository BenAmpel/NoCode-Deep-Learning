import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

from config import set_seed
from data_pipeline.splits import random_split, stratified_split
from eval.regression_metrics import format_regression_report
from models.target_scaling import TargetScaler, wrap_for_regression
from training.cross_val import cross_validate
from training.trainer import _make_scheduler, compute_class_weights


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_stratified_split_keeps_every_class_in_training():
    labels = ["common"] * 50 + ["rare"] * 2 + ["single"]
    train, val = stratified_split(labels, 0.2)
    assert sorted(train + val) == list(range(len(labels)))
    train_counts = Counter(labels[i] for i in train)
    assert set(train_counts) == {"common", "rare", "single"}
    assert Counter(labels[i] for i in val)["common"] == 10  # proportional share


def test_stratified_split_is_deterministic_and_leaves_global_random_alone():
    random.seed(123)
    expected_next = random.random()
    random.seed(123)
    labels = [i % 3 for i in range(30)]
    assert stratified_split(labels, 0.3) == stratified_split(labels, 0.3)
    assert random.random() == expected_next


def test_tiny_split_still_has_a_validation_row():
    train, val = stratified_split(["a", "a", "b"], 0.2)
    assert len(val) == 1 and len(train) == 2
    assert {["a", "a", "b"][i] for i in train} == {"a", "b"}  # borrowed from the class that can spare it


def test_all_singleton_classes_still_get_a_validation_row():
    train, val = stratified_split(["bus", "people"], 0.2)
    assert len(val) == 1 and len(train) == 1


def test_random_split_matches_previous_behaviour():
    indices = list(range(10))
    random.Random(42).shuffle(indices)
    assert random_split(10, 0.2) == (indices[2:], indices[:2])


def test_tabular_loader_keeps_rare_class_in_training(tmp_path):
    from modalities.tabular import load_tabular_data

    df = pd.DataFrame({"x": np.arange(23.0), "label": ["a"] * 20 + ["b"] * 2 + ["c"]})
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    _, _, classes, _, _, _, y_train, _, _ = load_tabular_data(str(path), label_col="label", augmentation="none")
    assert set(y_train.tolist()) == set(range(len(classes)))


# ---------------------------------------------------------------------------
# Class weights and schedulers
# ---------------------------------------------------------------------------

def test_class_weights_cover_all_classes_even_if_missing_from_training():
    weights = compute_class_weights([0, 0, 0, 1], num_classes=4)
    assert weights.shape == (4,)
    assert weights[2] == 1.0 and weights[3] == 1.0
    assert weights[1] > weights[0]  # rarer class weighs more


def test_warmup_cosine_survives_a_single_epoch():
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _make_scheduler(opt, "warmup_cosine", epochs=1, steps_per_epoch=5)
    opt.step()
    scheduler.step()  # used to raise ZeroDivisionError


# ---------------------------------------------------------------------------
# Regression target scaling
# ---------------------------------------------------------------------------

def test_target_scaler_outputs_original_units():
    inner = nn.Linear(3, 1)
    nn.init.zeros_(inner.weight)
    nn.init.zeros_(inner.bias)
    wrapped = TargetScaler(inner, mean=250.0, std=40.0)
    assert torch.allclose(wrapped(torch.randn(4, 3)), torch.full((4, 1), 250.0))


def test_constant_target_gets_unit_std():
    _, (mean, std) = wrap_for_regression(nn.Linear(1, 1), np.full(10, 7.0))
    assert mean == 7.0 and std == 1.0


def _train_mae(model, X, y, epochs=40):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X).reshape(-1), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((model(X).reshape(-1) - y).abs().mean())


def test_scaling_lets_a_network_fit_large_targets_quickly():
    set_seed(0)
    X = torch.randn(200, 3)
    y = 80 + 10 * (X @ torch.tensor([1.0, -0.5, 0.3]))  # scores around 80
    mlp = lambda: nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    set_seed(0)
    plain = _train_mae(mlp(), X, y)
    set_seed(0)
    scaled = _train_mae(wrap_for_regression(mlp(), y.numpy())[0], X, y)
    assert scaled < 5 < plain  # unscaled is still learning the offset


def test_cross_validation_scales_regression_targets_per_fold():
    set_seed(0)
    X = torch.randn(150, 3)
    y = 50 + 5 * (X @ torch.tensor([1.0, -1.0, 0.5]))  # offset of 50 to learn
    final = None
    for update in cross_validate(TensorDataset(X, y), model_factory=lambda: nn.Linear(3, 1),
                                 k=3, epochs=10, lr=0.05, optimizer_name="adam",
                                 task="regression", batch_size=16):
        if update.get("done"):
            final = update
    # Without per-fold target scaling a fresh Linear starts ~50 away and is
    # still far off after 10 epochs at this learning rate.
    assert final["mean_acc"] < 1.0


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def test_regression_report_says_validation_set():
    text = format_regression_report({"mae": 1.0, "mse": 1.0, "rmse": 1.0, "r2": 0.9, "mape": 2.0})
    assert "(validation set)" in text and "optimistic" in text
