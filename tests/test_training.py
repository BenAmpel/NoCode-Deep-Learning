import torch
from torch import nn
from torch.utils.data import TensorDataset

from config import set_seed
from training.cross_val import cross_validate
from training.trainer import _compute_loss


def test_regression_loss_flattens_instead_of_broadcasting():
    logits = torch.tensor([[1.0], [2.0], [3.0]])
    labels = torch.tensor([1.0, 2.0, 5.0])
    loss = _compute_loss(nn.MSELoss(), logits, labels, "regression")
    # (B, 1) vs (B,) broadcast to (B, B) would give a different, larger value.
    assert torch.isclose(loss, torch.tensor(4.0 / 3.0))


def test_regression_loss_handles_single_row_batch():
    loss = _compute_loss(nn.MSELoss(), torch.tensor([[2.0]]), torch.tensor([5.0]), "regression")
    assert torch.isclose(loss, torch.tensor(9.0))


def test_regression_cross_validation_learns_a_linear_target():
    set_seed(0)
    X = torch.randn(200, 3)
    y = X @ torch.tensor([1.5, -2.0, 0.5]) + 0.1 * torch.randn(200)
    dataset = TensorDataset(X, y)
    final = None
    for update in cross_validate(
        dataset,
        model_factory=lambda: nn.Linear(3, 1),
        k=3, epochs=30, lr=0.05, optimizer_name="adam", task="regression", batch_size=16,
    ):
        if update.get("done"):
            final = update
    assert final is not None
    # val_acc holds MAE for regression. With the (B, B) broadcast bug the model
    # can only learn the mean, leaving MAE around 1.9 on this data.
    assert final["mean_acc"] < 0.5


def test_set_seed_makes_runs_repeatable():
    set_seed(123)
    a = (torch.randn(5), nn.Linear(4, 2).weight.detach().clone())
    set_seed(123)
    b = (torch.randn(5), nn.Linear(4, 2).weight.detach().clone())
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
