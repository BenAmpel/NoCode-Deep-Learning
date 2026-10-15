import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from export.bundle_runtime import Bundle, UntrustedModelError
from training.checkpointer import load_checkpoint, save_checkpoint


class _RunsCode:
    """Unpickling this calls os.system: what a malicious file would do."""
    def __reduce__(self):
        import os
        return (os.system, ("echo pwned",))


def test_checkpoints_round_trip_with_weights_only(tmp_path):
    model = nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters())
    history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "val_acc": 70.0}]
    path = save_checkpoint(model, opt, 1, history, 0.6, str(tmp_path))
    fresh = nn.Linear(3, 2)
    loaded, _, epoch, hist, val_loss = load_checkpoint(path, fresh, torch.optim.Adam(fresh.parameters()))
    assert epoch == 1 and hist == history and val_loss == pytest.approx(0.6)
    assert torch.equal(loaded.weight.cpu(), model.weight)


def test_checkpoint_with_pickled_code_is_refused(tmp_path):
    path = tmp_path / "model_epoch001.pt"
    torch.save({"model_state": {}, "payload": _RunsCode()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(str(path), nn.Linear(3, 2))


def _joblib_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 0, 1, 1])
    joblib.dump(model, root / "model.joblib")
    (root / "labels.json").write_text(json.dumps({"0": "no", "1": "yes"}))
    (root / "preprocessing.json").write_text(json.dumps({
        "modality": "tabular", "numeric_cols": ["x"], "scaling_strategy": "none",
        "categorical_cols": [], "dummy_feature_names": [],
    }))
    return root


def test_untrusted_joblib_bundle_is_not_unpickled(tmp_path):
    bundle = _joblib_bundle(tmp_path / "shared_bundle")
    with pytest.raises(UntrustedModelError, match="pickle"):
        Bundle(bundle).predict({"x": 2.5})
    assert Bundle(bundle, trust_pickle=True).predict({"x": 2.5})["predictions"][0]["label"] == "yes"


def test_app_only_trusts_bundles_in_its_outputs_folder(tmp_path):
    from ui import inference_helpers as ih

    outside = _joblib_bundle(tmp_path / "downloaded_bundle")
    result = ih.predict_tabular(str(outside), {"x": 2.5})
    assert result[0]["label"] == "Error" and "pickle" in result[0]["error"]
