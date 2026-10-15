import pytest
import torch

from config import DEVICE
from models.text_models import PretrainedTransformerClassifier

# A ~1 MB randomly initialised DistilBERT published for library tests; it
# exercises the same attention code path as the real checkpoints.
TINY_REPO = "hf-internal-testing/tiny-random-DistilBertModel"


def test_pretrained_transformer_trains_on_current_device():
    try:
        model = PretrainedTransformerClassifier(num_classes=2, repo_id=TINY_REPO).to(DEVICE)
    except OSError as exc:  # offline
        pytest.skip(f"could not download {TINY_REPO}: {exc}")
    model.train()  # dropout active: SDPA on MPS raised NotImplementedError here
    input_ids = torch.randint(0, 100, (2, 8), device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, 2)
    logits.sum().backward()
