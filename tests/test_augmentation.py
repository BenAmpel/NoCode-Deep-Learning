import pickle

import numpy as np
import pytest
import torch

from data_pipeline.augmentation import (
    AUGMENTATION_LEVELS,
    get_audio_transforms,
    get_timeseries_augmentation,
)


def test_timeseries_none_is_identity():
    aug = get_timeseries_augmentation("none")
    window = np.ones((10, 3), dtype=np.float32)
    for _ in range(20):
        np.testing.assert_array_equal(aug(window), window)


@pytest.mark.parametrize("level", ["light", "medium", "heavy"])
def test_timeseries_levels_stay_close_to_input(level):
    np.random.seed(0)
    aug = get_timeseries_augmentation(level)
    window = np.ones((50, 2), dtype=np.float32)
    out = aug(window)
    assert out.shape == window.shape
    assert out.dtype == window.dtype
    # Heaviest setting uses a 0.2 scale sigma; a sign flip would mean a bug.
    assert np.all(out > 0)


@pytest.mark.parametrize("level", AUGMENTATION_LEVELS)
def test_audio_transforms_are_picklable(level):
    # DataLoader workers use spawn on macOS/Windows, which pickles the dataset.
    for tf in get_audio_transforms(level):
        pickle.loads(pickle.dumps(tf))


@pytest.mark.parametrize("level", AUGMENTATION_LEVELS)
def test_audio_transforms_preserve_shape(level):
    torch.manual_seed(0)
    wav_aug, wav_val, spec_aug, spec_val = get_audio_transforms(level)
    waveform = torch.rand(1, 16000) * 0.5
    spec = torch.rand(1, 64, 100)
    assert wav_aug(waveform).shape == waveform.shape
    assert spec_aug(spec).shape == spec.shape
    assert torch.equal(wav_val(waveform), waveform)
    assert torch.equal(spec_val(spec), spec)
